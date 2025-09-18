"""
Preprocessing Core v2 — minimal, dataset-agnostic API.

How to use
- Subclass PreprocessDataset and implement these methods:
  - build_models(cfg) -> Dict[str, Any]
  - build_processors(cfg, models) -> Dict[str, Processor]
  - iter_items(cfg) -> Iterable[Any]
  - to_sample(item, cfg) -> Sample
  - output_path(sample, cfg) -> str (absolute or workspace‑relative output .hdf5 path)

- Then call run_dataset(defn, cfg) to iterate items, convert to Samples, and write
  per‑sample HDF5 files using the provided processors.

This module makes no dataset assumptions; all specifics live in the subclass.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import h5py
import numpy as np
from einops import rearrange

from amplify.utils.preprocessing_utils import inital_save_h5, tracks_from_video


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

@dataclass
class Sample:
    """A single training example to write.

    Required
    - id: Unique identifier used in output file naming
    - videos: dict mapping view -> THWC uint8/float32 arrays

    Optional
    - text: str instruction for this sample
    - actions: np.ndarray action sequence for this sample
    - meta: arbitrary metadata
    """

    id: str
    videos: Dict[str, np.ndarray]
    text: Optional[str] = None
    actions: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None


class Processor(ABC):
    """Writes a modality for one Sample into an opened HDF5 file."""

    @abstractmethod
    def process(self, out_h5: h5py.File, sample: Sample) -> None:
        ...


class TrackProcessor(Processor):
    """Computes dense tracks per view and writes root/<view>/{tracks,vis}."""

    def __init__(self, model, init_queries: str = "uniform", reinit: bool = True, horizon: int = 16, n_tracks: int = 400, batch_size: int = 8):
        self.model = model
        self.init_queries = init_queries
        self.reinit = reinit
        self.horizon = horizon
        self.n_tracks = n_tracks
        self.batch_size = batch_size

    def process(self, out_h5: h5py.File, sample: Sample) -> None:
        root = out_h5.create_group("root") if "root" not in out_h5 else out_h5["root"]
        for view, video_thwc in sample.videos.items():
            video_tchw = rearrange(video_thwc, "t h w c -> t c h w")
            tracks, vis = tracks_from_video(
                video=video_tchw,
                track_model=self.model,
                init_queries=self.init_queries,
                reinit=self.reinit,
                horizon=self.horizon,
                n_tracks=self.n_tracks,
                batch_size=self.batch_size,
                dim_order="tchw",
            )
            vg = root.create_group(view) if view not in root else root[view]
            if "tracks" in vg:
                vg.__delitem__("tracks")
            if "vis" in vg:
                vg.__delitem__("vis")
            vg.create_dataset("tracks", data=tracks, dtype="float32")
            vg.create_dataset("vis", data=vis, dtype="float32")


class DepthProcessor(Processor):
    """Runs a provided depth function per frame and writes root/<view>/depth.

    Note: Depth is not used by AMPLIFY training by default; this is provided
    for completeness and for users who want to export this modality.
    """

    def __init__(self, depth_fn):
        """depth_fn: Callable[[np.ndarray (T,H,W,C)], np.ndarray (T,H,W)]"""
        self.depth_fn = depth_fn

    def process(self, out_h5: h5py.File, sample: Sample) -> None:
        root = out_h5.create_group("root") if "root" not in out_h5 else out_h5["root"]
        for view, video_thwc in sample.videos.items():
            depth = self.depth_fn(video_thwc)
            vg = root.create_group(view) if view not in root else root[view]
            if "depth" in vg:
                vg.__delitem__("depth")
            vg.create_dataset("depth", data=depth, dtype="float32")


class TextEmbeddingProcessor(Processor):
    """Writes top-level text_emb for a sample (if sample.text is set)."""

    def __init__(self, text_encoder):
        self.text_encoder = text_encoder

    def process(self, out_h5: h5py.File, sample: Sample) -> None:
        if not sample.text:
            return
        emb = self.text_encoder([sample.text]).cpu().numpy()
        if "text_emb" in out_h5:
            out_h5.__delitem__("text_emb")
        out_h5.create_dataset("text_emb", data=emb, dtype="float32")

class PreprocessDataset(ABC):
    """Implement these five methods to define your dataset preprocessing."""

    @abstractmethod
    def build_models(self, cfg) -> Dict[str, Any]:
        """Create and return any models needed by processors (e.g., trackers)."""
        ...

    @abstractmethod
    def build_processors(self, cfg, models: Dict[str, Any]) -> Dict[str, Processor]:
        """Return a dict of processors, e.g., {"tracks": TrackProcessor(...)}."""
        ...

    @abstractmethod
    def iter_items(self, cfg) -> Iterable[Any]:
        """Yield raw items to be converted into Samples (e.g., file paths/keys)."""
        ...

    @abstractmethod
    def to_sample(self, item: Any, cfg) -> Sample:
        """Map a raw item to a Sample (id, videos, optional text/actions/meta)."""
        ...

    @abstractmethod
    def output_path(self, sample: Sample, cfg) -> str:
        """Return the full output .hdf5 file path for this sample."""
        ...

    # Optional hooks
    def on_begin(self, cfg, models: Dict[str, Any]) -> None:
        pass

    def on_end(self, cfg, models: Dict[str, Any]) -> None:
        pass


def _open_outfile(save_path: str, view_names: Iterable[str], skip_exist: bool) -> Optional[h5py.File]:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return inital_save_h5(save_path, skip_exist, view_names=list(view_names))


def run_dataset(defn: PreprocessDataset, cfg) -> None:
    """Run preprocessing: build models, iterate items, write outputs."""
    models = defn.build_models(cfg)
    processors = defn.build_processors(cfg, models)
    defn.on_begin(cfg, models)

    try:
        for item in defn.iter_items(cfg):
            sample = defn.to_sample(item, cfg)
            save_path = defn.output_path(sample, cfg)
            out_h5 = _open_outfile(save_path, sample.videos.keys(), getattr(cfg, "skip_exist", True))
            if out_h5 is None:
                continue
            try:
                for proc in processors.values():
                    proc.process(out_h5, sample)
            finally:
                out_h5.close()
    finally:
        defn.on_end(cfg, models)
