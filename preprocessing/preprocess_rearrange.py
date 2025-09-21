"""Preprocessing pipeline for the rearrange manipulation dataset."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from preprocessing.preprocess_base import (
    PreprocessDataset,
    Sample,
    TrackProcessor,
    run_dataset,
)


class PreprocessRearrange(PreprocessDataset):
    """Generate CoTracker tracks for the rearrange dataset."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        mode = getattr(cfg, "mode", "tracks")
        if mode != "tracks":
            raise ValueError("PreprocessRearrange currently only supports mode='tracks'")

        reinit_suffix = f"_reinit_{cfg.horizon}" if getattr(cfg, "reinit", True) else ""
        self.extension = f"{cfg.init_queries}_{cfg.n_tracks}{reinit_suffix}"

        source_dir = Path(cfg.source_dir)
        if not source_dir.is_absolute():
            source_dir = Path(os.getcwd()) / source_dir
        self.source_dir = source_dir

        dest_dir = Path(cfg.dest_dir)
        if not dest_dir.is_absolute():
            dest_dir = Path(os.getcwd()) / dest_dir
        self.dest_dir = dest_dir

        self.dataset_name = str(cfg.dataset_name)
        self.view_name = str(getattr(cfg, "view_name", "agentview"))

        metadata_path = self.source_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found at {metadata_path}")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        episodes: List[Dict[str, Any]] = metadata.get("episodes", [])
        n_rollout = getattr(cfg, "n_rollout", None)
        if n_rollout is not None:
            episodes = episodes[: int(n_rollout)]
        if not episodes:
            raise ValueError("No episodes found in rearrange metadata")
        self.episodes = episodes

    # ------------------------------------------------------------------
    # PreprocessDataset abstract implementations
    # ------------------------------------------------------------------
    def build_models(self, cfg: DictConfig) -> Dict[str, Any]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "mps":
            device = torch.device("cpu")
        if cfg.reinit:
            cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
        else:
            cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
        return {"cotracker": cotracker.eval().to(device)}

    def build_processors(self, cfg: DictConfig, models: Dict[str, Any]) -> Dict[str, TrackProcessor]:
        return {
            "tracks": TrackProcessor(
                model=models["cotracker"],
                init_queries=cfg.init_queries,
                reinit=cfg.reinit,
                horizon=cfg.horizon,
                n_tracks=cfg.n_tracks,
                batch_size=cfg.batch_size,
            )
        }

    def iter_items(self, cfg: DictConfig) -> Iterable[Dict[str, Any]]:
        for ep in self.episodes:
            ep_id = int(ep["episode"])
            ep_dir = self.source_dir / "episodes" / f"ep_{ep_id:04d}"
            obs_path = ep_dir / "obs.npy"
            if not obs_path.exists():
                raise FileNotFoundError(f"Observation file not found: {obs_path}")
            yield {"episode": ep_id, "obs_path": obs_path, "meta": dict(ep)}

    def to_sample(self, item: Dict[str, Any], cfg: DictConfig) -> Sample:
        obs = np.load(item["obs_path"])
        if obs.ndim != 4:
            raise ValueError(f"Expected obs.npy to have shape (T, H, W, C), got {obs.shape}")
        sample_id = f"ep_{int(item['episode']):04d}"
        videos = {self.view_name: obs.astype(np.uint8)}
        return Sample(id=sample_id, videos=videos, meta=item["meta"])

    def output_path(self, sample: Sample, cfg: DictConfig) -> str:
        base = self.dest_dir / self.dataset_name / self.extension
        return str(base / f"{sample.id}.hdf5")


@hydra.main(config_path="../cfg/preprocessing", config_name="preprocess_rearrange", version_base="1.2")
def main(cfg: DictConfig) -> None:
    save_dir = Path(os.getcwd()) / cfg.dest_dir / cfg.dataset_name
    save_dir = save_dir / (
        f"{cfg.init_queries}_{cfg.n_tracks}{'_reinit_' + str(cfg.horizon) if cfg.reinit else ''}"
    )
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, str(save_dir / "config.yaml"))

    dataset = PreprocessRearrange(cfg)
    run_dataset(dataset, cfg)
    print("Done!")


if __name__ == "__main__":
    main()
