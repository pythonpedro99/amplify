import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from amplify.loaders.base_dataset import BaseDataset
from amplify.utils.data_utils import normalize_traj

try:  # pragma: no cover - optional dependency
    import h5py
except ImportError:  # pragma: no cover - optional dependency
    h5py = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    import h5py as _h5py


@dataclass
class _FilterCfg:
    enabled: bool
    n_slices: Optional[int]
    seed: int


class RearrangeDataset(BaseDataset):
    """Dataset loader for the rearrange manipulation benchmark.

    The dataset is organised as::

        <data_path>/metadata.json
        <data_path>/episodes/ep_0000/actions.npy
        <data_path>/episodes/ep_0000/obs.npy

    ``metadata.json`` stores a list of episode dictionaries containing at
    minimum ``{"episode": int, "n_actions": int}``.  ``actions.npy`` holds the
    action sequence for the episode while ``obs.npy`` stores the RGB
    observations.  Each dataset sample corresponds to a fixed number of frames
    extracted from an episode with optional frame skipping.  The loader mirrors
    the behaviour of :class:`LiberoDataset` so that it can be used with the
    existing training utilities (e.g. :mod:`train_inverse_dynamics`).
    """

    def __init__(
        self,
        *,
        root_dir: str,
        dataset_name: str,
        split: str,
        keys_to_load: List[str],
        img_shape: List[int],
        true_horizon: int,
        cfg: Optional[Dict],
        fraction: float,
        aug_cfg: Optional[Dict] = None,
    ) -> None:
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        self.cfg = cfg or {}

        self.track_keys = [k for k in keys_to_load if k in ["tracks", "vis"]]

        self.dataset_name = dataset_name
        self.split = split if split in {"train", "val", "valid"} else "train"
        self.img_shape = tuple(int(x) for x in img_shape)
        self.slice_hist = int(self.cfg.get("num_hist", 3))
        self.slice_pred = int(self.cfg.get("num_pred", 1))
        self.frameskip = int(self.cfg.get("frameskip", 1))
        if self.slice_hist < 0 or self.slice_pred < 0:
            raise ValueError("num_hist and num_pred must be non-negative")
        self.slice_len = self.slice_hist + self.slice_pred
        if self.slice_len <= 0:
            raise ValueError("Slice length must be positive")
        self.slice_span = self.frameskip * (self.slice_len - 1) + 1

        track_method = str(self.cfg.get("track_method", "uniform_400_reinit_16"))
        cameras_cfg = self.cfg.get("cond_cameraviews")
        if cameras_cfg is None:
            cameras_cfg = ["agentview"]
        self._configured_cameras = tuple(cameras_cfg)

        data_path = self.cfg.get("data_path")
        if data_path is None:
            raise ValueError("rearrange_dataset.data_path must be set")
        data_path = os.path.expanduser(str(data_path))
        if not os.path.isabs(data_path):
            data_path = os.path.join(root_dir, data_path)
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Rearrange dataset path does not exist: {self.data_path}")

        preprocessed_dir_cfg = self.cfg.get("preprocessed_dir")
        self.preprocessed_dir: Optional[Path] = None
        if preprocessed_dir_cfg is not None:
            preprocessed_dir = Path(os.path.expanduser(str(preprocessed_dir_cfg)))
            if not preprocessed_dir.is_absolute():
                preprocessed_dir = Path(root_dir) / preprocessed_dir
            self.preprocessed_dir = preprocessed_dir
        if self.track_keys:
            if self.preprocessed_dir is None:
                raise ValueError(
                    "rearrange_dataset.preprocessed_dir must be set when requesting tracks"
                )
            if not self.preprocessed_dir.exists():
                raise FileNotFoundError(
                    f"Preprocessed track directory not found: {self.preprocessed_dir}"
                )

        metadata_path = self.data_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found at {metadata_path}")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        episodes = metadata.get("episodes", [])
        n_rollout = self.cfg.get("n_rollout")
        if n_rollout is not None:
            episodes = episodes[: int(n_rollout)]
        if len(episodes) == 0:
            raise ValueError("Rearrange dataset metadata contains no episodes")
        self.episodes_meta = episodes

        self.obs_paths: List[Path] = []
        self.actions: List[torch.Tensor] = []
        self.seq_lengths: List[int] = []
        for ep in episodes:
            ep_id = int(ep["episode"])
            ep_dir = self.data_path / "episodes" / f"ep_{ep_id:04d}"
            action_arr = np.load(ep_dir / "actions.npy")
            self.actions.append(torch.as_tensor(action_arr, dtype=torch.float32))
            self.obs_paths.append(ep_dir / "obs.npy")
            self.seq_lengths.append(int(ep.get("n_actions", action_arr.shape[0])))

        if len(self.actions) == 0:
            raise ValueError("No action sequences found in the rearrange dataset")

        action_example = self.actions[0]
        self.action_dim = int(action_example.shape[-1] if action_example.ndim > 1 else 1)

        if self.obs_paths:
            obs_arr = np.load(self.obs_paths[0], mmap_mode="r")
            self.data_img_size = tuple(int(x) for x in obs_arr.shape[1:3])
        else:
            self.data_img_size = self.img_shape

        self.normalize_action = bool(self.cfg.get("normalize_action", False))
        if self.normalize_action:
            all_actions = torch.cat(self.actions, dim=0)
            self.action_mean = all_actions.mean(dim=0)
            self.action_std = all_actions.std(dim=0) + 1e-6
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
        self.action_mean_np = self.action_mean.cpu().numpy()
        self.action_std_np = self.action_std.cpu().numpy()

        split_ratio = float(self.cfg.get("split_ratio", 0.9))
        split_ratio = min(max(split_ratio, 0.0), 1.0)
        num_eps = len(self.seq_lengths)
        split_idx = int(math.floor(num_eps * split_ratio))
        if self.split in {"train"}:
            candidate_indices = list(range(split_idx)) or list(range(num_eps))
        else:
            candidate_indices = list(range(split_idx, num_eps)) or list(range(num_eps))

        if len(candidate_indices) == 0:
            raise ValueError(f"No episodes available for split '{split}'")

        self.fraction = float(fraction)
        self.selected_indices = self._apply_fraction(candidate_indices, self.fraction)
        if len(self.selected_indices) == 0:
            raise ValueError("Fraction selection produced an empty dataset")

        filter_actions = self.cfg.get("filter_actions", [4, 5])
        self.filter_actions = {int(a) for a in filter_actions}
        self.filter_cfg = {
            "train": _FilterCfg(
                enabled=bool(self.cfg.get("filter_train", False)),
                n_slices=self.cfg.get("n_slices_train"),
                seed=int(self.cfg.get("seed_train", 42)),
            ),
            "val": _FilterCfg(
                enabled=bool(self.cfg.get("filter_val", False)),
                n_slices=self.cfg.get("n_slices_val"),
                seed=int(self.cfg.get("seed_val", 99)),
            ),
        }

        self._obs_cache: Dict[int, np.ndarray] = {}
        self._track_file_cache: Dict[int, Path] = {}
        self._track_index_built = False
        self._track_dir_candidates: Optional[List[Path]] = None

        super().__init__(
            root_dir=root_dir,
            dataset_names=[dataset_name],
            track_method=track_method,
            cond_cameraviews=self._configured_cameras,
            keys_to_load=list(keys_to_load),
            img_shape=self.img_shape,
            true_horizon=int(true_horizon),
            track_pred_horizon=int(true_horizon),
            interp_method="linear",
            num_tracks=0,
            use_cached_index_map=bool(self.cfg.get("use_cached_index_map", False)),
            aug_cfg=aug_cfg,
        )

        self.track_keys = [k for k in self.keys_to_load if k in ["tracks", "vis"]]

    # ------------------------------------------------------------------
    # BaseDataset abstract implementations
    # ------------------------------------------------------------------
    def get_cache_file(self) -> str:
        cache_dir = os.path.expanduser("~/.cache/amplify/index_maps/rearrange")
        os.makedirs(cache_dir, exist_ok=True)
        frac_tag = f"{self.fraction:+.2f}".replace(".", "p")
        filter_cfg = self.filter_cfg["train" if self.split == "train" else "val"]
        filter_tag = "filtered" if filter_cfg.enabled else "all"
        fname = (
            f"{self.dataset_name}_{self.split}_{frac_tag}_"
            f"len{self.slice_len}_fs{self.frameskip}_{filter_tag}.json"
        )
        return os.path.join(cache_dir, fname)

    def create_index_map(self) -> List[Dict]:
        index_map: List[Dict] = []
        missing_tracks = 0
        for epi_idx in self.selected_indices:
            seq_len = self.seq_lengths[epi_idx]
            if seq_len < self.slice_span:
                continue

            track_path: Optional[Path] = None
            if self.track_keys:
                track_path = self._find_track_path(epi_idx)
                if track_path is None:
                    missing_tracks += 1
                    continue

            max_start = seq_len - self.slice_span + 1
            for start in range(max_start):
                frames = list(range(start, start + self.slice_span, self.frameskip))
                entry = {
                    "episode_idx": int(epi_idx),
                    "start_t": int(start),
                    "end_t": int(frames[-1] + 1),
                    "frames": frames,
                    "rollout_len": int(seq_len),
                }
                if track_path is not None:
                    entry["track_path"] = str(track_path)
                index_map.append(entry)

        filter_cfg = self.filter_cfg["train" if self.split == "train" else "val"]
        if filter_cfg.enabled:
            if filter_cfg.n_slices is None:
                raise ValueError("n_slices must be provided when filtering is enabled")
            index_map = self._filter_entries(index_map, filter_cfg.n_slices, filter_cfg.seed)

        if self.track_keys and missing_tracks:
            print(
                f"[RearrangeDataset] Skipped {missing_tracks} episode(s) without preprocessed tracks"
            )

        return index_map

    def _get_track_dir_candidates(self) -> List[Path]:
        if self.preprocessed_dir is None:
            return []
        if not hasattr(self, "_track_dir_candidates") or self._track_dir_candidates is None:
            candidates: List[Path] = []
            seen: set = set()

            dataset_dir = self.dataset_name if self.dataset_name else None
            data_path_name = self.data_path.name if self.data_path.name else None
            track_method = getattr(self, "track_method", None)
            split = self.split if self.split else None

            segments: List[Iterable[Optional[str]]] = [
                (dataset_dir, track_method, split),
                (data_path_name, track_method, split),
                (dataset_dir, split, track_method),
                (data_path_name, split, track_method),
                (track_method, split),
                (dataset_dir, track_method),
                (data_path_name, track_method),
                (track_method,),
                (dataset_dir, split),
                (data_path_name, split),
                (split,),
                (dataset_dir,),
                (data_path_name,),
                tuple(),
            ]

            for parts in segments:
                parts = tuple(p for p in parts if p)
                path = self.preprocessed_dir.joinpath(*parts) if parts else self.preprocessed_dir
                if path in seen:
                    continue
                seen.add(path)
                if path.exists():
                    candidates.append(path)

            self._track_dir_candidates = candidates
        return list(self._track_dir_candidates)

    def _extract_episode_basenames(self, epi_idx: int) -> List[str]:
        meta = self.episodes_meta[epi_idx] if epi_idx < len(self.episodes_meta) else {}
        base_names: List[str] = []

        for key in [
            "track_file",
            "track_path",
            "episode_name",
            "file",
            "path",
        ]:
            value = meta.get(key)
            if isinstance(value, str) and value:
                base_names.append(Path(value).stem)

        ep_id = int(meta.get("episode", epi_idx))
        base_names.append(f"ep_{ep_id:04d}")
        base_names.append(str(ep_id))

        unique: List[str] = []
        seen = set()
        for name in base_names:
            if name not in seen:
                seen.add(name)
                unique.append(name)
        return unique

    def _build_track_index(self) -> None:
        if self.preprocessed_dir is None or self._track_index_built:
            return
        self._track_index_built = True

        priority_cache: Dict[int, tuple[int, Path]] = {}
        for ext in (".hdf5", ".h5", ".npz"):
            for path in self.preprocessed_dir.rglob(f"*{ext}"):
                if not path.is_file():
                    continue
                track_method = getattr(self, "track_method", None)
                if track_method and track_method not in path.parts:
                    continue
                match = re.search(r"ep[_-]?(\d+)", path.stem)
                if not match:
                    continue
                ep_id = int(match.group(1))
                priority = 0 if self.split and self.split in path.parts else 1
                existing = priority_cache.get(ep_id)
                if existing is None or priority < existing[0]:
                    priority_cache[ep_id] = (priority, path)

        for ep_id, (_, path) in priority_cache.items():
            self._track_file_cache.setdefault(ep_id, path)

    def _find_track_path(self, epi_idx: int) -> Optional[Path]:
        if self.preprocessed_dir is None:
            return None

        meta = self.episodes_meta[epi_idx] if epi_idx < len(self.episodes_meta) else {}
        ep_id = int(meta.get("episode", epi_idx))
        cached = self._track_file_cache.get(ep_id)
        if cached is not None and cached.exists():
            return cached
        elif cached is not None and not cached.exists():
            del self._track_file_cache[ep_id]

        base_names = self._extract_episode_basenames(epi_idx)
        candidates = self._get_track_dir_candidates()
        for name in base_names:
            for directory in candidates:
                for ext in (".hdf5", ".h5", ".npz"):
                    candidate = directory / f"{name}{ext}"
                    if candidate.exists():
                        self._track_file_cache[ep_id] = candidate
                        return candidate

        self._build_track_index()
        cached = self._track_file_cache.get(ep_id)
        if cached is not None and cached.exists():
            return cached
        return None

    def load_images(self, idx_dict: Dict) -> Dict:
        episode_idx = idx_dict["episode_idx"]
        frames = idx_dict["frames"]
        obs = self._load_obs_episode(episode_idx)
        frame = obs[frames[0]].astype(np.float32)
        if frame.ndim != 3:
            raise ValueError(f"Unexpected obs shape {frame.shape} for episode {episode_idx}")
        frame = np.expand_dims(frame, axis=0)  # (1, H, W, C)
        return {"images": frame}

    def load_actions(self, idx_dict: Dict) -> Dict:
        episode_idx = idx_dict["episode_idx"]
        frames = idx_dict["frames"]
        actions = self.actions[episode_idx][frames]
        actions_np = actions.cpu().numpy().astype(np.float32)
        return {"actions": actions_np}

    def load_proprioception(self, idx_dict: Dict) -> Dict:
        return {}

    def load_tracks(self, idx_dict: Dict) -> Dict:
        track_path_str = idx_dict.get("track_path")
        if not track_path_str:
            return {}

        track_path = Path(track_path_str)
        if not track_path.exists():
            raise FileNotFoundError(f"Track file not found: {track_path}")

        frames: List[int] = idx_dict["frames"]
        if not frames:
            return {}
        start_t = frames[0]
        offsets = [f - start_t for f in frames]
        if any(offset < 0 for offset in offsets):
            raise ValueError(f"Invalid frame offsets computed from frames={frames}")

        if track_path.suffix.lower() in {".h5", ".hdf5"}:
            loader = self._load_tracks_from_hdf5
        elif track_path.suffix.lower() == ".npz":
            loader = self._load_tracks_from_npz
        else:
            raise ValueError(f"Unsupported track file format: {track_path.suffix}")

        return loader(track_path, frames, offsets, start_t)

    def _slice_hdf5_dataset(
        self,
        dset: Any,
        frames: List[int],
        offsets: List[int],
        start_t: int,
        track_path: Path,
        key: str,
    ) -> np.ndarray:
        if self.reinit and dset.ndim >= 3:
            if start_t >= dset.shape[0]:
                raise IndexError(
                    f"start_t {start_t} out of bounds for {key} in {track_path} with shape {dset.shape}"
                )
            data = np.asarray(dset[start_t])
            if offsets:
                max_offset = max(offsets)
                if max_offset >= data.shape[0]:
                    raise IndexError(
                        f"Offset {max_offset} exceeds horizon {data.shape[0]} for {key} in {track_path}"
                    )
                data = data[offsets]
        else:
            max_frame = max(frames)
            if max_frame >= dset.shape[0]:
                raise IndexError(
                    f"Frame {max_frame} exceeds sequence length {dset.shape[0]} for {key} in {track_path}"
                )
            data = np.asarray(dset[frames])
        return data.astype(np.float32)

    def _load_tracks_from_hdf5(
        self,
        track_path: Path,
        frames: List[int],
        offsets: List[int],
        start_t: int,
    ) -> Dict[str, np.ndarray]:
        if h5py is None:
            raise ImportError(
                "h5py is required to read preprocessed tracks saved as .hdf5 files. "
                "Please install h5py to enable track loading."
            )
        out: Dict[str, np.ndarray] = {}
        with h5py.File(track_path, "r") as f:
            root = f["root"] if "root" in f else f
            for key in self.track_keys:
                per_view: List[np.ndarray] = []
                for camera in self.cond_cameraviews:
                    if camera not in root:
                        raise KeyError(f"Camera '{camera}' not found in track file {track_path}")
                    cam_group = root[camera]
                    if key not in cam_group:
                        if key == "tracks":
                            raise KeyError(
                                f"Track dataset missing '{key}' for camera '{camera}' in {track_path}"
                            )
                        else:
                            continue
                    dset = cam_group[key]
                    data = self._slice_hdf5_dataset(dset, frames, offsets, start_t, track_path, key)
                    per_view.append(data)
                if per_view:
                    out[key] = np.stack(per_view, axis=0).astype(np.float32)
        return out

    def _slice_npz_array(
        self,
        arr: np.ndarray,
        frames: List[int],
        offsets: List[int],
        start_t: int,
        track_path: Path,
        key: str,
    ) -> np.ndarray:
        arr = np.asarray(arr)
        num_views = len(self.cond_cameraviews)

        if arr.ndim == 5:
            # (V, T, horizon, N, D)
            if arr.shape[0] != num_views:
                raise ValueError(
                    f"Expected first dim to match number of views ({num_views}) for {key} in {track_path}, got {arr.shape}"
                )
            if self.reinit:
                data = arr[:, start_t]
                data = data[:, offsets]
            else:
                data = arr[:, frames]
            return data.astype(np.float32)

        if arr.ndim == 4:
            if arr.shape[0] == num_views:
                # (V, T, N, D) or (V, T, horizon, N)
                if self.reinit and arr.shape[2] >= len(offsets):
                    data = arr[:, start_t]
                    data = data[:, offsets]
                else:
                    data = arr[:, frames]
                return data.astype(np.float32)
            else:
                if self.reinit:
                    data = arr[start_t]
                    data = data[offsets]
                else:
                    data = arr[frames]
                return np.expand_dims(data, axis=0).astype(np.float32)

        if arr.ndim == 3:
            if arr.shape[0] == num_views:
                if self.reinit:
                    data = arr[:, start_t]
                    data = data[:, offsets]
                else:
                    data = arr[:, frames]
                return data.astype(np.float32)
            else:
                if self.reinit:
                    data = arr[start_t]
                    data = data[offsets]
                else:
                    data = arr[frames]
                return np.expand_dims(data, axis=0).astype(np.float32)

        raise ValueError(
            f"Unsupported array shape {arr.shape} for key '{key}' in npz track file {track_path}"
        )

    def _load_tracks_from_npz(
        self,
        track_path: Path,
        frames: List[int],
        offsets: List[int],
        start_t: int,
    ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        with np.load(track_path, allow_pickle=False) as data:
            for key in self.track_keys:
                if key not in data:
                    if key == "tracks":
                        raise KeyError(
                            f"Track file {track_path} is missing required dataset '{key}'"
                        )
                    continue
                arr = data[key]
                sliced = self._slice_npz_array(arr, frames, offsets, start_t, track_path, key)
                out[key] = sliced.astype(np.float32)
        return out

    def load_text(self, idx_dict: Dict) -> Dict:
        return {}

    def process_data(self, data: Dict) -> Dict:
        if "images" in data:
            img = data["images"].astype(np.float32)
            data["images"] = np.clip(img / 255.0, 0.0, 1.0)

        if "actions" in data:
            actions = data["actions"].astype(np.float32)
            if actions.ndim == 1:
                actions = actions[:, None]
            if self.normalize_action:
                actions = (actions - self.action_mean_np) / self.action_std_np
            if actions.shape[0] < self.true_horizon:
                pad_len = self.true_horizon - actions.shape[0]
                actions = np.pad(actions, ((0, pad_len), (0, 0)))
            data["actions"] = actions

        if "tracks" in data:
            tracks = data["tracks"].astype(np.float32)
            if tracks.ndim != 4:
                raise ValueError(f"Expected tracks to have 4 dimensions, got {tracks.shape}")

            vis = data.get("vis")
            if vis is not None:
                vis = vis.astype(np.float32)
                if vis.ndim == 3:
                    vis = np.expand_dims(vis, axis=-1)
                elif vis.ndim != 4:
                    raise ValueError(f"Unexpected vis shape {vis.shape}")
                # Align time dimension before padding/truncating
                if vis.shape[1] != tracks.shape[1]:
                    min_len = min(vis.shape[1], tracks.shape[1])
                    tracks = tracks[:, :min_len]
                    vis = vis[:, :min_len]

            tracks = tracks[..., [1, 0]]
            tracks = np.nan_to_num(tracks, nan=0.0, posinf=0.0, neginf=0.0)
            tracks = normalize_traj(tracks, self.data_img_size)

            target_horizon = int(self.true_horizon)
            current_horizon = tracks.shape[1]
            if current_horizon < target_horizon:
                pad = target_horizon - current_horizon
                pad_cfg = ((0, 0), (0, pad), (0, 0), (0, 0))
                tracks = np.pad(tracks, pad_cfg)
                if vis is not None:
                    vis = np.pad(vis, ((0, 0), (0, pad), (0, 0), (0, 0)))
            elif current_horizon > target_horizon:
                tracks = tracks[:, :target_horizon]
                if vis is not None:
                    vis = vis[:, :target_horizon]

            data["tracks"] = tracks.astype(np.float32)
            if vis is not None:
                vis = np.nan_to_num(vis, nan=0.0, posinf=0.0, neginf=0.0)
                data["vis"] = vis.astype(np.float32)

        if "tracks" in data:
            data["traj"] = data.pop("tracks")

        return data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _apply_fraction(self, indices: List[int], fraction: float) -> List[int]:
        if not indices:
            return []
        frac = float(fraction)
        if abs(frac) >= 1.0:
            return list(indices)

        subset_size = max(1, int(round(len(indices) * abs(frac))))
        if frac >= 0:
            return indices[:subset_size]
        else:
            return indices[-subset_size:]

    def _filter_entries(self, entries: List[Dict], n_slices: int, seed: int) -> List[Dict]:
        rng = random.Random(seed)
        match_entries: List[Dict] = []
        non_match_entries: List[Dict] = []

        for entry in entries:
            episode_idx = entry["episode_idx"]
            frames = entry["frames"]
            if len(frames) < 2:
                non_match_entries.append(entry)
                continue

            actions = self.actions[episode_idx][frames]
            actions_np = actions.cpu().numpy()
            target = actions_np[-2]
            if target.ndim > 0:
                target_val = int(round(float(target[0])))
            else:
                target_val = int(round(float(target)))

            if target_val in self.filter_actions:
                match_entries.append(entry)
            else:
                non_match_entries.append(entry)

        if len(match_entries) >= n_slices:
            selected = rng.sample(match_entries, n_slices)
        else:
            selected = list(match_entries)
            remaining = max(0, n_slices - len(match_entries))
            if non_match_entries:
                selected.extend(rng.sample(non_match_entries, min(remaining, len(non_match_entries))))

        return selected

    def _load_obs_episode(self, episode_idx: int) -> np.ndarray:
        if episode_idx not in self._obs_cache:
            self._obs_cache[episode_idx] = np.load(self.obs_paths[episode_idx])
        return self._obs_cache[episode_idx]

