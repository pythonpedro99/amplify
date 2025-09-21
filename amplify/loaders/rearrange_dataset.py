import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import Resize

from amplify.loaders.base_dataset import BaseDataset
from amplify.utils.data_utils import (
    interpolate_traj,
    interpolate_traj_spline,
    normalize_traj,
)


@dataclass
class _FilterCfg:
    enabled: bool
    n_slices: Optional[int]
    seed: int


class RearrangeDataset(BaseDataset):
    """Dataset loader for the rearrange manipulation benchmark."""

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

        view_names = self.cfg.get("view_names")
        if view_names is not None:
            self.view_names = tuple(str(v) for v in view_names)
        else:
            self.view_names = (str(self.cfg.get("view_name", "agentview")),)
        self.primary_view = self.view_names[0]

        track_method_cfg = self.cfg.get("track_method")
        if track_method_cfg is None:
            reinit_suffix = ""
            if self.cfg.get("reinit", True):
                reinit_suffix = f"_reinit_{int(self.cfg.get('horizon', 16))}"
            track_method_cfg = (
                f"{self.cfg.get('init_queries', 'uniform')}"
                f"_{int(self.cfg.get('n_tracks', 400))}{reinit_suffix}"
            )
        self.track_method = str(track_method_cfg)

        preprocessed_dir = self.cfg.get("preprocessed_dir", "preprocessed_data")
        preprocessed_dir = os.path.expanduser(str(preprocessed_dir))
        if not os.path.isabs(preprocessed_dir):
            preprocessed_dir = os.path.join(root_dir, preprocessed_dir)
        self.preprocessed_dir = Path(preprocessed_dir)

        tracks_dir = self.cfg.get("tracks_dir")
        if tracks_dir is None:
            tracks_dir = self.preprocessed_dir / self.dataset_name / self.track_method
        else:
            tracks_dir = Path(os.path.expanduser(str(tracks_dir)))
            if not tracks_dir.is_absolute():
                tracks_dir = Path(root_dir) / tracks_dir
        self.track_root = Path(tracks_dir)
        self.track_file_template = str(self.cfg.get("track_file_template", "ep_{episode:04d}.hdf5"))

        data_path = self.cfg.get("data_path")
        if data_path is None:
            raise ValueError("rearrange_dataset.data_path must be set")
        data_path = os.path.expanduser(str(data_path))
        if not os.path.isabs(data_path):
            data_path = os.path.join(root_dir, data_path)
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Rearrange dataset path does not exist: {self.data_path}")

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

        super().__init__(
            root_dir=root_dir,
            dataset_names=[dataset_name],
            track_method=self.track_method,
            cond_cameraviews=list(self.view_names),
            keys_to_load=list(keys_to_load),
            img_shape=self.img_shape,
            true_horizon=int(true_horizon),
            track_pred_horizon=int(self.cfg.get("track_pred_horizon", true_horizon)),
            interp_method=str(self.cfg.get("interp_method", "linear")),
            num_tracks=int(self.cfg.get("n_tracks", 400)),
            use_cached_index_map=bool(self.cfg.get("use_cached_index_map", False)),
            aug_cfg=aug_cfg,
        )

        self.track_keys = [k for k in self.keys_to_load if k in {"tracks", "vis"}]
        self.image_obs_keys = ["images"]
        if self.obs_paths:
            sample_obs = np.load(self.obs_paths[0], mmap_mode="r")
            self.data_img_size = tuple(int(x) for x in sample_obs.shape[1:3])
            del sample_obs
        else:
            self.data_img_size = self.img_shape
        self.resize_transform = Resize(self.img_shape, antialias=False)

    # ------------------------------------------------------------------
    # BaseDataset abstract implementations
    # ------------------------------------------------------------------
    def get_cache_file(self) -> str:
        cache_dir = os.path.expanduser("~/.cache/amplify/index_maps/rearrange")
        os.makedirs(cache_dir, exist_ok=True)
        frac_tag = f"{self.fraction:+.2f}".replace(".", "p")
        filter_cfg = self.filter_cfg["train" if self.split == "train" else "val"]
        filter_tag = "filtered" if filter_cfg.enabled else "all"
        track_tag = self.track_method.replace(os.sep, "-")
        fname = (
            f"{self.dataset_name}_{self.split}_{frac_tag}_"
            f"len{self.slice_len}_fs{self.frameskip}_{filter_tag}_{track_tag}.json"
        )
        return os.path.join(cache_dir, fname)

    def create_index_map(self) -> List[Dict]:
        index_map: List[Dict] = []
        need_tracks = bool(self.track_keys)
        missing_tracks = 0

        for epi_idx in self.selected_indices:
            seq_len = self.seq_lengths[epi_idx]
            if seq_len < self.slice_span:
                continue

            track_path: Optional[Path] = None
            if need_tracks:
                episode_id = int(self.episodes_meta[epi_idx]["episode"])
                track_fname = self.track_file_template.format(
                    episode=episode_id,
                    episode_idx=epi_idx,
                    split=self.split,
                    dataset=self.dataset_name,
                )
                track_path = self.track_root / track_fname
                if not track_path.exists():
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

        if need_tracks and missing_tracks > 0:
            print(
                f"[RearrangeDataset] Skipped {missing_tracks} episodes without preprocessed tracks"
            )

        return index_map

    def load_images(self, idx_dict: Dict) -> Dict:
        episode_idx = idx_dict["episode_idx"]
        frames = idx_dict["frames"]
        obs = self._load_obs_episode(episode_idx)
        frame = obs[frames[0]].astype(np.float32)
        if frame.ndim != 3:
            raise ValueError(f"Unexpected obs shape {frame.shape} for episode {episode_idx}")
        frame = np.expand_dims(frame, axis=0)  # (V, H, W, C)
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
        if not self.track_keys:
            return {}
        track_path = idx_dict.get("track_path")
        if track_path is None or not os.path.exists(track_path):
            return {}

        start_t = idx_dict["start_t"]
        tracks_per_view: List[np.ndarray] = []
        vis_per_view: List[np.ndarray] = []

        with h5py.File(track_path, "r") as f:
            for view in self.view_names:
                group_name = f"root/{view}"
                if group_name not in f:
                    continue
                view_group = f[group_name]
                if "tracks" in view_group and "tracks" in self.track_keys:
                    tracks_per_view.append(view_group["tracks"][start_t:start_t + 1])
                if "vis" in view_group and "vis" in self.track_keys:
                    vis_per_view.append(view_group["vis"][start_t:start_t + 1])
                elif "visibility" in view_group and "vis" in self.track_keys:
                    vis_per_view.append(view_group["visibility"][start_t:start_t + 1])

        out: Dict[str, np.ndarray] = {}
        if tracks_per_view:
            out["tracks"] = np.concatenate(tracks_per_view, axis=0).astype(np.float32)
        if vis_per_view:
            out["vis"] = np.concatenate(vis_per_view, axis=0).astype(np.float32)
        return out

    def load_text(self, idx_dict: Dict) -> Dict:
        return {}

    def process_data(self, data: Dict) -> Dict:
        if "images" in data:
            img = data["images"].astype(np.float32)
            data["images"] = np.clip(img / 255.0, 0.0, 1.0)
            if tuple(img.shape[-3:-1]) != self.img_shape:
                img_t = torch.from_numpy(rearrange(data["images"], "v h w c -> v c h w")).float()
                resized = self.resize_transform(img_t).numpy().astype(np.float32)
                data["images"] = rearrange(resized, "v c h w -> v h w c")

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
            if tracks.shape[1] > self.true_horizon:
                tracks = tracks[:, : self.true_horizon]
            tracks = tracks[..., [1, 0]]
            tracks = np.nan_to_num(tracks, nan=0.0, posinf=0.0, neginf=0.0)
            tracks = normalize_traj(tracks, self.data_img_size)

            if self.track_pred_horizon != self.true_horizon:
                if self.interp_method == "linear":
                    tracks = interpolate_traj(tracks, self.track_pred_horizon)
                elif self.interp_method == "spline":
                    tracks = interpolate_traj_spline(tracks, self.track_pred_horizon)
                else:
                    raise NotImplementedError(f"Unknown interpolation method {self.interp_method}")
                if isinstance(tracks, torch.Tensor):
                    tracks = tracks.cpu().numpy()

            if isinstance(tracks, torch.Tensor):
                tracks = tracks.cpu().numpy()
            data["tracks"] = tracks

        if "vis" in data:
            vis = data["vis"].astype(np.float32)
            if vis.ndim == 3:
                vis = np.expand_dims(vis, axis=-1)
            vis = np.nan_to_num(vis, nan=0.0, posinf=0.0, neginf=0.0)
            if self.track_pred_horizon != self.true_horizon:
                vis = interpolate_traj(vis, self.track_pred_horizon)
                if isinstance(vis, torch.Tensor):
                    vis = vis.cpu().numpy()
            if isinstance(vis, torch.Tensor):
                vis = vis.cpu().numpy()
            data["vis"] = vis

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
