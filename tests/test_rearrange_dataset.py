import json
from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")
np = pytest.importorskip("numpy")

from amplify.loaders.rearrange_dataset import RearrangeDataset


def _build_dummy_rearrange_dataset(tmp_path: Path) -> Path:
    data_root = tmp_path / "rearrange_data"
    ep_dir = data_root / "episodes" / "ep_0000"
    ep_dir.mkdir(parents=True, exist_ok=True)

    metadata = {"episodes": [{"episode": 0, "n_actions": 4}]}
    with open(data_root / "metadata.json", "w") as f:
        json.dump(metadata, f)

    actions = np.arange(4, dtype=np.float32).reshape(4, 1)
    np.save(ep_dir / "actions.npy", actions)

    obs = np.random.randint(0, 255, size=(4, 32, 32, 3), dtype=np.uint8)
    np.save(ep_dir / "obs.npy", obs)

    track_root = tmp_path / "preprocessed" / "rearrange_test" / "uniform_400_reinit_16" / "train"
    track_root.mkdir(parents=True, exist_ok=True)
    track_path = track_root / "ep_0000.hdf5"

    tracks = np.zeros((4, 4, 2, 2), dtype=np.float32)
    vis = np.ones((4, 4, 2), dtype=np.float32)
    for start_idx in range(4):
        for step in range(4):
            coords = np.array([[start_idx + step, start_idx - step], [step, start_idx]], dtype=np.float32)
            tracks[start_idx, step] = coords

    with h5py.File(track_path, "w") as f:
        root = f.create_group("root")
        view = root.create_group("agentview")
        view.create_dataset("tracks", data=tracks)
        view.create_dataset("vis", data=vis)

    return track_root


def test_rearrange_dataset_loads_tracks(tmp_path):
    _build_dummy_rearrange_dataset(tmp_path)

    cfg = {
        "data_path": "rearrange_data",
        "preprocessed_dir": "preprocessed/rearrange_test",
        "num_hist": 3,
        "num_pred": 1,
        "frameskip": 1,
        "track_method": "uniform_400_reinit_16",
        "n_tracks": 2,
    }

    dataset = RearrangeDataset(
        root_dir=str(tmp_path),
        dataset_name="rearrange_test",
        split="train",
        keys_to_load=["tracks", "images"],
        img_shape=[32, 32],
        true_horizon=4,
        cfg=dict(cfg),
        fraction=1.0,
    )

    assert len(dataset) > 0
    sample = dataset[0]
    assert "traj" in sample
    assert sample["traj"].shape[1] == dataset.true_horizon
    assert np.isfinite(sample["traj"]).all()
    assert "vis" not in sample

    dataset_with_vis = RearrangeDataset(
        root_dir=str(tmp_path),
        dataset_name="rearrange_test",
        split="train",
        keys_to_load=["tracks", "vis", "images"],
        img_shape=[32, 32],
        true_horizon=4,
        cfg=dict(cfg),
        fraction=1.0,
    )

    sample_with_vis = dataset_with_vis[0]
    assert "traj" in sample_with_vis
    assert "vis" in sample_with_vis
    assert sample_with_vis["vis"].shape[:3] == sample_with_vis["traj"].shape[:3]


def test_rearrange_dataset_errors_on_track_count_mismatch(tmp_path):
    _build_dummy_rearrange_dataset(tmp_path)

    cfg = {
        "data_path": "rearrange_data",
        "preprocessed_dir": "preprocessed/rearrange_test",
        "num_hist": 3,
        "num_pred": 1,
        "frameskip": 1,
        "track_method": "uniform_400_reinit_16",
        "n_tracks": 3,
    }

    dataset = RearrangeDataset(
        root_dir=str(tmp_path),
        dataset_name="rearrange_test",
        split="train",
        keys_to_load=["tracks", "images"],
        img_shape=[32, 32],
        true_horizon=4,
        cfg=dict(cfg),
        fraction=1.0,
    )

    with pytest.raises(ValueError) as exc_info:
        dataset[0]

    message = str(exc_info.value)
    assert "expected 3 tracks" in message
    assert "got 2" in message
    assert "dataset='rearrange_test'" in message
    assert "split='train'" in message
    assert "track_method='uniform_400_reinit_16'" in message
    assert "track_path='" in message
