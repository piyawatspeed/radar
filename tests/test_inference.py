import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Provide lightweight stubs for heavy optional dependencies so that importing
# ``inference`` during testing does not require the full runtime stack.
def _not_implemented(*args, **kwargs):  # pragma: no cover - helper for stubs
    raise NotImplementedError

sys.modules.setdefault("numpy", types.ModuleType("numpy"))

pil_pkg = types.ModuleType("PIL")
pil_image = types.ModuleType("PIL.Image")
pil_imagefile = types.ModuleType("PIL.ImageFile")
pil_pkg.Image = pil_image
pil_pkg.ImageFile = pil_imagefile
sys.modules.setdefault("PIL", pil_pkg)
sys.modules.setdefault("PIL.Image", pil_image)
sys.modules.setdefault("PIL.ImageFile", pil_imagefile)

torch_pkg = types.ModuleType("torch")
torch_cuda = types.ModuleType("torch.cuda")
torch_cuda.is_available = lambda: False
torch_nn = types.ModuleType("torch.nn")
torch_nn_functional = types.ModuleType("torch.nn.functional")
torch_pkg.nn = torch_nn
torch_pkg.cuda = torch_cuda
sys.modules.setdefault("torch", torch_pkg)
sys.modules.setdefault("torch.cuda", torch_cuda)
sys.modules.setdefault("torch.nn", torch_nn)
sys.modules.setdefault("torch.nn.functional", torch_nn_functional)

stub_ddp = types.ModuleType("ddp")
stub_ddp.LEFT_MASK_PX = 0
stub_ddp.MIN_TOL_MINUTES = 5
stub_ddp.NEIGHBOR_MINUTES = 5
stub_ddp.RADAR_A_ID = "NKM"
stub_ddp.RADAR_B_ID = "NJK"
stub_ddp.TOL_FRAC = 0.5
stub_ddp.HSVParams = type("HSVParams", (), {})
stub_ddp.TinyUNet = type("TinyUNet", (), {})
stub_ddp._load_bool_mask_npz = _not_implemented
stub_ddp._weak_label_core = _not_implemented
stub_ddp.add_minutes = lambda ts, minutes: ts
stub_ddp.build_warp_grid = lambda *args, **kwargs: None
stub_ddp.compute_center_px = lambda *args, **kwargs: (0.0, 0.0)
stub_ddp.compute_mpp = lambda *args, **kwargs: 1.0
stub_ddp.helper_channels = lambda *args, **kwargs: []
stub_ddp.list_images = lambda root: []
stub_ddp.warp_cache_path = lambda *args, **kwargs: Path("/tmp/warp")
stub_ddp._parse_center_tuple = lambda value: (0.0, 0.0)
stub_ddp.parse_timestamp = lambda path: None
stub_ddp.ts_to_dt = lambda ts: ts
stub_ddp.WEAK_LABEL_MODE = "default"
stub_ddp.HSV_MULTI_RANGES = ()
stub_ddp.HSV_H_LO = 0.0
stub_ddp.HSV_H_HI = 0.0
stub_ddp.HSV_S_MIN = 0.0
stub_ddp.HSV_V_MIN = 0.0
stub_ddp.RADAR_A_CENTER_OVERRIDE = None
stub_ddp.RADAR_B_CENTER_OVERRIDE = None
stub_ddp.RADAR_A_LATLON = (0.0, 0.0)
stub_ddp.RADAR_B_LATLON = (0.0, 0.0)
stub_ddp.RADAR_A_RANGE_KM = 0.0
stub_ddp.RADAR_B_RANGE_KM = 0.0
sys.modules.setdefault("ddp", stub_ddp)

from inference import _discover_radar_dirs
from ddp import RADAR_A_ID, RADAR_B_ID


def test_discover_radar_dirs_prefers_identifier(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()

    radar_a_dir = data_root / f"20231019_{RADAR_A_ID.lower()}_frames"
    radar_b_dir = data_root / f"20231019_{RADAR_B_ID.lower()}_frames"
    radar_a_dir.mkdir()
    radar_b_dir.mkdir()

    # Extra directory to ensure we don't simply rely on sorting order.
    (data_root / "misc_radar").mkdir()

    radar_a, radar_b = _discover_radar_dirs(data_root, None, None)

    assert radar_a.resolve() == radar_a_dir.resolve()
    assert radar_b.resolve() == radar_b_dir.resolve()


def test_discover_radar_dirs_errors_when_same_dir_matches_both(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()

    shared_dir = data_root / f"20231019_{RADAR_A_ID.lower()}_{RADAR_B_ID.lower()}"
    shared_dir.mkdir()
    (data_root / "misc_candidate").mkdir()

    with pytest.raises(RuntimeError) as excinfo:
        _discover_radar_dirs(data_root, None, None)

    message = str(excinfo.value)
    assert RADAR_A_ID in message and RADAR_B_ID in message
    assert "overlap" in message.lower()


def test_discover_radar_dirs_errors_when_override_blocks_other_match(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()

    shared_dir = data_root / f"20231019_{RADAR_A_ID.lower()}_{RADAR_B_ID.lower()}"
    shared_dir.mkdir()
    (data_root / "fallback" ).mkdir()

    radar_a_override = str(shared_dir)

    with pytest.raises(RuntimeError) as excinfo:
        _discover_radar_dirs(data_root, radar_a_override, None)

    message = str(excinfo.value)
    assert "radar b" in message.lower()
    assert "--radar-b" in message
