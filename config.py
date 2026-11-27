# -*- coding: utf-8 -*-
"""
Config loader duy nhất cho toàn bộ pipeline DFCCNet.

Mọi script chỉ nhận MỘT tham số dòng lệnh:

    python train.py --config configs/default.yaml
    python test.py  --config configs/default.yaml
    python generate_conv.py --config configs/default.yaml

Toàn bộ tham số (data path, epochs, learning rate, freeze_backbone, ...)
đều nằm trong YAML. Khi chạy, `configs/default.yaml` được nạp làm giá trị
mặc định, file `--config` truyền vào được merge đè lên trên, rồi phần `common`
được merge tiếp vào section riêng của script:

    common.<k>  +  <section>.<k>   ->   cfg.<k>

Ví dụ `python train.py -c configs/finetune_decoder.yaml` =>
`cfg.data_path` (= common), `cfg.epochs`, `cfg.freeze_backbone` (= train).

Không cần cài gì thêm ngoài PyYAML (`pip install -r requirements.txt`).
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Thiếu PyYAML -> chay:  pip install -r requirements.txt"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class Section(dict):
    """dict truy cập được bằng attribute: cfg.epochs thay vì cfg['epochs']."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"Không có khóa '{name}' trong config "
                f"(file: {self.get('_config_path', '?')})"
            ) from exc

    def __repr__(self):
        path = self.get("_config_path", "?")
        keys = ", ".join(k for k in self if not k.startswith("_"))
        return f"<Section {path}: {keys}>"


# ---------------------------------------------------------------------------
def _read(path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Không tìm thấy file config: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Config phải là YAML mapping (dạng key: value): {p}")
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path) -> dict:
    """Đọc 1 file YAML thuần (dùng cho notebook / tool bên ngoài)."""
    return _read(path)


def _apply_extra(target: dict, args, extra_keys: dict) -> None:
    """Cờ dòng lệnh (nếu người dùng có truyền) đè lên giá trị trong YAML."""
    for flag, key in extra_keys.items():
        if hasattr(args, key):
            target[key] = getattr(args, key)


def load_config(section=None, path=None, argv=None, extra=None) -> Section:
    """
    Nạp config và trả về section đã merge với `common`.

    section : None -> trả về cây config đầy đủ (dict lồng nhau)
              'train' -> Section phẳng {**common, **train}
    path    : file config của người dùng (mặc định: --config / configs/default.yaml)
    argv    : list đối số dòng lệnh (mặc định: sys.argv[1:]); truyền [] để test
    extra   : dict {flag: {**kwargs}} để thêm cờ riêng cho script (xem density_show.py)
    """
    parser = argparse.ArgumentParser(
        prog="dfccnet",
        description="DFCCNet - mọi tham số nằm trong file YAML (chỉ truyền --config).",
    )
    parser.add_argument(
        "-c", "--config",
        default=str(path or DEFAULT_CONFIG),
        help="File YAML chứa tham số (mặc định: configs/default.yaml)",
    )
    # cờ riêng của script: SUPPRESS để CHỈ ghi đè config khi người dùng truyền
    extra_keys = {}
    for flag, kwargs in (extra or {}).items():
        kw = dict(kwargs)
        kw.setdefault("dest", flag.lstrip("-").replace("-", "_"))
        kw["default"] = argparse.SUPPRESS
        parser.add_argument(flag, **kw)
        extra_keys[flag] = kw["dest"]
    args = parser.parse_args(argv)

    user_path = Path(args.config)
    config_path = str(user_path if user_path.is_absolute() else Path.cwd() / user_path)

    # giá trị mặc định lấy từ chính default.yaml -> không có số magic trong code
    cfg = _read(DEFAULT_CONFIG)
    if user_path.resolve() != DEFAULT_CONFIG.resolve():
        cfg = _deep_merge(cfg, _read(user_path))
    cfg["_config_path"] = config_path

    if section is None:
        _apply_extra(cfg, args, extra_keys)
        return Section(cfg)

    if section not in cfg or not isinstance(cfg[section], dict):
        raise KeyError(
            f"Config không có section '{section}' "
            f"(có: {', '.join(k for k in cfg if not k.startswith('_'))})"
        )

    flat = {}
    for name in ("common", section):
        block = cfg.get(name)
        if isinstance(block, dict):
            flat.update(copy.deepcopy(block))
    flat["_config_path"] = config_path
    flat["_section"] = section

    _apply_extra(flat, args, extra_keys)

    return Section(flat)


def save_config(cfg, path) -> Path:
    """Ghi config đã resolve (phục vụ tái lập kết quả) ra YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plain = {k: v for k, v in dict(cfg).items() if not k.startswith("_")}
    p.write_text(
        yaml.safe_dump(plain, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return p
