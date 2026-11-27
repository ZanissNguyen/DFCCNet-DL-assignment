import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from config import load_config

"""
Density map visualization

    python density_show.py --config configs/default.yaml
    python density_show.py --config configs/default.yaml --npy chicken/annocation/42.npy

Đường dẫn mặc định lấy trong config (section `density_show.npy`);
`--npy` chỉ là lối tắt để xem nhanh một file bất kỳ mà không phải sửa YAML.
"""
cfg = load_config("density_show", extra={
    "--npy": {"type": str, "help": "Xem nhanh 1 file .npy, ghi đè density_show.npy của config"},
})

load_npy = np.load(cfg.npy)
np.set_printoptions(threshold=np.inf)
plt.imshow(load_npy, cmap=cm.jet)
plt.show()
