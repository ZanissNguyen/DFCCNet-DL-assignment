# MANUAL.md — Hướng dẫn chạy các module của DFCCNet

Tài liệu này mô tả **từng module Python trong repo**, **khóa cấu hình (YAML)** của nó và
**các lệnh chạy mẫu**. Đọc thêm phần cài đặt ở [`README.md`](README.md)
(section *Prerequisites* / *Contributing*) và [`requirements.txt`](requirements.txt).

> **Điểm mới:** mọi tham số đã gom vào file YAML. Mỗi script chỉ nhận **một đối số
> duy nhất `--config`** (hoặc `-c`) — không còn `-dp/-o/-ts/-ep/-g/-wm/-lr`.

---

## 1. Tổng quan

### 1.1 Bảng module

| Module | Loại | Vai trò | Gọi ra |
|---|---|---|---|
| `config.py` | library | Nạp + merge YAML, là module cấu hình **duy nhất** của cả pipeline | – |
| `configs/default.yaml` | config | Config **mặc định** = base của mọi config khác | – |
| `configs/finetune_decoder.yaml` | config | Ví dụ: freeze backbone, chỉ fine-tune decoder | – |
| `generate_conv.py` | script (`--config`) | Sinh feature-convolution kernels (`.npy`) | `<conv_dir>/*.npy` |
| `train.py` | script (`--config`) | Huấn luyện backbone (ResNet50-FPN) ± `DensityRegressor` | `backbone.pth`, `generate_density.pth`, `stats.txt`, `config.yaml` |
| `test.py` | script (`--config`) | Đánh giá (MAE/RMSE) + vẽ density map | `<test.output_dir>/*_myout.png` |
| `density_show.py` | script (`--config` + `--npy`) | Visualize một file density `.npy` | cửa sổ matplotlib |
| `model.py` | library | `Resnet50FPN`, `DensityRegressor`, `weights_normal_init` | – |
| `utils.py` | library | Transform, `extract_features`, `ExtraLoss`, `visualize`, hằng số `MAPS` | – |

### 1.2 Hệ thống config

```
configs/default.yaml          ← file mặc định, mọi config khác merge đè lên đây
        │
        ├── common:           ← dùng chung cho MỌI script
        │     data_path, images_dir, anno_file, split_file,
        │     density_dir, conv_dir, gpu
        │
        ├── generate_conv:    ← riêng generate_conv.py
        ├── train:            ← riêng train.py     (epochs, freeze_backbone, ...)
        ├── test:             ← riêng test.py      (backbone_ckpt, regressor_ckpt, ...)
        └── density_show:     ← riêng density_show.py
```

Khi chạy, `common` được **merge vào section riêng** của script:

```python
cfg = load_config("train")      # => {**common, **train}
```

nên `cfg.data_path` và `cfg.epochs` cùng nằm ở cấp trên, không cần `cfg['train']['epochs']`.

**Cách tạo config cho experiment riêng:** chỉ ghi phần *chênh lệch* với `default.yaml`:

> 📄 **[`configs/example.yaml`](configs/example.yaml)** — file-minh-họa liệt kê **mọi khóa**
> của cả 5 section, kèm kiểu dữ liệu / giá trị mặc định / giá trị hợp lệ và quy tắc
> merge. Đây là tài liệu chính thức về định dạng config; nó cũng là config hợp lệ
> (trùng giá trị với `default.yaml`) nên chạy thử được ngay.

```yaml
# configs/my_run.yaml
train:
  output_dir: ./output/my_run
  epochs: 30
  freeze_backbone: true
```

```bash
python train.py --config configs/my_run.yaml
```

Các khóa không xuất hiện trong `my_run.yaml` vẫn lấy từ `default.yaml` (merge sâu theo section).

### 1.3 Tiền đề chung (áp dụng cho MỌI script)

* **`--config` là đối số duy nhất.** Không truyền gì cả thì dùng `configs/default.yaml`.
* **Chạy từ thư mục gốc repo** (working directory = repo root): các đường dẫn trong YAML
  (`./chicken/`, `./output`, `./result/*.pth`) đều là **tương đối** so với cwd.
* **Bắt buộc có GPU CUDA** — toàn bộ code gọi `.cuda()` cứng, không có nhánh CPU
  (kiểm tra: `python -c "import torch; print(torch.cuda.is_available())"`).
* **`common.data_path` phải kết thúc bằng `/`** vì code nối chuỗi kiểu
  `data_path + cfg.anno_file` (không dùng `os.path.join`).
* `train.py` / `test.py` tự `os.makedirs(output_dir, exist_ok=True)` — không cần tạo tay.
* **PyYAML** là dependency mới: `pip install -r requirements.txt` (đã có trong file).

### 1.4 Cấu trúc dữ liệu mà code mong muốn

```
repo/
├── configs/
│   ├── default.yaml             # base config (mọi script)
│   └── finetune_decoder.yaml    # ví dụ freeze backbone
├── chicken/                     # common.data_path = ./chicken/
│   ├── images/                  #   ảnh .jpg (600 ảnh)         [common.images_dir]
│   ├── chicken_annocation.json  #   annotation                  [common.anno_file]
│   ├── Train_Test_Val_chicken.json  # split 416/61/123          [common.split_file]
│   ├── annocation/              #   GT density .npy/train       [common.density_dir]
│   │   └── <im_id>.npy          #   (repo KHÔNG có sẵn — xem §7)
│   ├── single_chick/            #   1.jpg … 20.jpg              [generate_conv.source_dir]
│   └── conv/                    #   output generate_conv.py    [common.conv_dir]
│       ├── map3_conv.npy  map4_conv.npy
│       └── map3_conv_08.npy … map3_conv_12.npy
├── result/
│   ├── backbone.pth             # pretrained (torch.save cả module)
│   └── generate_density.pth     # pretrained DensityRegressor (state_dict)
├── output/                      # train: *.pth, stats.txt, config.yaml  [train.output_dir]
└── showimg/                     # test: *_myout.png                     [test.output_dir]
```

---

## 2. `generate_conv.py` — sinh feature convolution kernels

**Chạy TRƯỚC** mọi bước train/test (nếu `<conv_dir>/*.npy` đã có sẵn thì bỏ qua).

### Khóa config (section `generate_conv`)

| Khóa | Kiểu | Default | Mô tả |
|---|---|---|---|
| `source_dir` | `str` | `single_chick` | Ảnh nguồn `<data_path><source_dir>/1.jpg … <n_source_images>.jpg` |
| `n_source_images` | `int` | `20` | Số ảnh nguồn đọc vào |

Kế thừa từ `common`: `data_path`, `conv_dir` (nơi ghi 10 file `.npy`), `gpu`.

### Ví dụ

```bash
# mặc định (configs/default.yaml)
python generate_conv.py --config configs/default.yaml
python generate_conv.py            # --config có default, không bắt buộc truyền
```

> Ghi 10 file `.npy` vào `common.conv_dir` (default `./chicken/conv/`).
> Script tự `mkdir` thư mục này nếu chưa có (đã sửa — bản gốc `np.save` lỗi nếu thiếu).

### Tham số nội bộ (không có khóa config)

`scale = 100 * (600/1280)`, `feat_map_keys = ['map3','map4']`,
`change_scales = [0.8, 0.9, 1.1, 1.2]`.

---

## 3. `train.py` — huấn luyện model

### Khóa config (section `train`)

| Khóa | Kiểu | Default | Mô tả |
|---|---|---|---|
| `output_dir` | `str` | `./output` | Nơi lưu `*.pth`, `stats.txt`, `config.yaml` (tự tạo thư mục) |
| `eval_split` | `str` | `val` | Split đánh giá **sau MỖI epoch** để chọn best model: `train`/`val`/`test` |
| `epochs` | `int` | `50` | Số epoch |
| `learning_rate` | `float` | `1.0e-5` | LR cho optimizer (và của backbone nếu không freeze) |
| `extra_loss_weight` | `float` | `1.0e-9` | Hệ số `ExtraLoss` (loss theo bounding box) cộng vào MSE |
| `freeze_backbone` | `bool` | `false` | **true = đóng băng ResNet50-FPN, chỉ train `DensityRegressor`** (xem §3.1) |
| `save_backbone` | `bool\|null` | `~` (null) | `null` = tự quyết (`not freeze_backbone`); set `true/false` để ép |
| `init_backbone_ckpt` | `str\|null` | `~` (null) | Load backbone pretrained **trước khi freeze**: `./result/backbone.pth` (Dense-Chicken). `null` = dùng ImageNet trong `model.py` |
| `init_regressor_ckpt` | `str\|null` | `~` (null) | Load decoder pretrained **trước khi freeze**: `./result/generate_density.pth`. `null` = `weights_normal_init` (random) |
| `step_size` | `int` | `20` | `StepLR.step_size` |
| `step_gamma` | `float` | `0.9` | `StepLR.gamma` |

Kế thừa từ `common`: `data_path`, `anno_file`, `split_file`, `images_dir`, `density_dir`, `conv_dir`, `gpu`.

### 3.1 Cơ chế fine-tune (mục tiêu của nhóm: đổi decoder)

Thứ tự khởi tạo trong `train.py` (bắt buộc: **load xong mới freeze**):

```
1) Resnet50FPN() / DensityRegressor(10)         ← khởi tạo
2) load init_backbone_ckpt  (result/backbone.pth)          ← Dense-Chicken pretrained
   load init_regressor_ckpt (result/generate_density.pth)  ← Dense-Chicken pretrained
3) nếu freeze_backbone: requires_grad=False + eval()        ← SAU khi load
4) optimizer = Adam(regressor.parameters())    ← CHỈ tham số decoder; optimizer0 = None
```

```yaml
train:
  init_backbone_ckpt: ./result/backbone.pth           # 2) load pretrained Dense-Chicken
  init_regressor_ckpt: ./result/generate_density.pth
  freeze_backbone: true     # 3) ĐÓNG BĂNG ResNet50-FPN
  save_backbone: false      # ~ (null) đã tự ra false khi freeze
```

Khi `freeze_backbone: true`:

* `requires_grad=False` cho **toàn bộ** tham số backbone → autograd không dựng graph qua nó
  (gradient chỉ chạy tới regressor — tiết kiệm VRAM + thời gian);
* backbone luôn ở `.eval()` → BatchNorm dùng stats pretrained, **không** cập nhật running mean/var;
* `optimizer`/`scheduler` của backbone = `None` (không tạo) → mỗi bước chỉ `optimizer.step()` của regressor;
* chỉ `generate_density.pth` được lưu (nếu `save_backbone` không bị ép `true`);
* file config resolve được ghi vào `<output_dir>/config.yaml` để tái lập experiment.

Ví dụ đầy đủ: [`configs/finetune_decoder.yaml`](configs/finetune_decoder.yaml).

### Hành vi

* Optimizer: `Adam`. Loss: `MSELoss + extra_loss_weight * ExtraLoss`.
* Best model (val MAE nhỏ nhất) ghi đè mỗi lần cải thiện:
  * `<output_dir>/backbone.pth` → `torch.save(resnet50_conv, ...)` (lưu **cả object**, chỉ khi `save_backbone`)
  * `<output_dir>/generate_density.pth` → `regressor.state_dict()` (luôn lưu)
* `<output_dir>/stats.txt` → 1 dòng/epoch: `loss,train_mae,train_rmse,val_mae,val_rmse`

### Ví dụ

```bash
# train toàn bộ (mặc định: 50 epoch, không freeze)
python train.py --config configs/default.yaml

# chỉ fine-tune decoder (freeze backbone)
python train.py --config configs/finetune_decoder.yaml
```

> **Không có `--resume`:** mỗi lần chạy đều bắt đầu từ pretrained
> `ResNet50_Weights.IMAGENET1K_V1` (`model.py` dùng `pretrained=True`).
> Muốn tiếp tục thì tự sửa code để `load_state_dict` từ checkpoint cũ.
> (Khi freeze, backbone giữ nguyên weight pretrained — đúng ý đồ "chỉ đổi decoder".)

---

## 4. `test.py` — đánh giá (MAE / RMSE) + visualize

### Khóa config (section `test`)

| Khóa | Kiểu | Default | Mô tả |
|---|---|---|---|
| `output_dir` | `str` | `./showimg` | Ảnh `<im_id>_myout.png` (tự tạo thư mục) |
| `eval_split` | `str` | `val` | Split đánh giá: `val` / `test` |
| `backbone_ckpt` | `str` | `./result/backbone.pth` | Checkpoint backbone (`torch.save(model)` — load `weights_only=False`) |
| `regressor_ckpt` | `str` | `./result/generate_density.pth` | State_dict `DensityRegressor` |

Kế thừa từ `common`: `data_path`, `anno_file`, `split_file`, `images_dir`, `conv_dir`, `gpu`.

### Hành vi

* Checkpoint lấy theo **config** (không còn load từ cwd như bản gốc) →
  muốn test model vừa train, chỉ trỏ `regressor_ckpt` vào `<train.output_dir>/...`.
* Với mỗi ảnh: `pred_cnt = output.sum()` vs `dots.shape[0]` → cộng dồn `SAE`/`SSE` → `MAE`, `RMSE`.
* In thêm `average time`, `average fps`.
* Ảnh kết quả: `<test.output_dir>/<im_id>_myout.png` (4 panel: input + điểm GT, overlay, density map).

### Ví dụ

```bash
# pretrained trong result/ (mặc định)
python test.py --config configs/default.yaml

# checkpoint của experiment fine-tune decoder (configs/finetune_decoder.yaml
# đã trỏ regressor_ckpt vào ./output/finetune_decoder/generate_density.pth)
python test.py --config configs/finetune_decoder.yaml

# đổi split nhanh bằng cách sửa test.eval_split trong YAML, hoặc:
python test.py --config configs/default.yaml   # rồi sửa YAML: eval_split: test
```

---

## 5. `density_show.py` — visualize một density map

```bash
# file lấy từ config:  density_show.npy  (mặc định ./chicken/annocation/1.npy)
python density_show.py --config configs/default.yaml

# hoặc truyền nhanh 1 file bất kỳ (flag phụ duy nhất của toàn repo)
python density_show.py --npy chicken/annocation/42.npy
```

`--npy` ghi đè giá trị trong YAML (tiện xem nhanh, không phải sửa file).

---

## 6. Pipeline chuẩn (thứ tự chạy)

```bash
# 0) môi trường (xem README → Contributing)
python -m venv .venv && source .venv/bin/activate     # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113

# 1) sinh feature convolution kernels  (bỏ qua nếu <conv_dir>/*.npy đã có)
python generate_conv.py --config configs/default.yaml

# 2) train — toàn bộ siêu tham số nằm trong YAML
python train.py --config configs/default.yaml
#    hoặc fine-tune riêng decoder:
python train.py --config configs/finetune_decoder.yaml

# 3) test bằng checkpoint (trỏ test.regressor_ckpt trong YAML nếu muốn model vừa train)
python test.py --config configs/default.yaml

# 4) (tuỳ chọn) xem density map
python density_show.py --npy chicken/annocation/1.npy
```

---

## 7. Vấn đề đã biết khi chạy (known issues)

### 7.1 Đã được sửa trong repo này

1. ~~`train.py` lỗi cú pháp `default=`~~ → đã xoá argparse, chuyển sang YAML (`config.py`).
2. ~~Tên file annotation lệch nhau~~ (`chicken_annotation.json` vs `chicken_annocation.json`)
   → cả 4 script đều đọc `common.anno_file` (= file thật `chicken_annocation.json`).
3. ~~`showimg/` không tự tạo~~ → `test.py` (và `train.py`) đã `os.makedirs(..., exist_ok=True)`.
4. ~~`test.py` load checkpoint từ cwd~~ → đọc `test.backbone_ckpt` / `test.regressor_ckpt` từ YAML.
5. ~~`torch.load` mặc định `weights_only=True` (torch ≥ 2.6)~~ → `test.py` truyền
   `weights_only=False` (có fallback `TypeError` cho torch < 1.13).
6. ~~`StepLR(..., verbose=True)` bị gỡ ở torch ≥ 2.2~~ → đã bỏ.
7. ~~`utils.py` tạo `nn.MSELoss().cuda()` lúc import~~ → đã bỏ (chặn import trên máy không CUDA).
8. ~~`chicken/conv/` hardcode~~ → đổi qua `common.conv_dir`; thư mục được `mkdir` tự động.
9. ~~`generate_conv.py` hardcode 20 ảnh / ghi `chicken/conv/`~~ → `source_dir`, `n_source_images`, `conv_dir` từ YAML.

### 7.2 Còn tồn tại (cần biết trước khi chạy)

1. **`torchvision` mới chỉ *warning*, không lỗi:** `model.py` dùng
   `resnet50(pretrained=True)` → torchvision ≥ 0.13 in `UserWarning: The parameter
   'pretrained' is deprecated` nhưng **vẫn chạy** (đã kiểm với torchvision 0.27).
   Chỉ khi bản torchvision của bạn xoá hẳn kwarg mới cần đổi sang
   `weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1`.
2. **Thiếu ground-truth density map** ở `<data_path>annocation/<im_id>.npy`
   (lưu ý chính tả `annocation`) — repo chỉ có ảnh + point annotation.
   Thiếu thì `np.load` sẽ lỗi ngay epoch đầu. 2 cách:
   * lấy `annocation/` từ bộ dữ liệu gốc của tác giả, hoặc
   * sinh tạm từ `points` bằng Gaussian kernel (cell *"Generate density maps"*
     trong [`notebooks/kaggle.ipynb`](notebooks/kaggle.ipynb), đã smoke-test).
3. **`--data_path` không có `os.path.join`** → thiếu dấu `/` cuối sẽ ra
   `...chickenchicken_annocation.json`. (Vẫn đúng với `common.data_path`.)
4. **`ExtraLoss` chỉ dùng box đầu tiên** (`for ...: ... break`) và repo hiện có
   `boxs: []` rỗng cho phần lớn ảnh → về thực chất loss thêm vào ≈ 0
   (`train.extra_loss_weight` không ảnh hưởng nhiều).
5. **Không có CPU fallback** — `.cuda()` cứng ở mọi nơi; trên máy không có GPU
   chỉ chạy được Kaggle (xem §8).
6. **Không có `--resume`** — mỗi lần train bắt đầu từ ImageNet pretrained.

---

## 8. Chạy trên Kaggle GPU

Repo có sẵn [`notebooks/kaggle.ipynb`](notebooks/kaggle.ipynb). Notebook sẽ:

1. **Nạp source** — ưu tiên đọc thẳng từ path
   `/kaggle/input/<slug>/train.py` (đẩy source lên Kaggle Dataset),
   nếu không thì giải nén `code.zip` từ `/kaggle/input/<slug>/code.zip`
   về `/kaggle/working/src`.
2. **Nạp dataset** — tìm folder có `Train_Test_Val_chicken.json`, hoặc tự giải nén
   `Dense-Chicken.zip`.
3. Cài phần thiếu của `requirements.txt` (**không** cài lại `torch` — Kaggle đã có sẵn CUDA build).
4. Sinh density map tạm bằng Gaussian nếu thiếu `annocation/` (§7.2).
5. **Ghi `run.yaml`** vào `/kaggle/working` — merge `common` + `train` + `test` với
   path của REPO/DATA. Toggle ở cell config: `FREEZE_BACKBONE`, `SAVE_BACKBONE`,
   `INIT_BACKBONE` / `INIT_REGRESSOR` (`'auto'` = tự dùng `result/*.pth` nếu file tồn tại).
6. Chạy pipeline bằng `subprocess` với **đúng cú pháp config mới**:
   `python generate_conv.py --config run.yaml` → `python train.py --config run.yaml`
   → `python test.py --config run.yaml` (đọc kết quả theo regex `[train|test]/...`)
7. Gom `stats.txt`, `*.pth`, `showimg/*.png` về `/kaggle/working/output`
   để **Save Version** lưu lại.

Cách đẩy mã nguồn lên Kaggle (2 lựa chọn, notebook hỗ trợ cả hai):

```text
(a) Đọc source từ path:
    Tạo Kaggle Dataset → upload cả folder repo (train.py, model.py, configs/, ...)
    → trong notebook, REPO = /kaggle/input/<dataset-slug>/

(b) Upload code.zip:
    7z a code.zip train.py test.py generate_conv.py density_show.py model.py utils.py \
        config.py requirements.txt configs/ chicken/
    → upload code.zip vào một Kaggle Dataset → notebook tự giải nén về /kaggle/working/src
```

> Kaggle mount dataset là **read-only** → mọi output (checkpoint, ảnh) đều ghi về
> `/kaggle/working`, không ghi đè vào `/kaggle/input`.
