# Resnet50FPN được import để pickle của backbone.pth unpickle được (model.Resnet50FPN)
from model import DensityRegressor, Resnet50FPN
from utils import MAPS, Transform, extract_features, visualize, set_conv_dir, load_checkpoint
from PIL import Image
import os
import torch
import json
import numpy as np
from tqdm import tqdm
import time

from config import load_config

# Mọi tham số đọc từ YAML -> chỉ còn 1 đối số dòng lệnh:  --config configs/xxx.yaml
cfg = load_config("test")
set_conv_dir(cfg.conv_dir)

data_path = cfg.data_path
anno_file = data_path + cfg.anno_file
data_split_file = data_path + cfg.split_file
im_dir = data_path + cfg.images_dir

output_dir = cfg.output_dir
os.makedirs(output_dir, exist_ok=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)

resnet50_conv = load_checkpoint(cfg.backbone_ckpt)
resnet50_conv.cuda()
resnet50_conv.eval()

regressor = DensityRegressor(10, pool='mean')
regressor.load_state_dict(load_checkpoint(cfg.regressor_ckpt))
regressor.cuda()
regressor.eval()

with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)

cnt = 0
SAE = 0  # sum of absolute errors
SSE = 0  # sum of square errors

print("Evaluation on {} data".format(cfg.eval_split))
im_ids = data_split[cfg.eval_split]
pbar = tqdm(im_ids)
t_all = []
for im_id in pbar:
    t1 = time.time()
    anno = annotations[im_id]
    dots = np.array(anno['points'])

    image = Image.open('{}/{}'.format(im_dir, im_id))
    image.load()
    sample = {'image': image}
    sample = Transform(sample)
    image = sample['image']
    image = image.cuda()

    with torch.no_grad():
        output = regressor(extract_features(resnet50_conv, image.unsqueeze(0), MAPS))

    gt_cnt = dots.shape[0]
    pred_cnt = output.sum().item()
    t2 = time.time()
    t_all.append(t2 - t1)
    cnt = cnt + 1
    err = abs(gt_cnt - pred_cnt)
    SAE += err
    SSE += err**2

    pbar.set_description('{:<8}: actual-predicted: {:6d}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}'.\
                         format(im_id, gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE/cnt, (SSE/cnt)**0.5))
    print("")
    rslt_file = "{}/{}_myout.png".format(output_dir, im_id)
    """
    Save the visualization
    """
    visualize(image.detach().cpu(), output.detach().cpu(), rslt_file, dots)
print('average time:', np.mean(t_all) / 1)
print('average fps:', 1 / np.mean(t_all))
print('On {} data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(cfg.eval_split, SAE/cnt, (SSE/cnt)**0.5))
