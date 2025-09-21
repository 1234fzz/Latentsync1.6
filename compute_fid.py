# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# 最小 FID 实现（inception-v3 + 论文公式）

import os
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from PIL import Image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 Inception-v3 并去掉最后一层
model = inception_v3(pretrained=True, transform_input=False).eval().to(device)
model.fc = torch.nn.Identity()   # 提取 2048 维特征

transform = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def extract_features(video_path):
    """从 mp4 中提取 16 帧 → 2048 维特征"""
    from decord import VideoReader
    vr = VideoReader(video_path)
    frames = vr[20:36]                  # 16 帧
    feats = []
    for frm in frames.asnumpy():
        img = transform(Image.fromarray(frm)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img).cpu().numpy().squeeze()
        feats.append(feat)
    return np.array(feats)               # (16, 2048)

def compute_fid(pred_dir, gt_dir):
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.mp4')])
    gt_files   = sorted([f for f in os.listdir(gt_dir)   if f.endswith('.mp4')])
    assert len(pred_files) == len(gt_files), "视频数量必须一致"

    pred_feats = []
    gt_feats   = []
    for p, g in tqdm(zip(pred_files, gt_files), total=len(pred_files)):
        pred_feats.append(extract_features(os.path.join(pred_dir, p)))
        gt_feats.append(extract_features(os.path.join(gt_dir, g)))

    pred_feats = np.concatenate(pred_feats)  # (N*16, 2048)
    gt_feats   = np.concatenate(gt_feats)

    mu1, sigma1 = np.mean(pred_feats, axis=0), np.cov(pred_feats, rowvar=False)
    mu2, sigma2 = np.mean(gt_feats, axis=0), np.cov(gt_feats, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

if __name__ == "__main__":
    pred_dir = "/root/autodl-tmp/results_ls30__42"
    gt_dir   = "/root/autodl-tmp/HDTF256/Test_30"
    fid = compute_fid(pred_dir, gt_dir)
    print(f"FID: {fid:.3f}")