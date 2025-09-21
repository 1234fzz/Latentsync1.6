# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# 最小 SSIM 实现（16 帧平均）

import os
import torch
import torchvision.transforms as T
import numpy as np
from torchvision.models import inception_v3
from torchmetrics.functional import structural_similarity_index_measure as ssim
from decord import VideoReader
from tqdm import tqdm
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# 统一预处理
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

def extract_frames(video_path):
    vr = VideoReader(video_path)
    frames = vr[20:36]          # 16 帧
    return [transform(Image.fromarray(frm)).unsqueeze(0) for frm in frames.asnumpy()]

def compute_ssim(pred_dir, gt_dir):
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.mp4')])
    gt_files   = sorted([f for f in os.listdir(gt_dir)   if f.endswith('.mp4')])
    assert len(pred_files) == len(gt_files), "数量必须一致"

    ssim_list = []
    for p, g in tqdm(zip(pred_files, gt_files), total=len(pred_files)):
        pred_frames = extract_frames(os.path.join(pred_dir, p))
        gt_frames   = extract_frames(os.path.join(gt_dir, g))

        ssim_vals = []
        for pf, gf in zip(pred_frames, gt_frames):
            pf = pf.to(device)
            gf = gf.to(device)
            val = ssim(pf, gf, data_range=1.0).item()
            ssim_vals.append(val)
        ssim_list.append(np.mean(ssim_vals))

    return np.mean(ssim_list)

if __name__ == "__main__":
    pred_dir = "/root/autodl-tmp/results_ls30__42"
    gt_dir   = "/root/autodl-tmp/HDTF256/Test_30"
    ssim_val = compute_ssim(pred_dir, gt_dir)
    print(f"SSIM: {ssim_val:.3f}")