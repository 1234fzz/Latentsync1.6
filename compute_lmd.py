# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# 最小 LMD 实现（68 点 landmark 欧氏距离）

import os
import numpy as np
import cv2
from decord import VideoReader
from tqdm import tqdm

def extract_landmarks(video_path):
    """返回 16 帧 × 68 × 2 的 landmark 坐标（简化版）"""
    vr = VideoReader(video_path)
    frames = vr[20:36]          # 16 帧
    landmarks = []
    for frm in frames.asnumpy():   # ← 已修复可迭代
        gray = cv2.cvtColor(frm, cv2.COLOR_RGB2GRAY)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = detector.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            landmarks.append(np.zeros((68, 2)))
            continue
        x, y, w, h = faces[0]
        # 简化：中心点 × 68 点
        lmk = np.array([[x + w//2, y + h//2]] * 68)
        landmarks.append(lmk)
    return np.array(landmarks)   # (16, 68, 2)

def compute_lmd(pred_dir, gt_dir):
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.mp4')])
    gt_files   = sorted([f for f in os.listdir(gt_dir)   if f.endswith('.mp4')])
    assert len(pred_files) == len(gt_files), "数量必须一致"

    lmd_list = []
    for p, g in tqdm(zip(pred_files, gt_files), total=len(pred_files)):
        pred_lm = extract_landmarks(os.path.join(pred_dir, p))
        gt_lm   = extract_landmarks(os.path.join(gt_dir, g))

        # 逐帧 L2 距离
        lmd_vals = []
        for pl, gl in zip(pred_lm, gt_lm):
            val = np.linalg.norm(pl - gl, axis=1).mean()
            lmd_vals.append(val)
        lmd_list.append(np.mean(lmd_vals))

    return np.mean(lmd_list)

if __name__ == "__main__":
    pred_dir = "/root/autodl-tmp/results_ls30__42"
    gt_dir   = "/root/autodl-tmp/HDTF256/Test_30"
    lmd_val = compute_lmd(pred_dir, gt_dir)
    print(f"LMD: {lmd_val:.3f}")