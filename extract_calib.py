import json, cv2, os, tqdm, random
os.makedirs('/root/autodl-tmp/data/calib_latentsync', exist_ok=True)

with open('/root/autodl-tmp/data/train.jsonl') as f:
    lines = [json.loads(l) for l in f]

n = len(lines)                      # 369 就 369
random.shuffle(lines)               # 打乱顺序
for i, item in enumerate(tqdm.tqdm(lines)):
    cap = cv2.VideoCapture(item['video_path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, item['sync_frame'])
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (512, 512))
        cv2.imwrite(f'/root/autodl-tmp/data/calib_latentsync/calib_{i:05d}.png', frame)
    cap.release()

print(f'{n} 张校准图抽取完成！')
