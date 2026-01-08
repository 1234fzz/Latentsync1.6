import torch, torch.nn as nn
from latentsync.models.unet import UNet3DConditionModel
import sys
sys.path.append('/root/autodl-tmp/kk/svd_layer')
from svd_layer import SVDLinear

def save_s_parameters(rank=128):
    net = UNet3DConditionModel()
    replaced = 0
    for name, m in list(net.named_modules()):
        if isinstance(m, nn.Linear) or (isinstance(m, (nn.Conv2d, nn.Conv3d)) and all(k == 1 for k in m.kernel_size)):
            W = m.weight.data.squeeze()
            b = m.bias.data if m.bias is not None else torch.zeros(m.weight.shape[0], device=m.weight.device)
            svd_mod = SVDLinear(W, b, rank=rank)
            parent_name, child_name = name.rsplit('.', 1)
            parent = dict(net.named_modules())[parent_name]
            setattr(parent, child_name, svd_mod)
            replaced += 1
    print(f'已替换 {replaced} 层为 SVDLinear(rank={rank})')

    # 只保存 S 参数
    s_params = {name: p for name, p in net.named_parameters() if 'S' in name}
    torch.save(s_params, '/root/autodl-tmp/kk/output/latentsync_svd_r128_s_only.pth')
    print('已保存：latentsync_svd_r128_s_only.pth（S 参数）')

if __name__ == '__main__':
    save_s_parameters(rank=128)
