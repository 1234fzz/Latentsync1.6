import torch, torch.nn as nn
from latentsync.models.unet import UNet3DConditionModel
import sys
sys.path.append('/root/autodl-tmp/kk/svd_layer')
from svd_layer import SVDLinear

def replace_linear_with_svd(model, rank=128):
    replaced = 0
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1)):
            W = m.weight.data.squeeze()
            b = m.bias.data if m.bias is not None else None
            svd_mod = SVDLinear(W, b, rank=rank)
            parent_name, child_name = name.rsplit('.', 1)
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, child_name, svd_mod)
            replaced += 1
    print(f'已替换 {replaced} 层为 SVDLinear(rank={rank})')
    return model

if __name__ == '__main__':
    # 空壳模型（已初始化）
    net = UNet3DConditionModel()
    # 不 load_state_dict，直接替换权重矩阵
    net_svd = replace_linear_with_svd(net, rank=128)
    # 保存新结构
    torch.save(net_svd.state_dict(), '/root/autodl-tmp/kk/output/latentsync_svd_r128.pth')
    print('已保存：latentsync_svd_r128.pth')
