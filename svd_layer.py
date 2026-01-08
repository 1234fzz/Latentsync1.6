import torch
import torch.nn.functional as F

class SVDLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, rank: int):
        super().__init__()
        U, S, V = torch.svd(weight.float(), some=False)
        U_r, S_r, V_r = U[:, :rank], S[:rank], V[:, :rank]
        # 注册 U、S、V 为可训练参数
        self.register_parameter('U', torch.nn.Parameter(U_r))
        self.register_parameter('S', torch.nn.Parameter(S_r))
        self.register_parameter('V', torch.nn.Parameter(V_r))
        self.bias = torch.nn.Parameter(bias) if bias is not None else None
        self.rank = rank

    def forward(self, x):
        # 前向：x → V^T → S → U → +bias
        tmp = F.linear(x, self.V.T)          # [B, rank]
        tmp = tmp * self.S                   # [B, rank] 对角缩放
        return self.bias + F.linear(tmp, self.U)  # [B, out]
