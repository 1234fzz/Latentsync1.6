from svd_layer import SVDLinear
import torch

def test_error():
    torch.manual_seed(42)
    W = torch.randn(512, 512).cuda()
    b = torch.randn(512).cuda()
    ref = torch.nn.Linear(512, 512, bias=True).cuda()
    ref.weight.data = W
    ref.bias.data   = b

    svd = SVDLinear(W, b, rank=128).cuda()

    x = torch.randn(8, 512).cuda()
    with torch.no_grad():
        out_ref = ref(x)
        out_svd = svd(x)
    err = torch.norm(out_ref - out_svd) / torch.norm(out_ref)
    print(f'L2 relative error = {err:.6f}  (expect < 1e-3)')
    assert err < 1e-3, "精度不达标！"
    print('✅ 单元测试通过！')

if __name__ == '__main__':
    test_error()
