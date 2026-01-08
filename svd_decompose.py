import torch, matplotlib.pyplot as plt

def low_rank_approx(W, rank):
    U, S, V = torch.svd(W.float())
    U_r, S_r, V_r = U[:, :rank], S[:rank], V[:, :rank]
    return U_r @ torch.diag(S_r) @ V_r.T, (U, S, V)   # 返回原始 U,S,V

if __name__ == '__main__':
    R = 512
    W = torch.randn(R, R).cuda()
    ranks, errors = [], []
    for r in [32, 64, 128, 256, 384, 448, 512]:
        W_hat, (U, S, V) = low_rank_approx(W, r)
        err = torch.norm(W - W_hat).item() / torch.norm(W).item()
        ranks.append(r); errors.append(err)
        torch.save({'U': U[:, :r], 'S': S[:r], 'V': V[:, :r]}, f'svd_r{r}.pt')
    plt.plot(ranks, errors, marker='o')
    plt.xlabel('Rank'); plt.ylabel('Relative Error')
    plt.title('SVD Low-Rank Approximation Error')
    plt.grid(True)
    plt.savefig('rank_error.pdf')
    print('图已保存：rank_error.pdf')
