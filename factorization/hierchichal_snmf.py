import torch        
import torch.nn as nn
from factorization.seminmf import NMFSemiNMF
      
class HierarchicalNMFModule(nn.Module):
    def __init__(self, pretrained_layers):
        super().__init__()
        L = len(pretrained_layers)
        # Register each pretrained Wᵢ
        for i, layer in enumerate(pretrained_layers, start=1):
            W_i = layer.W.detach().clone()    # (prev_dim, rank_i)
            self.register_parameter(f"W{i}", nn.Parameter(W_i))
        # Register final H  (shape: r_L × hidden_dim)
        H_L = pretrained_layers[-1].H.detach().clone()
        self.register_parameter("H", nn.Parameter(H_L))
        self.num_layers = len(pretrained_layers)

    def forward(self):
        M = self.W1
        for i in range(2, self.num_layers+1):
            M = M @ getattr(self, f"W{i}")
        return M @ self.H



def train_hierarchical_nmf(
    A: torch.Tensor,
    ranks: list[int],
    device: str = "cpu",
    pretrain_kwargs: dict = None,
    ft_lr: float = 1e-3,
    ft_iters: int = 500,
    cls: any = NMFSemiNMF,
    cls_args = {},
    fine_tune=True
):
    """
    A:            (n_samples, hidden_dim) nonnegative data matrix
    ranks:        [r1, r2, …, r_L]
    """
    A = A.to(device)
    pretrain_kwargs = pretrain_kwargs or {}

    # 1) Greedy MU pre-training of each layer
    layers = []
    current_data = A               # shape: (n_samples, hidden_dim)
    for r in ranks:
        layer = cls(r,**cls_args, fitting_device=device)
        layer.fit(current_data.T, **pretrain_kwargs)
        layers.append(layer)
        current_data = layer.H.detach()  # shape: (r, hidden_dim)

    # 2) Build joint model for fine-tuning
    joint = HierarchicalNMFModule(layers).to(device)
    optimizer = torch.optim.Adam(joint.parameters(), lr=ft_lr)
    loss_fn   = nn.MSELoss()
    if fine_tune:
        # 3) Fine-tune end-to-end
        for epoch in range(ft_iters):
            optimizer.zero_grad()
            A_hat = joint()                 # (n_samples, hidden_dim)
            loss  = loss_fn(A_hat, A)
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print(f"[FT] Epoch {epoch:4d}  Loss = {loss.item():.6e}")

    return joint, layers




