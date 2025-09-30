import torch
import torch.nn as nn
import math, torch


# ───────────────────────────────────────────────────────────
#  SVD initialisation  (Ding et al. 2010, eq. (17)-(18))
# ───────────────────────────────────────────────────────────
@torch.no_grad()
def init_svd(A: torch.Tensor, K: int, eps: float = 1e-6):
    """
    Semi-NMF SVD init:
        • F_init ← U · sqrt(Σ)              (d_features, K)   (mixed signs OK)
        • G_init ← sqrt(Σ) · Vᵀ (⩾0)        (n_samples,  K)   (non-negative)

    A is (d_features, n_samples).
    """
    # 1) truncated SVD -----------------------------------------------------------
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    U   = U[:, :K]                       # (d, K)
    S   = S[:K]                          # (K,)
    Vh  = Vh[:K, :]                      # (K, n)

    sroot = S.sqrt()                     # √Σ  (K,)

    # 2) build initial factors ---------------------------------------------------
    F_init = U * sroot.unsqueeze(0)      # broadcast → (d, K)

    G_init = (sroot.unsqueeze(1) * Vh).T # (n, K)
    G_init = G_init.clamp_min(eps)       # strictly positive (Semi-NMF needs this)

    return F_init, G_init


@torch.no_grad()
def init_knn(
    A: torch.Tensor,
    K: int,
    n_iter: int = 15,
    eps: float = 1e-6,
    chunk_size: int = 10_000,
):
    """
    Semi-NMF k-means init (Ding et al., 2010):
      • F_init ← cluster centroids      (d_features, K)
      • G_init ← 1-of-K assignment      (n_samples, K)

    Works in chunks to avoid an n×K×d intermediate, and uses scatter_add
    for centroid updates instead of a Python loop.
    """
    d, n = A.shape
    device = A.device

    # pick K distinct columns as initial centres
    perm = torch.randperm(n, device=device)
    centres = A[:, perm[:K]].T  # (K, d)

    # we'll need X=(n, d) and a place to store labels
    X = A.T  # (n, d)
    labels = torch.empty(n, dtype=torch.long, device=device)

    for _ in range(n_iter):
        # --- 1) assign labels in chunks to avoid O(nK d) alloc ---
        c2 = (centres * centres).sum(dim=1).unsqueeze(0)
        for start in range(0, n, chunk_size):
            end   = min(n, start + chunk_size)
            block = X[start:end]                    # (b, d)  – b ≤ chunk_size

            # --- distance²(x, c) = ‖x‖² + ‖c‖² − 2·x·cᵀ ----------------------------
            # (1) ‖x‖²   : (b, 1)
            x2 = (block * block).sum(dim=1, keepdim=True)

            # (2) ‖c‖²   : (1, K)  – pre-compute once outside loop if you like

            # (3) x·cᵀ    : (b, K)
            dot = block @ centres.T                # matmul, no broadcast (b, d) @ (d, K)

            # combine → (b, K)
            dist2 = x2 + c2 - 2.0 * dot

            # labels for this chunk
            labels[start:end] = dist2.argmin(dim=1)

        # --- 2) update centroids with a single scatter_add ---
        # counts per cluster
        counts = torch.bincount(labels, minlength=K).unsqueeze(1)  # (K, 1)

        # sum of X for each label
        sums = torch.zeros(K, d, device=device)
        sums.scatter_add_(
            0,
            labels.view(-1, 1).expand(-1, d),
            X
        )  # (K, d)

        # avoid division by zero
        counts_clamped = counts.clamp_min(1)
        centres = sums / counts_clamped

        # handle empty clusters by re-sampling from X
        empty = (counts.squeeze(1) == 0).nonzero(as_tuple=False).view(-1)
        if empty.numel() > 0:
            rand_idx = torch.randint(0, n, (empty.numel(),), device=device)
            centres[empty] = X[rand_idx]

    # --- 3) build outputs ---
    F_init = centres.T  # (d, K)
    G_init = torch.zeros(n, K, device=device)
    G_init[torch.arange(n, device=device), labels] = 1.0
    G_init.clamp_min_(eps)

    return F_init, G_init

def wta_features(F: torch.Tensor, pct_keep: float = 0.05, by_abs: bool = True):
    """
    Hard WTA: keep only the top pct_keep fraction in each column of F.
    Vectorized via thresholding.
    """
    d, K = F.shape
    k = max(1, math.ceil(pct_keep * d))
    scores = F.abs() if by_abs else F

    # 1) get the top-k values for each of the K columns (shape (k,K))
    topk_vals, _ = torch.topk(scores, k, dim=0, sorted=True)

    # 2) the smallest of those top-k is your threshold per column
    thresh = topk_vals[-1, :]        # shape (K,)

    # 3) build a Boolean mask in one go
    mask = scores >= thresh.unsqueeze(0)  # (d, K)

    # 4) apply it
    F.mul_(mask)

def fix_hoyer_scale(F: torch.Tensor, G: torch.Tensor,
                     eps: float = 1e-8) -> None:
    # 1) normalize each column of G to unit ℓ₂
    col_norms = G.norm(dim=0, keepdim=True).clamp_min(eps)
    G.div_(col_norms)
    # 2) compensate in F
    F.mul_(col_norms.squeeze(0))

def wta_cols(G: torch.Tensor, pct_keep: float):
    """
    In-place WTA sparsity on each COLUMN of G.
    Keeps only the top `pct_keep` fraction of entries per column.
    """
    n, K = G.shape
    m    = max(1, math.ceil(pct_keep * n))
    # find top‐m indices in each column
    _, idx = torch.topk(G.abs(), m, dim=0, largest=True, sorted=False)  # (m,K)
    mask   = torch.zeros_like(G).scatter_(0, idx, 1.0)
    G.mul_(mask)


def _positive_part(X):
    return (X.abs() + X) * 0.5

def _negative_part(X):
    return (X.abs() - X) * 0.5

class NMFSemiNMF(nn.Module):
    """
    Semi-Nonnegative Matrix Factorization via multiplicative updates:
    A \approx W @ H  with W >= 0, H unconstrained.
    W: (n_samples x rank), H: (rank x n_features)
    """
    def __init__(self, rank, fitting_device='cpu', sparsity=0.01):
        super().__init__()
        self.rank = rank
        self.fitting_device = fitting_device
        # factors
        # nn.Parameter initialized in fit
        self.F_ = None # unconstrained (num_factors, K)
        self.G_ = None # positive (num_samples, K)
        self.sparsity = sparsity
        self.H = None
        self.W = None

    def fit(self, A, max_iter=500, tol=1e-4, reg=1e-6, patience=100, verbose=True, init="random"):
        """
        Fit semi-NMF: only W constrained to be >=0.
        H is updated in closed form, W multiplicatively.

        A: (n x m) data matrix
        rank: K
        reg: small regularizer for invertibility
        """
        # move data
        A = A.to(self.fitting_device)
        mlp_dim, num_samples  = A.shape
        K = self.rank

        
        if init == "knn":
            F0, G0 = init_knn(A, K)                  # D × K , n × K
        elif init == "svd":
            F0, G0 = init_svd(A, K)
        elif init == "random":
            G0 = torch.rand(num_samples, K, device=self.fitting_device).clamp_min(1e-6)
            F0 = torch.randn(mlp_dim, K, device=self.fitting_device)
        else:
            raise ValueError(f"Unknown init '{init}', use 'random' or 'knn'.")

        self.G_ = nn.Parameter(G0.to(self.fitting_device))
        self.F_ = nn.Parameter(F0.to(self.fitting_device))
        
        best_loss = float('inf')
        best_F = None
        best_G = None
        num_no_improve = 0
        prev_loss = float('inf')
        best_it = max_iter
        prev_loss = float('inf')
        with torch.no_grad():
            
            for it in range(max_iter):
                GtG = self.G_.T @ self.G_
                reg_I = torch.eye(K, device=self.fitting_device) * reg
                inv = torch.linalg.inv(GtG + reg_I)
                F_new = A @ self.G_ @ inv
                self.F_.data.copy_(F_new)
                wta_features(self.F_, pct_keep=self.sparsity)
                fix_hoyer_scale(self.F_, self.G_)
                
                # update G
                P = A.T @ self.F_
                Q = self.F_.T @ self.F_
                P_plus = _positive_part(P)
                P_minus = _negative_part(P)
                Q_plus = _positive_part(Q)
                Q_minus = _negative_part(Q)

                numer = P_plus + self.G_ @ Q_minus
                denom = P_minus + self.G_ @ Q_plus 
                G_new = self.G_ * torch.sqrt((numer / denom))
                self.G_.data.copy_(G_new)


                # ——— compute loss & check improvement —————————————
                A_approx = self.F_ @ self.G_.T
                loss = torch.norm(A - A_approx, p='fro')**2

                if loss.item() < best_loss - tol:
                    best_loss = loss.item()
                    best_F = self.F_.data.clone()
                    best_G = self.G_.data.clone()
                    num_no_improve = 0
                    best_it = it
                else:
                    num_no_improve += 1

                if verbose and (it % 50 == 0 or num_no_improve == 1):
                    print(f"Iter {it:4d}: loss={loss.item():.6f}  "
                          f"(best={best_loss:.6f}, no_improve={num_no_improve})")

                # early stopping
                if num_no_improve >= patience:
                    if verbose:
                        print(f"Stopping early at iter {it} "
                              f"(no improvement in {patience} iters)")
                    break
        with open("training_summary.log", "a") as logf:
            logf.write(
                f"{init},{self.sparsity},{best_it}\n"
            )
        if best_F is not None:
            with torch.no_grad():
                self.F_.data.copy_(best_F)
                self.G_.data.copy_(best_G)
        self.W = self.G_.detach().clone()
        self.H = self.F_.T
        return self
