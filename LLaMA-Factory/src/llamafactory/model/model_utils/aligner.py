import torch

def lir(embed, c, r=1):
    """
    Remove the language information from embeddings.
    Args:
        embed (torch.Tensor): Input embeddings.
        c (torch.Tensor): Principal components for projection.
        r (int): Number of components to remove.
    Returns:
        torch.Tensor: Embeddings with language information removed.
    """
    
    # Ensure both tensors are on the same device
    # device = embed.device
    c = c.to(embed)

    c = c / torch.norm(c, dim=0, keepdim=True)
    proj = torch.matmul(embed, c[:, :r])
    return embed - torch.matmul(proj, c[:, :r].T)

def lsar(W, k, returns_all=False):
    """
    Low-rank subtraction-based alignment representation.
    Args:
        W (torch.Tensor): Input matrix of shape (d, D).
        k (int): Rank for approximation.
        returns_all (bool): Whether to return additional matrices.
    Returns:
        torch.Tensor: New alignment representation.
    """
    d, D = W.shape
    wc = W @ torch.ones(D, device=W.device) / D
    u, s, vh = torch.linalg.svd(W - wc.view(-1, 1) @ torch.ones(1, D, device=W.device), full_matrices=False)
    Ws = u[:, :k]
    Gamma = vh[:k, :].T @ torch.diag(s[:k])
    best_fit_W = wc.view(-1, 1) @ torch.ones(1, D, device=W.device) + Ws @ Gamma.T

    wc_new = torch.linalg.pinv(best_fit_W).T @ torch.ones(D, device=W.device)
    wc_new /= (wc_new ** 2).sum()
    prod = best_fit_W - wc_new.view(-1, 1) @ torch.ones(1, D, device=W.device)

    print(torch.norm(W - wc_new.view(-1, 1) @ torch.ones(1, D, device=W.device) - prod, dim=0))

    if returns_all:
        u, s, vh = torch.linalg.svd(prod, full_matrices=False)
        Ws_new = u[:, :k]
        Gamma_new = vh[:k, :].T @ torch.diag(s[:k])
        return wc_new, prod, Ws_new, Gamma_new

    return wc_new, prod

def build_aligner(align_method, lan_emb):
    """
    Build an alignment function based on the specified method.
    Args:
        align_method (str): Alignment method.
        lan_emb (dict): Language embeddings {language: torch.Tensor}.
    Returns:
        function: Alignment function.
    """
    # Ensure all embeddings are torch.Tensors
    lan_emb = {lan: torch.tensor(emb, dtype=torch.float32) if not isinstance(emb, torch.Tensor) else emb 
                for lan, emb in lan_emb.items()}
    
    if align_method == "none":
        align = lambda embed, _: embed
    elif align_method == "demean":
        lan_mean_emb = {lan: torch.mean(emb, dim=0) for lan, emb in lan_emb.items()}
        align = lambda embed, lan: embed - lan_mean_emb[lan]
    elif align_method.startswith("lsar+"):
        rank = align_method.split("+")[-1]
        rank = rank.split("/")
        if len(rank) == 1:
            n_removed = rank = int(rank[0])
        elif len(rank) == 2:
            n_removed, rank = int(rank[0]), int(rank[1])
        else:
            raise ValueError(f"Invalid align_method: {align_method}")
        lan_mean_emb = {lan: torch.mean(emb, dim=0) for lan, emb in lan_emb.items()}
        _, _, Ws, _ = lsar(torch.stack(list(lan_mean_emb.values())).T, rank, returns_all=True)
        align = lambda embed, lan: lir(embed, Ws, r=n_removed)
    elif align_method.startswith("lir+"):
        rank = int(align_method.split("+")[-1])
        lan_svd = {lan: torch.linalg.svd(emb.T, full_matrices=False) for lan, emb in lan_emb.items()}
        lan_info = {lan: u[:, :20] for lan, (u, diag, vh) in lan_svd.items()}
        align = lambda embed, lan: lir(embed, lan_info[lan], r=rank)
    else:
        raise NotImplementedError

    return align
