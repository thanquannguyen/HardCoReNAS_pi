import torch
import torch.nn as nn
import numpy as np

def decompose_layer(layer, rank_ratio=0.5):
    """
    Decompose a Conv2d layer using SVD.
    Approximates Conv2d(in, out, k, k) with two Conv2d layers:
    Conv2d(in, rank, k, k) -> Conv2d(rank, out, 1, 1)
    (This is one variation, often called Spatial SVD or similar depending on how we reshape)
    
    Here we implement a standard decomposition for 2D kernels.
    Weight shape: [out, in, k, k] -> flatten to [out, in*k*k] or similar?
    Actually for standard Conv decomposition (Zhang et al.), we often do:
    [out, in, k, k] -> [out, in*k*k] -> SVD -> U, S, V
    But keeping spatial dimensions is better.
    
    Let's use the scheme:
    W (out, in, k, k)
    Reshape to (out, in * k * k)
    SVD -> U (out, rank), S (rank,), V (rank, in*k*k)
    
    This replaces one conv with:
    1. Conv(in, rank, k, k) - using V (reshaped)
    2. Conv(rank, out, 1, 1) - using U * S
    """
    if not isinstance(layer, nn.Conv2d):
        return layer
    
    if layer.kernel_size == (1, 1):
        # 1x1 conv, just matrix multiplication
        # W: [out, in, 1, 1] -> [out, in]
        # SVD -> [out, rank] @ [rank, in]
        # Replace with two 1x1 convs
        return decompose_1x1(layer, rank_ratio)
        
    if layer.groups > 1:
        print(f"Skipping grouped convolution (groups={layer.groups})")
        return layer

    # For kxk conv
    W = layer.weight.data
    out_c, in_c, k1, k2 = W.shape
    
    # Reshape for SVD
    # We want to keep spatial kernel in the first layer usually?
    # Let's try CP decomposition or simple SVD on flattened weights.
    # Common approach: Decompose into (out, rank, 1, 1) and (rank, in, k, k)
    # W ~ (out, in*k*k)
    
    flat_W = W.view(out_c, -1)
    try:
        U, S, V = torch.linalg.svd(flat_W, full_matrices=False)
    except:
        return layer # SVD failed
        
    # Determine rank
    full_rank = min(out_c, in_c * k1 * k2)
    rank = max(1, int(full_rank * rank_ratio))
    
    U = U[:, :rank]
    S = S[:rank]
    V = V[:rank, :]
    
    # Reconstruct weights
    # Layer 1: (rank, in, k, k)
    # V is (rank, in*k*k)
    W1 = V.view(rank, in_c, k1, k2)
    
    # Layer 2: (out, rank, 1, 1)
    # U is (out, rank) -> U * S
    W2 = (U @ torch.diag(S)).view(out_c, rank, 1, 1)
    
    # Create new layers
    layer1 = nn.Conv2d(in_c, rank, kernel_size=(k1, k2), stride=layer.stride, 
                       padding=layer.padding, dilation=layer.dilation, groups=1, bias=False)
    layer1.weight.data = W1
    
    layer2 = nn.Conv2d(rank, out_c, kernel_size=1, stride=1, padding=0, bias=layer.bias is not None)
    layer2.weight.data = W2
    if layer.bias is not None:
        layer2.bias.data = layer.bias.data
        
    return nn.Sequential(layer1, layer2)

def decompose_1x1(layer, rank_ratio=0.5):
    W = layer.weight.data
    out_c, in_c, _, _ = W.shape
    
    flat_W = W.view(out_c, in_c)
    try:
        U, S, V = torch.linalg.svd(flat_W, full_matrices=False)
    except:
        return layer
        
    rank = max(1, int(min(out_c, in_c) * rank_ratio))
    
    U = U[:, :rank]
    S = S[:rank]
    V = V[:rank, :]
    
    W1 = V.view(rank, in_c, 1, 1)
    W2 = (U @ torch.diag(S)).view(out_c, rank, 1, 1)
    
    layer1 = nn.Conv2d(in_c, rank, 1, stride=layer.stride, padding=layer.padding, bias=False)
    layer1.weight.data = W1
    
    layer2 = nn.Conv2d(rank, out_c, 1, bias=layer.bias is not None)
    layer2.weight.data = W2
    if layer.bias is not None:
        layer2.bias.data = layer.bias.data
        
    return nn.Sequential(layer1, layer2)

def apply_low_rank(model, rank_ratio=0.5):
    """
    Recursively apply low-rank decomposition to all Conv2d layers.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            # Only decompose if it's worth it (e.g. large enough)
            if module.weight.numel() > 1000:
                print(f"Decomposing {name}...")
                new_module = decompose_layer(module, rank_ratio)
                setattr(model, name, new_module)
        else:
            apply_low_rank(module, rank_ratio)
    return model
