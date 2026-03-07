import torch

def quantize_tensor_symmetric(x, bit_width=8, min_val=None, max_val=None, channel_dim=None):
    """
    Symmetric quantization: Maps [-max_abs, max_abs] to [-2^(b-1), 2^(b-1)-1].
    Scale = max_abs / (2^(b-1) - 1)
    """
    q_min = -(2 ** (bit_width - 1))
    q_max = 2 ** (bit_width - 1) - 1

    if min_val is None or max_val is None:
        if channel_dim is None:
            max_abs = torch.max(torch.abs(x))
        else:
            # Calculate max_abs along specific dimensions
            # For shapes like (N, C, H, W) and channel_dim=1:
            # We want to keep dim 1, and reduce all others.
            # Easiest way: flatten all except channel_dim, then max.
            
            # Move channel dim to front, flatten rest
            x_t = x.transpose(0, channel_dim) # (C, N, H, W...)
            x_flat = x_t.reshape(x_t.shape[0], -1)
            max_abs = torch.max(torch.abs(x_flat), dim=1)[0] # (C,)
            
            # Reshape scale for broadcasting back to x
            # shape: [1, C, 1, 1...]
            shape = [1] * x.ndim
            shape[channel_dim] = x.shape[channel_dim]
            max_abs = max_abs.view(*shape)
    else:
        max_abs = max(abs(min_val), abs(max_val))
        
    scale = max_abs / q_max
    
    # Handle zero scales
    scale = torch.where(scale == 0, torch.tensor(1.0, device=x.device, dtype=x.dtype), scale)
        
    x_int = torch.round(x / scale).clamp(q_min, q_max)
    x_quant = x_int * scale
    
    return x_quant, scale, torch.tensor(0.0)

def quantize_tensor_asymmetric(x, bit_width=8, min_val=None, max_val=None, channel_dim=None):
    """
    Asymmetric quantization: Maps [min, max] to [0, 2^b - 1].
    Scale = (max - min) / (2^b - 1)
    Zero_point = -round(min / scale)
    """
    q_min = 0
    q_max = 2 ** bit_width - 1
    
    if min_val is None or max_val is None:
        if channel_dim is None:
            min_val = torch.min(x)
            max_val = torch.max(x)
        else:
            x_t = x.transpose(0, channel_dim)
            x_flat = x_t.reshape(x_t.shape[0], -1)
            
            min_val = torch.min(x_flat, dim=1)[0]
            max_val = torch.max(x_flat, dim=1)[0]
            
            shape = [1] * x.ndim
            shape[channel_dim] = x.shape[channel_dim]
            min_val = min_val.view(*shape)
            max_val = max_val.view(*shape)
    
    scale = (max_val - min_val) / (q_max - q_min)
    scale = torch.where(scale == 0, torch.tensor(1.0, device=x.device, dtype=x.dtype), scale)
        
    zero_point = torch.round(-min_val / scale)
    zero_point = zero_point.clamp(q_min, q_max)

    x_int = torch.round(x / scale + zero_point).clamp(q_min, q_max)
    x_quant = (x_int - zero_point) * scale
    
    return x_quant, scale, zero_point

def quantize_tensor(x, bit_width=8, method='symmetric', min_val=None, max_val=None, channel_dim=None):
    """
    Extensions: bit_width can now be a torch.Tensor for per-channel bits.
    """
    if isinstance(bit_width, (torch.Tensor, list)):
        # GRANULAR BIT-WIDTH MODE
        if channel_dim is None:
            raise ValueError("channel_dim must be specified for granular bit-widths")
        
        # Ensure bit_width is a tensor on the correct device
        if not isinstance(bit_width, torch.Tensor):
            bit_width = torch.tensor(bit_width, device=x.device)
        
        # Quantize each channel individually
        q_x = torch.zeros_like(x)
        # Permute to bring channel_dim to front
        permute_dims = list(range(x.ndim))
        permute_dims[0], permute_dims[channel_dim] = permute_dims[channel_dim], permute_dims[0]
        x_p = x.permute(*permute_dims)
        q_x_p = q_x.permute(*permute_dims)
        
        for i in range(x.shape[channel_dim]):
            bits = int(bit_width[i])
            q_channel, _, _ = quantize_tensor(x_p[i:i+1], bit_width=bits, method=method)
            q_x_p[i] = q_channel.squeeze(0)
            
        return q_x, None, None # Scales/ZPs are per-channel in this mode
        
    if method == 'symmetric':
        return quantize_tensor_symmetric(x, bit_width, min_val, max_val, channel_dim)
    elif method == 'asymmetric':
        return quantize_tensor_asymmetric(x, bit_width, min_val, max_val, channel_dim)
    else:
        raise ValueError(f"Unknown quantization method: {method}")

def block_quantize(x, block_size=16, bit_width=8, method='symmetric'):
    """
    Quantizes tensor in small blocks.
    Improves precision by narrowing the dynamic range per block.
    """
    orig_shape = x.shape
    x_flat = x.view(-1)
    num_blocks = (x_flat.numel() + block_size - 1) // block_size
    
    # Pad to block_size if necessary
    pad_size = num_blocks * block_size - x_flat.numel()
    if pad_size > 0:
        x_flat = torch.cat([x_flat, torch.zeros(pad_size, device=x.device)])
        
    x_blocks = x_flat.view(num_blocks, block_size)
    q_blocks = torch.zeros_like(x_blocks)
    
    for i in range(num_blocks):
        q_b, _, _ = quantize_tensor(x_blocks[i], bit_width=bit_width, method=method)
        q_blocks[i] = q_b
        
    q_x = q_blocks.view(-1)
    if pad_size > 0:
        q_x = q_x[:-pad_size]
        
    return q_x.view(*orig_shape)
