import torch
import torch.nn as nn

class ActivationQuantizer(nn.Module):
    """
    Module to quantize activations during Forward Pass.
    Used for QAT (Fake Quantization) and Calibration.
    """
    def __init__(self, bit_width=8, method='asymmetric', momentum=0.9, num_channels=None):
        super().__init__()
        self.bit_width = bit_width
        self.method = method
        self.momentum = momentum
        self.num_channels = num_channels
        
        # State buffers
        num_stats = num_channels if num_channels else 1
        self.register_buffer('running_min', torch.zeros(num_stats))
        self.register_buffer('running_max', torch.zeros(num_stats))
        self.register_buffer('scale', torch.ones(num_stats))
        self.register_buffer('zero_point', torch.zeros(num_stats))
        
        # Handle bit_width as buffer if it's granular
        if isinstance(bit_width, (list, torch.Tensor)):
            if not isinstance(bit_width, torch.Tensor):
                bit_width = torch.tensor(bit_width)
            self.register_buffer('bit_width_buffer', bit_width.float())
            self.is_granular_bits = True
        else:
            self.is_granular_bits = False
        
        # Flags
        self.initialized = False
        
    def forward(self, x):
        # CRITICAL: Ensure all buffers are on the same device as input
        target_device = x.device
        if self.running_min.device != target_device:
            self.running_min = self.running_min.to(target_device)
            self.running_max = self.running_max.to(target_device)
            self.scale = self.scale.to(target_device)
            self.zero_point = self.zero_point.to(target_device)
        
        if self.training:
            # Update ranges
            if self.num_channels:
                # Per-channel stats
                if x.ndim == 4:    # (N, C, H, W) -> (C, N*H*W)
                    x_t = x.transpose(0, 1).reshape(x.shape[1], -1)
                elif x.ndim == 2:  # (N, C) -> (C, N)
                    x_t = x.transpose(0, 1)
                else:              # Fallback
                    x_t = x.reshape(self.num_channels, -1)
                
                current_min = x_t.min(dim=1)[0]
                current_max = x_t.max(dim=1)[0]
            else:
                current_min = x.detach().min()
                current_max = x.detach().max()
            
            if not self.initialized:
                self.running_min.copy_(current_min)
                self.running_max.copy_(current_max)
                self.initialized = True
            else:
                self.running_min.mul_(self.momentum).add_(current_min * (1 - self.momentum))
                self.running_max.mul_(self.momentum).add_(current_max * (1 - self.momentum))
        
        # Handle granular bit_width (ensure q_min/q_max are compatible for clamp)
        if self.is_granular_bits:
            q_max = 2 ** self.bit_width_buffer - 1
            q_min = torch.zeros_like(q_max)
        else:
            q_max = 2 ** self.bit_width - 1
            q_min = 0
            
        scale = (self.running_max - self.running_min) / (q_max - q_min)
        scale = torch.clamp(scale, min=1e-8)  # Avoid division by zero
        
        zero_point = -self.running_min / scale
        zero_point = torch.round(zero_point).clamp(q_min, q_max)
        
        # Update buffers
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)
        
        # Fake Quantize
        if self.num_channels:
            # Broadcast scale and zero_point back to x
            shape = [1] * x.ndim
            if x.ndim >= 2:
                shape[1 if x.ndim == 4 else 1] = self.num_channels # (N, C, ...)
            else:
                shape[0] = self.num_channels
            s = self.scale.view(*shape)
            z = self.zero_point.view(*shape)
            
            # Handle per-channel q_min/q_max if bit_width is granular
            if self.is_granular_bits:
                qm = q_max.view(*shape)
                qn = q_min.view(*shape)
            else:
                qm = q_max
                qn = q_min
                
            x_int = torch.round(x / s + z).clamp(qn, qm)
            x_dq = (x_int - z) * s
        else:
            x_int = torch.round(x / self.scale + self.zero_point).clamp(q_min, q_max)
            x_dq = (x_int - self.zero_point) * self.scale
        
        return x_dq

    def extra_repr(self):
        return f"bit_width={self.bit_width}, method={self.method}"
