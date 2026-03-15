"""
Device Management for PyTorch GPU Computation

Centralized GPU device and dtype management for V5.1.
All computations run on CUDA with float64 precision.
"""

import torch
from typing import Union, Optional

# Global device and dtype settings
_device: Optional[torch.device] = None
_dtype: torch.dtype = torch.float64


class DeviceManager:
    """
    Centralized GPU device management.

    Ensures CUDA is available and provides utilities for tensor creation.
    V5.1 is GPU-only; raises error if CUDA is not available.
    """

    def __init__(self, device_id: int = 0):
        """
        Initialize device manager.

        Parameters
        ----------
        device_id : int
            CUDA device ID (default 0)
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU required for Engine V5.1. "
                "Please install PyTorch with CUDA support: "
                "pip install torch --index-url https://download.pytorch.org/whl/cu118"
            )

        self.device = torch.device(f'cuda:{device_id}')
        self.dtype = torch.float64

        # Set as global defaults
        global _device, _dtype
        _device = self.device
        _dtype = self.dtype

        # Print GPU info
        gpu_name = torch.cuda.get_device_name(device_id)
        gpu_mem = torch.cuda.get_device_properties(device_id).total_memory / 1e9
        compute_cap = torch.cuda.get_device_capability(device_id)

        print(f"Engine V5.1 initialized on GPU:")
        print(f"  Device: {gpu_name}")
        print(f"  Memory: {gpu_mem:.1f} GB")
        print(f"  Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
        print(f"  Precision: float64")

    def tensor(self, data, requires_grad: bool = False) -> torch.Tensor:
        """
        Create tensor on GPU with float64.

        Parameters
        ----------
        data : array-like
            Input data (list, numpy array, etc.)
        requires_grad : bool
            Whether to track gradients

        Returns
        -------
        torch.Tensor
            Tensor on GPU with float64 dtype
        """
        return torch.tensor(data, dtype=self.dtype, device=self.device,
                           requires_grad=requires_grad)

    def zeros(self, *shape, requires_grad: bool = False) -> torch.Tensor:
        """Create zero tensor on GPU."""
        return torch.zeros(*shape, dtype=self.dtype, device=self.device,
                          requires_grad=requires_grad)

    def ones(self, *shape, requires_grad: bool = False) -> torch.Tensor:
        """Create ones tensor on GPU."""
        return torch.ones(*shape, dtype=self.dtype, device=self.device,
                         requires_grad=requires_grad)

    def empty(self, *shape) -> torch.Tensor:
        """Create uninitialized tensor on GPU."""
        return torch.empty(*shape, dtype=self.dtype, device=self.device)

    def linspace(self, start: float, end: float, steps: int) -> torch.Tensor:
        """Create linspace tensor on GPU."""
        return torch.linspace(start, end, steps, dtype=self.dtype, device=self.device)

    def arange(self, start: float, end: float, step: float = 1.0) -> torch.Tensor:
        """Create arange tensor on GPU."""
        return torch.arange(start, end, step, dtype=self.dtype, device=self.device)

    def from_numpy(self, arr) -> torch.Tensor:
        """Convert numpy array to GPU tensor."""
        return torch.from_numpy(arr).to(dtype=self.dtype, device=self.device)

    def to_numpy(self, tensor: torch.Tensor):
        """Convert GPU tensor to numpy array."""
        return tensor.detach().cpu().numpy()

    def synchronize(self):
        """Synchronize CUDA operations (for timing)."""
        torch.cuda.synchronize(self.device)

    def memory_allocated(self) -> float:
        """Return allocated GPU memory in MB."""
        return torch.cuda.memory_allocated(self.device) / 1e6

    def memory_reserved(self) -> float:
        """Return reserved GPU memory in MB."""
        return torch.cuda.memory_reserved(self.device) / 1e6

    def empty_cache(self):
        """Clear unused GPU memory."""
        torch.cuda.empty_cache()


def get_device() -> torch.device:
    """Get the global CUDA device."""
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _device


def get_dtype() -> torch.dtype:
    """Get the global dtype (float64)."""
    return _dtype


def ensure_tensor(x: Union[torch.Tensor, float, list],
                  device: Optional[torch.device] = None,
                  dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Ensure input is a tensor on the correct device.

    Parameters
    ----------
    x : tensor, float, or list
        Input to convert
    device : torch.device, optional
        Target device (default: global device)
    dtype : torch.dtype, optional
        Target dtype (default: float64)

    Returns
    -------
    torch.Tensor
        Tensor on specified device
    """
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = get_dtype()

    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    else:
        return torch.tensor(x, dtype=dtype, device=device)
