"""
Backend Abstraction for CPU/GPU Computation

Provides unified device management for Engine V5.3, supporting both
CPU and CUDA GPU backends via PyTorch.

Usage:
    from utils import Backend, get_backend

    # Auto-detect best device
    backend = Backend()

    # Force CPU mode
    backend = Backend(device='cpu')

    # Force GPU mode
    backend = Backend(device='cuda')

    # Create tensors on the backend
    x = backend.zeros(100, 100)
    y = backend.tensor([1.0, 2.0, 3.0])
"""

import torch
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Any
import warnings


@dataclass
class DeviceInfo:
    """Information about the compute device."""
    name: str
    type: str  # 'cpu' or 'cuda'
    index: Optional[int]
    memory_total_gb: Optional[float]
    memory_allocated_mb: Optional[float]
    compute_capability: Optional[Tuple[int, int]]

    def __repr__(self) -> str:
        if self.type == 'cpu':
            return f"DeviceInfo(CPU)"
        return (f"DeviceInfo({self.name}, "
                f"memory={self.memory_total_gb:.1f}GB, "
                f"compute={self.compute_capability})")


# Global backend instance
_global_backend: Optional['Backend'] = None


class Backend:
    """
    Unified backend for CPU/GPU computation.

    Provides device-aware tensor creation and utilities.
    Automatically selects CUDA if available, otherwise falls back to CPU.

    Parameters
    ----------
    device : str, optional
        Device specification: 'auto', 'cpu', 'cuda', or 'cuda:N'.
        Default 'auto' selects CUDA if available.
    dtype : torch.dtype, optional
        Default data type. Default float64 for numerical accuracy.
    verbose : bool
        Print device information on initialization.

    Attributes
    ----------
    device : torch.device
        The compute device
    dtype : torch.dtype
        Default data type
    is_cuda : bool
        Whether using CUDA GPU
    """

    def __init__(
        self,
        device: str = 'auto',
        dtype: torch.dtype = torch.float64,
        verbose: bool = True
    ):
        self.dtype = dtype

        # Resolve device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.is_cuda = self.device.type == 'cuda'
        self.is_cpu = self.device.type == 'cpu'

        # Get device info
        self._device_info = self._get_device_info()

        if verbose:
            self._print_info()

    def _get_device_info(self) -> DeviceInfo:
        """Get information about the current device."""
        if self.is_cuda:
            idx = self.device.index or 0
            props = torch.cuda.get_device_properties(idx)
            return DeviceInfo(
                name=props.name,
                type='cuda',
                index=idx,
                memory_total_gb=props.total_memory / 1e9,
                memory_allocated_mb=torch.cuda.memory_allocated(idx) / 1e6,
                compute_capability=(props.major, props.minor)
            )
        else:
            import platform
            return DeviceInfo(
                name=f"CPU ({platform.processor() or 'Unknown'})",
                type='cpu',
                index=None,
                memory_total_gb=None,
                memory_allocated_mb=None,
                compute_capability=None
            )

    def _print_info(self):
        """Print device information."""
        info = self._device_info
        print(f"Engine V5.3 Backend:")
        print(f"  Device: {info.name}")
        print(f"  Type: {info.type.upper()}")
        if info.memory_total_gb:
            print(f"  Memory: {info.memory_total_gb:.1f} GB")
        if info.compute_capability:
            print(f"  Compute: SM {info.compute_capability[0]}.{info.compute_capability[1]}")
        print(f"  Precision: {self.dtype}")

    @property
    def device_info(self) -> DeviceInfo:
        """Get device information."""
        return self._device_info

    # =========================================================================
    # Tensor Creation
    # =========================================================================

    def tensor(
        self,
        data: Any,
        dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """
        Create tensor from data on this device.

        Parameters
        ----------
        data : array-like
            Input data
        dtype : torch.dtype, optional
            Override default dtype
        requires_grad : bool
            Enable gradient tracking
        """
        if dtype is None:
            dtype = self.dtype
        return torch.tensor(data, dtype=dtype, device=self.device,
                           requires_grad=requires_grad)

    def zeros(
        self,
        *shape,
        dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """Create zero tensor on this device."""
        if dtype is None:
            dtype = self.dtype
        return torch.zeros(*shape, dtype=dtype, device=self.device,
                          requires_grad=requires_grad)

    def ones(
        self,
        *shape,
        dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """Create ones tensor on this device."""
        if dtype is None:
            dtype = self.dtype
        return torch.ones(*shape, dtype=dtype, device=self.device,
                         requires_grad=requires_grad)

    def empty(
        self,
        *shape,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Create uninitialized tensor on this device."""
        if dtype is None:
            dtype = self.dtype
        return torch.empty(*shape, dtype=dtype, device=self.device)

    def full(
        self,
        shape,
        fill_value: float,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Create tensor filled with value on this device."""
        if dtype is None:
            dtype = self.dtype
        return torch.full(shape, fill_value, dtype=dtype, device=self.device)

    def linspace(
        self,
        start: float,
        end: float,
        steps: int,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Create linspace tensor on this device."""
        if dtype is None:
            dtype = self.dtype
        return torch.linspace(start, end, steps, dtype=dtype, device=self.device)

    def arange(
        self,
        start: float,
        end: float,
        step: float = 1.0,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Create arange tensor on this device."""
        if dtype is None:
            dtype = self.dtype
        return torch.arange(start, end, step, dtype=dtype, device=self.device)

    def eye(
        self,
        n: int,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Create identity matrix on this device."""
        if dtype is None:
            dtype = self.dtype
        return torch.eye(n, dtype=dtype, device=self.device)

    # =========================================================================
    # Conversions
    # =========================================================================

    def from_numpy(self, arr) -> torch.Tensor:
        """Convert numpy array to tensor on this device."""
        return torch.from_numpy(arr).to(dtype=self.dtype, device=self.device)

    def to_numpy(self, tensor: torch.Tensor):
        """Convert tensor to numpy array (moves to CPU if needed)."""
        return tensor.detach().cpu().numpy()

    def to_device(
        self,
        x: Union[torch.Tensor, Any],
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        Ensure input is a tensor on this device.

        Parameters
        ----------
        x : tensor, float, or array-like
            Input to convert
        dtype : torch.dtype, optional
            Target dtype (default: self.dtype)
        """
        if dtype is None:
            dtype = self.dtype

        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=dtype)
        else:
            return torch.tensor(x, dtype=dtype, device=self.device)

    # =========================================================================
    # Memory Management
    # =========================================================================

    def synchronize(self):
        """Synchronize CUDA operations (no-op on CPU)."""
        if self.is_cuda:
            torch.cuda.synchronize(self.device)

    def memory_allocated(self) -> float:
        """Return allocated memory in MB (0 for CPU)."""
        if self.is_cuda:
            return torch.cuda.memory_allocated(self.device) / 1e6
        return 0.0

    def memory_reserved(self) -> float:
        """Return reserved memory in MB (0 for CPU)."""
        if self.is_cuda:
            return torch.cuda.memory_reserved(self.device) / 1e6
        return 0.0

    def empty_cache(self):
        """Clear unused GPU memory (no-op on CPU)."""
        if self.is_cuda:
            torch.cuda.empty_cache()

    # =========================================================================
    # Utility
    # =========================================================================

    def __repr__(self) -> str:
        return f"Backend(device={self.device}, dtype={self.dtype})"

    def benchmark_available(self) -> bool:
        """Check if CUDA benchmarking is possible."""
        return self.is_cuda and torch.backends.cudnn.is_available()

    def set_benchmark_mode(self, enabled: bool = True):
        """Enable/disable cuDNN benchmark mode for optimal convolution algorithms."""
        if self.is_cuda:
            torch.backends.cudnn.benchmark = enabled


# =============================================================================
# Global Backend Functions
# =============================================================================

def get_backend() -> Backend:
    """
    Get the global backend instance.

    Creates a new backend with auto device selection if none exists.
    """
    global _global_backend
    if _global_backend is None:
        _global_backend = Backend(verbose=False)
    return _global_backend


def set_backend(backend: Backend) -> None:
    """Set the global backend instance."""
    global _global_backend
    _global_backend = backend


def get_device() -> torch.device:
    """Get the device from the global backend."""
    return get_backend().device


def get_dtype() -> torch.dtype:
    """Get the dtype from the global backend."""
    return get_backend().dtype


def ensure_tensor(
    x: Union[torch.Tensor, float, list],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Ensure input is a tensor on the specified device.

    Parameters
    ----------
    x : tensor, float, or list
        Input to convert
    device : torch.device, optional
        Target device (default: global backend device)
    dtype : torch.dtype, optional
        Target dtype (default: float64)
    """
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = get_dtype()

    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    else:
        return torch.tensor(x, dtype=dtype, device=device)


# =============================================================================
# Device Selection Utilities
# =============================================================================

def cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def get_cuda_device_count() -> int:
    """Get number of CUDA devices."""
    return torch.cuda.device_count() if cuda_available() else 0


def select_device(prefer_cuda: bool = True) -> torch.device:
    """
    Select the best available device.

    Parameters
    ----------
    prefer_cuda : bool
        If True, prefer CUDA over CPU when available
    """
    if prefer_cuda and cuda_available():
        return torch.device('cuda')
    return torch.device('cpu')


def print_device_info():
    """Print information about all available devices."""
    print("Available Devices:")
    print(f"  CPU: Available")

    if cuda_available():
        for i in range(get_cuda_device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  CUDA:{i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    else:
        print("  CUDA: Not available")
