"""
Utility modules for Engine V5.4

Provides unified CPU/GPU backend management.
"""

from .backend import (
    Backend,
    DeviceInfo,
    get_backend,
    set_backend,
    get_device,
    get_dtype,
    ensure_tensor,
    cuda_available,
    get_cuda_device_count,
    select_device,
    print_device_info,
)

__all__ = [
    'Backend',
    'DeviceInfo',
    'get_backend',
    'set_backend',
    'get_device',
    'get_dtype',
    'ensure_tensor',
    'cuda_available',
    'get_cuda_device_count',
    'select_device',
    'print_device_info',
]
