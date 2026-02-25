#!/usr/bin/env python3
"""
Test Backend CPU/GPU Toggle System

Tests the unified backend abstraction for Engine V5.3.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from utils import (
    Backend, DeviceInfo,
    get_backend, set_backend, get_device, get_dtype,
    ensure_tensor, cuda_available, select_device, print_device_info
)


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_backend_creation():
    """Test backend creation on different devices."""
    print_header("Test 1: Backend Creation")

    # Test CPU backend
    print("\n  [1.1] CPU Backend")
    cpu_backend = Backend(device='cpu', verbose=False)
    assert cpu_backend.is_cpu
    assert not cpu_backend.is_cuda
    assert cpu_backend.device.type == 'cpu'
    print(f"    ✓ CPU backend created: {cpu_backend}")

    # Test auto backend
    print("\n  [1.2] Auto Backend")
    auto_backend = Backend(device='auto', verbose=False)
    print(f"    ✓ Auto backend selected: {auto_backend.device}")

    # Test CUDA backend (if available)
    if cuda_available():
        print("\n  [1.3] CUDA Backend")
        cuda_backend = Backend(device='cuda', verbose=False)
        assert cuda_backend.is_cuda
        assert not cuda_backend.is_cpu
        print(f"    ✓ CUDA backend created: {cuda_backend}")
    else:
        print("\n  [1.3] CUDA Backend - SKIPPED (not available)")

    print("\n  All backend creation tests passed!")


def test_tensor_creation():
    """Test tensor creation on backends."""
    print_header("Test 2: Tensor Creation")

    for device in ['cpu'] + (['cuda'] if cuda_available() else []):
        print(f"\n  [{device.upper()}]")
        backend = Backend(device=device, verbose=False)

        # zeros
        z = backend.zeros(10, 10)
        assert z.shape == (10, 10)
        assert z.device.type == device
        assert z.dtype == torch.float64
        print(f"    ✓ zeros: {z.shape}, device={z.device}")

        # ones
        o = backend.ones(5)
        assert o.shape == (5,)
        assert (o == 1.0).all()
        print(f"    ✓ ones: {o.shape}")

        # linspace
        ls = backend.linspace(-100, 80, 1001)
        assert ls.shape == (1001,)
        assert ls[0] == -100.0
        assert ls[-1] == 80.0
        print(f"    ✓ linspace: {ls.shape}, range=[{ls[0]}, {ls[-1]}]")

        # tensor from list
        t = backend.tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.device.type == device
        print(f"    ✓ tensor: {t.shape}")

        # from numpy
        arr = np.random.randn(100)
        tn = backend.from_numpy(arr)
        assert tn.shape == (100,)
        assert tn.device.type == device
        print(f"    ✓ from_numpy: {tn.shape}")

        # to numpy
        back_arr = backend.to_numpy(tn)
        assert isinstance(back_arr, np.ndarray)
        assert np.allclose(arr, back_arr)
        print(f"    ✓ to_numpy: round-trip verified")

    print("\n  All tensor creation tests passed!")


def test_global_backend():
    """Test global backend functions."""
    print_header("Test 3: Global Backend")

    # Get default backend
    backend = get_backend()
    print(f"  Default backend: {backend}")

    # Get device and dtype
    device = get_device()
    dtype = get_dtype()
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")

    # Set new backend
    new_backend = Backend(device='cpu', verbose=False)
    set_backend(new_backend)
    assert get_device().type == 'cpu'
    print(f"  ✓ set_backend works: now on {get_device()}")

    # ensure_tensor
    x = ensure_tensor([1.0, 2.0, 3.0])
    assert isinstance(x, torch.Tensor)
    assert x.device.type == 'cpu'
    print(f"  ✓ ensure_tensor: {x.shape} on {x.device}")

    # Reset to auto
    set_backend(Backend(device='auto', verbose=False))
    print(f"  ✓ Reset to auto: {get_device()}")

    print("\n  All global backend tests passed!")


def test_device_info():
    """Test device info retrieval."""
    print_header("Test 4: Device Info")

    # CPU info
    cpu_backend = Backend(device='cpu', verbose=False)
    info = cpu_backend.device_info
    assert isinstance(info, DeviceInfo)
    assert info.type == 'cpu'
    print(f"  CPU: {info}")

    # CUDA info
    if cuda_available():
        cuda_backend = Backend(device='cuda', verbose=False)
        info = cuda_backend.device_info
        assert info.type == 'cuda'
        assert info.memory_total_gb > 0
        print(f"  CUDA: {info}")
    else:
        print("  CUDA: Not available")

    print("\n  All device info tests passed!")


def test_memory_management():
    """Test memory management functions."""
    print_header("Test 5: Memory Management")

    backend = Backend(device='auto', verbose=False)

    if backend.is_cuda:
        # Allocate some memory
        x = backend.zeros(1000, 1000)
        backend.synchronize()

        mem = backend.memory_allocated()
        print(f"  Allocated: {mem:.2f} MB")

        reserved = backend.memory_reserved()
        print(f"  Reserved: {reserved:.2f} MB")

        # Empty cache
        del x
        backend.empty_cache()
        print(f"  ✓ empty_cache called")
    else:
        print("  Memory management is no-op on CPU")
        backend.synchronize()  # Should be no-op
        backend.empty_cache()  # Should be no-op
        print(f"  ✓ CPU no-op functions work")

    print("\n  All memory management tests passed!")


def test_lut_on_both_backends():
    """Test LUT works on both CPU and GPU."""
    print_header("Test 6: LUT on Both Backends")

    from ionic.lut import TTP06LUT

    for device in ['cpu'] + (['cuda'] if cuda_available() else []):
        print(f"\n  [{device.upper()}]")

        # Create LUT on device
        lut = TTP06LUT(device=device)
        print(f"    ✓ LUT created with {len(lut.tables)} tables")

        # Test lookup
        V = torch.linspace(-80, 40, 100, device=device, dtype=torch.float64)
        m_inf = lut.lookup('m_inf', V)

        assert m_inf.device.type == device
        assert m_inf.shape == V.shape
        assert not torch.isnan(m_inf).any()
        print(f"    ✓ Lookup works: m_inf range [{m_inf.min():.4f}, {m_inf.max():.4f}]")

        # Test batch lookup
        gating = lut.get_all_gating(V, celltype_is_endo=True)
        assert 'm_inf' in gating
        assert 's_inf' in gating
        print(f"    ✓ Batch gating lookup: {len(gating)} variables")

    print("\n  All LUT backend tests passed!")


def test_model_on_both_backends():
    """Test TTP06 model works on both CPU and GPU."""
    print_header("Test 7: TTP06 Model on Both Backends")

    from ionic import TTP06Model, CellType

    for device in ['cpu'] + (['cuda'] if cuda_available() else []):
        print(f"\n  [{device.upper()}]")

        # Create model
        model = TTP06Model(celltype=CellType.EPI, device=device, use_lut=True)
        print(f"    ✓ Model created on {device}")

        # Get initial state
        state = model.get_initial_state()
        assert state.device.type == device
        print(f"    ✓ Initial state on {state.device}")

        # Run a few steps
        dt = 0.01
        for i in range(100):
            t = i * dt
            I_stim = torch.tensor(-52.0, device=device) if 1.0 <= t < 2.0 else None
            state = model.step(state, dt, I_stim)

        V = model.get_voltage(state)
        print(f"    ✓ 100 steps completed, V = {V.item():.2f} mV")

    print("\n  All model backend tests passed!")


def benchmark_cpu_vs_gpu():
    """Benchmark CPU vs GPU performance."""
    print_header("Test 8: CPU vs GPU Benchmark")

    if not cuda_available():
        print("  CUDA not available - skipping benchmark")
        return

    from ionic import TTP06Model, CellType
    import time

    n_steps = 1000
    dt = 0.01

    results = {}

    for device in ['cpu', 'cuda']:
        print(f"\n  [{device.upper()}]")

        model = TTP06Model(celltype=CellType.EPI, device=device, use_lut=True)
        state = model.get_initial_state()

        # Warmup
        for _ in range(10):
            state = model.step(state, dt, None)

        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        t0 = time.perf_counter()
        for i in range(n_steps):
            t = i * dt
            I_stim = torch.tensor(-52.0, device=device) if 10.0 <= t < 11.0 else None
            state = model.step(state, dt, I_stim)

        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        ms_per_step = (elapsed / n_steps) * 1000
        results[device] = ms_per_step
        print(f"    {n_steps} steps in {elapsed:.3f}s ({ms_per_step:.3f} ms/step)")

    if 'cuda' in results and 'cpu' in results:
        speedup = results['cpu'] / results['cuda']
        print(f"\n  GPU Speedup: {speedup:.2f}x")

    print("\n  Benchmark complete!")


def main():
    print("=" * 60)
    print("  Engine V5.3 Backend Test Suite")
    print("=" * 60)

    print_device_info()

    test_backend_creation()
    test_tensor_creation()
    test_global_backend()
    test_device_info()
    test_memory_management()
    test_lut_on_both_backends()
    test_model_on_both_backends()
    benchmark_cpu_vs_gpu()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    main()
