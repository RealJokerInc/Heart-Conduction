"""Tests for ionic surrogate model components."""

import torch
import pytest


# ---------------------------------------------------------------------------
# Phase 1: ChebyshevReadout tests
# ---------------------------------------------------------------------------

class TestChebyshevReadout:
    """Tests for the Chebyshev polynomial readout layer."""

    def test_chebyshev_shape(self):
        """Output shape and dtype for batched input."""
        from surrogate.model.chebyshev import ChebyshevReadout

        layer = ChebyshevReadout(n_dims=16, degree=3)
        z = torch.randn(32, 16)
        Vm = torch.randn(32)
        out = layer(z, Vm)
        assert out.shape == (32,)
        assert out.dtype == torch.float32

    def test_chebyshev_params(self):
        """66 trainable parameters: C(16,4) + b_vm(1) + b(1)."""
        from surrogate.model.chebyshev import ChebyshevReadout

        layer = ChebyshevReadout(n_dims=16, degree=3)
        n_params = sum(p.numel() for p in layer.parameters())
        assert n_params == 16 * (3 + 1) + 1 + 1  # 66

    def test_chebyshev_zero_init(self):
        """With C=0, b_vm=0, b=0 → I_ion = 0."""
        from surrogate.model.chebyshev import ChebyshevReadout

        layer = ChebyshevReadout(n_dims=16, degree=3)
        z = torch.randn(8, 16)
        Vm = torch.randn(8)
        out = layer(z, Vm)
        assert torch.allclose(out, torch.zeros(8), atol=1e-7)

    def test_chebyshev_linear_recovery(self):
        """Setting C[:,0]=w recovers a linear function: I_ion = Σw_k + b_vm*Vm + b.

        T₀ = 1 everywhere, so C[:,0]*T₀ = C[:,0]. With all other C=0,
        phi_k = C[k,0], so I_ion = sum(C[:,0]) + b_vm*Vm + b.
        """
        from surrogate.model.chebyshev import ChebyshevReadout

        layer = ChebyshevReadout(n_dims=4, degree=3)
        with torch.no_grad():
            layer.C[:, 0] = torch.tensor([1.0, -2.0, 0.5, 3.0])
            layer.b_vm.fill_(0.1)
            layer.b.fill_(-0.5)

        z = torch.randn(10, 4)
        Vm = torch.randn(10)
        out = layer(z, Vm)
        expected = (1.0 - 2.0 + 0.5 + 3.0) + 0.1 * Vm + (-0.5)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_chebyshev_cubic(self):
        """Single dim, C=[0,0,0,1]: output = T₃(z̃).

        T₃(0.5) = 4·(0.5)³ - 3·(0.5) = 4·0.125 - 1.5 = -1.0
        """
        from surrogate.model.chebyshev import ChebyshevReadout

        layer = ChebyshevReadout(n_dims=1, degree=3)
        with torch.no_grad():
            layer.C[0] = torch.tensor([0.0, 0.0, 0.0, 1.0])

        # z̃ = 0.5 with default bounds [-1, 1]: z = 0.5 maps to z̃ = 0.5
        # (since default bounds are already [-1, 1], normalization is identity)
        z = torch.tensor([[0.5]])
        Vm = torch.tensor([0.0])
        out = layer(z, Vm)
        # T₃(0.5) = 4·0.125 - 3·0.5 = 0.5 - 1.5 = -1.0
        assert torch.allclose(out, torch.tensor([-1.0]), atol=1e-5)

    def test_chebyshev_set_bounds(self):
        """set_bounds updates normalization and changes output."""
        from surrogate.model.chebyshev import ChebyshevReadout

        layer = ChebyshevReadout(n_dims=4, degree=3)
        with torch.no_grad():
            layer.C.fill_(1.0)  # Non-zero so normalization matters

        z = torch.ones(2, 4) * 5.0
        Vm = torch.zeros(2)

        out_before = layer(z, Vm).clone()

        # Shift bounds so z=5 maps to interior instead of clamped edge
        layer.set_bounds(
            z_min=torch.tensor([0.0, 0.0, 0.0, 0.0]),
            z_max=torch.tensor([10.0, 10.0, 10.0, 10.0]),
        )
        out_after = layer(z, Vm)
        assert not torch.allclose(out_before, out_after)

    def test_chebyshev_constant_dim(self):
        """z_min == z_max for a dimension → no NaN/Inf (eps + clamp protect)."""
        from surrogate.model.chebyshev import ChebyshevReadout

        layer = ChebyshevReadout(n_dims=4, degree=3)
        with torch.no_grad():
            layer.C.fill_(1.0)

        layer.set_bounds(
            z_min=torch.tensor([0.5, -1.0, -1.0, -1.0]),
            z_max=torch.tensor([0.5, 1.0, 1.0, 1.0]),  # dim 0: constant
        )
        z = torch.randn(8, 4)
        Vm = torch.randn(8)
        out = layer(z, Vm)
        assert torch.isfinite(out).all()

    def test_chebyshev_out_of_bounds(self):
        """Input z outside [z_min, z_max] → clamped, no divergence."""
        from surrogate.model.chebyshev import ChebyshevReadout

        layer = ChebyshevReadout(n_dims=4, degree=3)
        with torch.no_grad():
            layer.C.fill_(1.0)

        layer.set_bounds(
            z_min=torch.tensor([0.0, 0.0, 0.0, 0.0]),
            z_max=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        )
        # Way outside bounds
        z = torch.tensor([[100.0, -100.0, 50.0, -50.0]])
        Vm = torch.tensor([0.0])
        out = layer(z, Vm)
        assert torch.isfinite(out).all()
        # Clamped values should match z at boundary
        z_boundary = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        out_boundary = layer(z_boundary, Vm)
        assert torch.allclose(out, out_boundary, atol=1e-5)


# ---------------------------------------------------------------------------
# Phase 2: IonicSurrogate tests
# ---------------------------------------------------------------------------

class TestIonicSurrogate:
    """Tests for the full 3-stage ionic surrogate model."""

    def test_surrogate_shapes(self):
        """Batched input → correct output shapes and dtypes."""
        from surrogate.model.ionic_surrogate import IonicSurrogate

        model = IonicSurrogate()
        latent = torch.randn(32, 16)
        Vm = torch.randn(32)
        dt = torch.full((32,), 0.01)

        latent_new, I_ion, gates = model(latent, Vm, dt)
        assert latent_new.shape == (32, 16)
        assert I_ion.shape == (32,)
        assert gates.shape == (32, 18)
        assert latent_new.dtype == torch.float32
        assert I_ion.dtype == torch.float32
        assert gates.dtype == torch.float32

    def test_surrogate_single(self):
        """Unbatched (1D) input → 1D outputs."""
        from surrogate.model.ionic_surrogate import IonicSurrogate

        model = IonicSurrogate()
        latent = torch.randn(16)
        Vm = torch.tensor(0.0)
        dt = torch.tensor(0.01)

        latent_new, I_ion, gates = model(latent, Vm, dt)
        assert latent_new.shape == (16,)
        assert I_ion.shape == ()
        assert gates.shape == (18,)

    def test_surrogate_param_count_inference(self):
        """642 inference params (derived from constructor args)."""
        from surrogate.model.ionic_surrogate import IonicSurrogate

        ld, ad, deg, sp = 16, 8, 3, 8
        model = IonicSurrogate(
            latent_dim=ld, attn_dim=ad, cheby_degree=deg, split=sp
        )
        # W_q(ld*ad) + W_k(2*ad) + W_v(2*ad) + W_out(ad*ld)
        # + cc1(sp*ld + ld) + cc2(sp*ld + ld)
        # + C(ld*(deg+1)) + b_vm(1) + b(1)
        expected = (
            ld * ad          # W_q
            + 2 * ad         # W_k
            + 2 * ad         # W_v
            + ad * ld        # W_out
            + sp * ld + ld   # cc1 weight + bias
            + sp * ld + ld   # cc2 weight + bias
            + ld * (deg + 1) # C
            + 1 + 1          # b_vm, b
        )
        assert expected == 642
        assert model.inference_param_count() == expected

    def test_surrogate_param_count_training(self):
        """948 training params = 642 inference + 306 scaffold."""
        from surrogate.model.ionic_surrogate import IonicSurrogate

        ld, n_g = 16, 18
        model = IonicSurrogate(latent_dim=ld, n_gates=n_g)
        scaffold_params = ld * n_g + n_g  # weight + bias = 306
        total = sum(p.numel() for p in model.parameters())
        assert total == model.inference_param_count() + scaffold_params
        assert total == 948

    def test_surrogate_stage1_contractivity(self):
        """Stage 1 contracts: ||latent_mid - target|| < ||latent_prev - target||.

        For sigmoid gate ∈ (0,1), the interpolation
        mid = prev + gate*(target - prev) always moves closer to target.
        """
        from surrogate.model.ionic_surrogate import IonicSurrogate

        model = IonicSurrogate()
        model.eval()

        torch.manual_seed(42)
        B = 100
        latent_prev = torch.randn(B, 16)
        Vm = torch.randn(B)
        dt = torch.full((B,), 0.01)

        with torch.no_grad():
            x = torch.stack([Vm, dt], dim=-1)
            k = model.W_k(x)
            v = model.W_v(x)
            q = latent_prev.unsqueeze(-1) * model.W_q
            score = (q * k.unsqueeze(1)).sum(-1) * model.scale
            gate = torch.sigmoid(score)
            target = v @ model.W_out
            latent_mid = latent_prev + gate * (target - latent_prev)

        dist_before = (latent_prev - target).norm(dim=-1)
        dist_after = (latent_mid - target).norm(dim=-1)
        # gate ∈ (0,1) guarantees contraction (strict for gate > 0)
        assert (dist_after < dist_before).all()

    def test_surrogate_spectral_norm(self):
        """cc1 and cc2 have spectral norm ≈ 1.0 (spectral_norm wrapper active)."""
        from surrogate.model.ionic_surrogate import IonicSurrogate

        model = IonicSurrogate()
        # Run many forward passes so power iteration converges
        for _ in range(100):
            latent = torch.randn(4, 16)
            Vm = torch.randn(4)
            dt = torch.full((4,), 0.01)
            model(latent, Vm, dt)

        for name, module in [("cc1", model.cc1), ("cc2", model.cc2)]:
            # Verify spectral_norm hooks are registered
            assert hasattr(module, "weight_orig"), f"{name} missing weight_orig"
            sigma = torch.linalg.matrix_norm(module.weight, ord=2)
            assert sigma <= 1.0 + 0.01, f"{name} spectral norm {sigma} > 1.01"

    def test_surrogate_no_scaffold(self):
        """scaffold=False → gates is None, param count = 642."""
        from surrogate.model.ionic_surrogate import IonicSurrogate

        model = IonicSurrogate(scaffold=False)
        latent = torch.randn(4, 16)
        Vm = torch.randn(4)
        dt = torch.full((4,), 0.01)

        _, _, gates = model(latent, Vm, dt)
        assert gates is None
        assert model.inference_param_count() == 642
        total = sum(p.numel() for p in model.parameters())
        assert total == 642

    def test_surrogate_remove_scaffold(self):
        """remove_scaffold() drops decoder. Second call is idempotent."""
        from surrogate.model.ionic_surrogate import IonicSurrogate

        model = IonicSurrogate(scaffold=True)
        assert hasattr(model, "decoder")
        total_before = sum(p.numel() for p in model.parameters())
        assert total_before == 948

        model.remove_scaffold()
        assert not hasattr(model, "decoder")
        total_after = sum(p.numel() for p in model.parameters())
        assert total_after == 642

        # Idempotent — no error on second call
        model.remove_scaffold()
        assert not hasattr(model, "decoder")

    def test_surrogate_gradient_flow(self):
        """10-step autoregressive rollout: loss.backward() completes, no NaN."""
        from surrogate.model.ionic_surrogate import IonicSurrogate

        model = IonicSurrogate()
        latent = torch.zeros(8, 16)
        Vm = torch.zeros(8)
        dt = torch.full((8,), 0.01)

        I_ions = []
        gates_list = []
        for _ in range(10):
            latent, I_ion, gates = model(latent, Vm, dt)
            I_ions.append(I_ion)
            gates_list.append(gates)
            # Simple Vm update (not physically accurate, just for gradient test)
            Vm = Vm + dt * (-I_ion)

        # Include both I_ion and gates in loss so all params get gradients
        loss = (
            torch.stack(I_ions).pow(2).mean()
            + torch.stack(gates_list).pow(2).mean()
        )
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad).all(), f"NaN/Inf grad in {name}"

    def test_surrogate_no_import_cascade(self):
        """Model package doesn't break existing data imports."""
        from surrogate.data.storage import TraceStorage  # noqa: F401
        from surrogate.model import IonicSurrogate  # noqa: F401
