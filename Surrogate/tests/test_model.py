"""Tests for ionic surrogate v3 model components."""

import torch
import pytest


# ---------------------------------------------------------------------------
# v3 Phase 1: NernstComputer tests
# ---------------------------------------------------------------------------

class TestNernst:
    """Tests for the Nernst reversal potential module."""

    def test_nernst_values(self):
        """Known concentrations produce correct reversal potentials (match TTP06)."""
        from surrogate.model.nernst import NernstComputer, RTONF

        nernst = NernstComputer()

        # Typical TTP06 resting concentrations
        Na_i = torch.tensor([11.6])   # mM
        K_i = torch.tensor([138.3])   # mM
        Ca_i = torch.tensor([0.00008])  # mM

        E_Na, E_K, E_Ca, E_Ks = nernst(Na_i, K_i, Ca_i)

        # Reference values computed from TTP06 currents.py formulas:
        #   E_Na = RTONF * ln(140 / 11.6)
        #   E_K  = RTONF * ln(5.4 / 138.3)
        #   E_Ca = 0.5 * RTONF * ln(2.0 / 0.00008)
        #   E_Ks = RTONF * ln((5.4 + 0.03*140) / (138.3 + 0.03*11.6))
        import math
        expected_E_Na = RTONF * math.log(140.0 / 11.6)
        expected_E_K = RTONF * math.log(5.4 / 138.3)
        expected_E_Ca = 0.5 * RTONF * math.log(2.0 / 0.00008)
        expected_E_Ks = RTONF * math.log((5.4 + 0.03 * 140.0) / (138.3 + 0.03 * 11.6))

        assert torch.allclose(E_Na, torch.tensor([expected_E_Na], dtype=torch.float32), atol=1e-3)
        assert torch.allclose(E_K, torch.tensor([expected_E_K], dtype=torch.float32), atol=1e-3)
        assert torch.allclose(E_Ca, torch.tensor([expected_E_Ca], dtype=torch.float32), atol=1e-3)
        assert torch.allclose(E_Ks, torch.tensor([expected_E_Ks], dtype=torch.float32), atol=1e-3)

    def test_nernst_differentiable(self):
        """Backward through log produces no NaN gradients."""
        from surrogate.model.nernst import NernstComputer

        nernst = NernstComputer()

        Na_i = torch.tensor([11.6], requires_grad=True)
        K_i = torch.tensor([138.3], requires_grad=True)
        Ca_i = torch.tensor([0.00008], requires_grad=True)

        E_Na, E_K, E_Ca, E_Ks = nernst(Na_i, K_i, Ca_i)
        loss = E_Na.sum() + E_K.sum() + E_Ca.sum() + E_Ks.sum()
        loss.backward()

        assert Na_i.grad is not None and torch.isfinite(Na_i.grad).all()
        assert K_i.grad is not None and torch.isfinite(K_i.grad).all()
        assert Ca_i.grad is not None and torch.isfinite(Ca_i.grad).all()

        # Also test near-zero Ca_i (the risky case)
        Ca_i_tiny = torch.tensor([1e-15], requires_grad=True)
        _, _, E_Ca_tiny, _ = nernst(
            torch.tensor([11.6]), torch.tensor([138.3]), Ca_i_tiny
        )
        E_Ca_tiny.sum().backward()
        assert Ca_i_tiny.grad is not None and torch.isfinite(Ca_i_tiny.grad).all()

    def test_nernst_normalization_range(self):
        """Physiological inputs produce normalized output in [-2, 2]."""
        from surrogate.model.nernst import NernstComputer

        nernst = NernstComputer()

        # Batch of physiological concentrations
        B = 50
        torch.manual_seed(123)
        Na_i = torch.rand(B) * 16 + 4       # [4, 20] mM
        K_i = torch.rand(B) * 15 + 130      # [130, 145] mM
        Ca_i = torch.rand(B) * 0.002 + 5e-5  # [5e-5, 2.05e-3] mM
        Ca_ss = torch.rand(B) * 0.002 + 5e-5
        Vm = torch.rand(B) * 130 - 90       # [-90, 40] mV

        E_Na, E_K, E_Ca, E_Ks = nernst(Na_i, K_i, Ca_i)
        env_norm = nernst.normalize_environment(
            Vm, E_Na, E_K, E_Ca, E_Ks, Na_i, K_i, Ca_i, Ca_ss
        )

        assert env_norm.shape == (B, 9)
        assert torch.isfinite(env_norm).all()
        # Within [-2, 2] for physiological inputs
        assert env_norm.abs().max() <= 2.0, (
            f"Normalized env outside [-2, 2]: max abs = {env_norm.abs().max():.3f}"
        )


# ---------------------------------------------------------------------------
# v3 Phase 1: IonicStage1 tests (attention + MLP + compression)
# ---------------------------------------------------------------------------

class TestStage1:
    """Tests for Stage 1: attention + MLP + compression."""

    def test_stage1_shapes(self):
        """Batched (32, 20) input produces correct output shapes."""
        from surrogate.model.stage1 import IonicStage1

        model = IonicStage1()
        B = 32
        carried = torch.randn(B, 20)
        Vm = torch.randn(B)

        cs_out, cond_lat, conc_new, gf, gc = model(carried, Vm)
        assert cs_out.shape == (B, 20)
        assert cond_lat.shape == (B, 8)
        assert conc_new.shape == (B, 4)
        assert gf.shape == (B, 14)
        assert gc.shape == (B, 5)
        # forward() returns carried_state unchanged (no dynamics)
        assert torch.allclose(cs_out, carried)

    def test_stage1_unbatched(self):
        """Unbatched (20,) input produces 1D output shapes."""
        from surrogate.model.stage1 import IonicStage1

        model = IonicStage1()
        carried = torch.randn(20)
        Vm = torch.tensor(0.0)

        cs_out, cond_lat, conc_new, gf, gc = model(carried, Vm)
        assert cs_out.shape == (20,)
        assert cond_lat.shape == (8,)
        assert conc_new.shape == (4,)
        assert gf.shape == (14,)
        assert gc.shape == (5,)

    def test_stage1_contractivity(self):
        """Attention contracts: ||z_mid - target|| < ||carried - target||.

        For sigmoid gate in (0,1), the update
        mid = prev + gate*(target - prev) always moves closer to target.
        """
        from surrogate.model.stage1 import IonicStage1

        model = IonicStage1()
        model.eval()

        torch.manual_seed(42)
        B = 100
        carried = torch.randn(B, 20)
        Vm = torch.randn(B)

        with torch.no_grad():
            z_mid = model.voltage_attention(carried, Vm)
            x = Vm.unsqueeze(-1)
            v = model.voltage_attention.W_v(x)
            target = v @ model.voltage_attention.W_out

        dist_before = (carried - target).norm(dim=-1)
        dist_after = (z_mid - target).norm(dim=-1)
        # gate in (0,1) guarantees contraction (strict for gate > 0)
        assert (dist_after < dist_before).all()

    def test_stage1_alpha_zero(self):
        """ionic_mixing_logit=-100 (sigmoid~0): dzdt ionic rate ~ attention delta."""
        from surrogate.model.stage1 import IonicStage1

        model = IonicStage1()
        with torch.no_grad():
            model.ionic_mixing_logit.fill_(-100.0)

        torch.manual_seed(7)
        carried = torch.randn(8, 20)
        Vm = torch.randn(8)

        with torch.no_grad():
            dz = model.dzdt(carried, Vm)

        # Recompute attention delta to compare
        with torch.no_grad():
            z_mid = model.voltage_attention(carried, Vm)
            delta = z_mid - carried
            ionic_delta = delta[:, :16]

        # With alpha~0, ionic rate should be nearly identical to attention delta
        ionic_rate = dz[:, :16]
        assert torch.allclose(ionic_rate, ionic_delta, atol=1e-5), (
            f"Max diff: {(ionic_rate - ionic_delta).abs().max():.2e}"
        )

    def test_stage1_beta_zero(self):
        """gate_conductance_logit=-100 (sigmoid~0) makes cond_lat ~ linear path."""
        from surrogate.model.stage1 import IonicStage1

        model = IonicStage1()
        with torch.no_grad():
            model.gate_conductance_logit.fill_(-100.0)

        torch.manual_seed(7)
        carried = torch.randn(8, 20)
        Vm = torch.randn(8)

        with torch.no_grad():
            _, cond_lat, _, _, _ = model(carried, Vm)

        # Compare against linear path of carried_state (forward doesn't advance state)
        with torch.no_grad():
            expected = model.gate_conductance_linear(carried)

        assert torch.allclose(cond_lat, expected, atol=1e-5), (
            f"Max diff: {(cond_lat - expected).abs().max():.2e}"
        )

    def test_stage1_conc_no_mlp(self):
        """Concentration rate is unchanged by MLP modifications.

        MLP only applies to ionic dims. Concentration dims use attention delta
        only. Changing MLP weights should not affect conc rate in dzdt.
        """
        from surrogate.model.stage1 import IonicStage1

        model = IonicStage1()
        torch.manual_seed(42)
        carried = torch.randn(8, 20)
        Vm = torch.randn(8)

        with torch.no_grad():
            dz1 = model.dzdt(carried, Vm)
            conc_rate1 = dz1[:, 16:]

        # Drastically change MLP weights
        with torch.no_grad():
            model.ionic_mixing_mlp[0].weight.fill_(99.0)
            model.ionic_mixing_mlp[0].bias.fill_(99.0)
            model.ionic_mixing_mlp[2].weight.fill_(99.0)
            model.ionic_mixing_mlp[2].bias.fill_(99.0)

        with torch.no_grad():
            dz2 = model.dzdt(carried, Vm)
            conc_rate2 = dz2[:, 16:]

        assert torch.allclose(conc_rate1, conc_rate2, atol=1e-7), (
            f"Concentration rate changed after MLP modification: max diff = "
            f"{(conc_rate1 - conc_rate2).abs().max():.2e}"
        )

    def test_stage1_param_count(self):
        """Parameter count matches expected for default small config."""
        from surrogate.model.stage1 import IonicStage1

        model = IonicStage1()

        # Inference params (no scaffolds)
        # W_q: 20*4=80, W_k: 1*4=4, W_v: 1*4=4, W_out: 4*20=80
        # ionic_mixing_mlp[0]: 16*16+16=272, ionic_mixing_mlp[2]: 16*16+16=272
        # ionic_mixing_logit: 16
        # gate_conductance_linear: 20*8=160 (no bias)
        # gate_conductance_mlp[0]: 20*12+12=252, [2]: 12*12+12=156, [4]: 12*8+8=104
        # gate_conductance_logit: 8
        expected_inference = (
            20 * 4         # W_q
            + 1 * 4        # W_k (dt removed: was 2*4=8)
            + 1 * 4        # W_v (dt removed: was 2*4=8)
            + 4 * 20       # W_out
            + 16 * 16 + 16 # ionic_mixing_mlp[0] weight + bias
            + 16 * 16 + 16 # ionic_mixing_mlp[2] weight + bias
            + 16           # ionic_mixing_logit
            + 20 * 8       # gate_conductance_linear (no bias) -- input = carried_dim=20
            + 20 * 12 + 12 # gate_conductance_mlp[0] weight + bias
            + 12 * 12 + 12 # gate_conductance_mlp[2] weight + bias
            + 12 * 8 + 8   # gate_conductance_mlp[4] weight + bias
            + 8            # gate_conductance_logit
        )
        assert expected_inference == 1408
        assert model.inference_param_count() == expected_inference

        # Scaffold params
        # ionic_state_decoder: 16*14+14=238, gate_conductance_decoder: 8*5+5=45
        expected_scaffold = (16 * 14 + 14) + (8 * 5 + 5)
        assert expected_scaffold == 283

        total = sum(p.numel() for p in model.parameters())
        assert total == expected_inference + expected_scaffold
        assert total == 1691

    def test_stage1_remove_scaffold(self):
        """remove_scaffold() drops decoders. Second call is idempotent."""
        from surrogate.model.stage1 import IonicStage1

        model = IonicStage1(scaffold=True)
        assert hasattr(model, "ionic_state_decoder")
        assert hasattr(model, "gate_conductance_decoder")
        total_before = sum(p.numel() for p in model.parameters())
        assert total_before == 1691

        model.remove_scaffold()
        assert not hasattr(model, "ionic_state_decoder")
        assert not hasattr(model, "gate_conductance_decoder")
        total_after = sum(p.numel() for p in model.parameters())
        assert total_after == 1408

        # Idempotent -- no error on second call
        model.remove_scaffold()
        assert not hasattr(model, "ionic_state_decoder")
        assert not hasattr(model, "gate_conductance_decoder")

        # Forward still works after scaffold removal
        carried = torch.randn(4, 20)
        Vm = torch.randn(4)
        cs_out, cond_lat, conc_new, gf, gc = model(carried, Vm)
        assert gf is None
        assert gc is None

    def test_stage1_gradient_flow(self):
        """dzdt + compression: loss.backward() completes with no NaN gradients."""
        from surrogate.model.stage1 import IonicStage1

        model = IonicStage1()
        torch.manual_seed(99)
        carried = torch.randn(8, 20)  # non-zero to avoid rms_norm div-by-zero
        Vm = torch.randn(8)

        # Use dzdt for dynamics + forward for scaffold
        dz = model.dzdt(carried, Vm)
        carried_new = carried + 0.01 * dz  # Euler step
        _, cond_lat, _, gf, gc = model(carried_new, Vm)

        loss = dz.pow(2).mean() + gf.pow(2).mean() + gc.pow(2).mean() + cond_lat.pow(2).mean()
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad).all(), f"NaN/Inf grad in {name}"

    def test_stage1_dzdt_shape(self):
        """dzdt(z, V) returns same shape as z, float64 when input is float64."""
        from surrogate.model.stage1 import IonicStage1

        model = IonicStage1().double()
        z = torch.randn(8, 20, dtype=torch.float64)
        Vm = torch.randn(8, dtype=torch.float64)

        dz = model.dzdt(z, Vm)
        assert dz.shape == z.shape
        assert dz.dtype == torch.float64
        assert torch.isfinite(dz).all()

        # Unbatched
        dz_single = model.dzdt(z[0], Vm[0])
        assert dz_single.shape == (20,)

    def test_stage1_dzdt_numerical(self):
        """At alpha~0 (init), dzdt ≈ voltage_attention(z, V) - z."""
        from surrogate.model.stage1 import IonicStage1

        model = IonicStage1()
        # Alpha is already near zero at init (ALPHA_INIT=-5.0, sigmoid(-5)≈0.007)
        torch.manual_seed(42)
        z = torch.randn(16, 20)
        Vm = torch.randn(16)

        with torch.no_grad():
            dz = model.dzdt(z, Vm)
            z_mid = model.voltage_attention(z, Vm)
            expected_delta = z_mid - z

        # Ionic dims: dzdt ≈ delta + alpha*correction ≈ delta (alpha≈0.007)
        assert torch.allclose(dz[:, :16], expected_delta[:, :16], atol=0.05), (
            f"Max diff: {(dz[:, :16] - expected_delta[:, :16]).abs().max():.4f}"
        )
        # Conc dims: dzdt == delta exactly (no MLP)
        assert torch.allclose(dz[:, 16:], expected_delta[:, 16:], atol=1e-7)

    def test_stage1_forward_no_dynamics(self):
        """forward() returns carried_state unchanged (compression + scaffold only)."""
        from surrogate.model.stage1 import IonicStage1

        model = IonicStage1()
        torch.manual_seed(0)
        carried = torch.randn(8, 20)
        Vm = torch.randn(8)

        with torch.no_grad():
            cs_out, cond_lat, conc, gf, gc = model(carried, Vm)

        # carried_state returned unchanged
        assert torch.allclose(cs_out, carried)
        # conc is just a slice of carried
        assert torch.allclose(conc, carried[:, 16:])
        # cond_lat, gf, gc are derived but carried is not modified
        assert cond_lat.shape == (8, 8)
        assert gf.shape == (8, 14)
        assert gc.shape == (8, 5)

    def test_residual_bypass(self):
        """residual_bypass at extreme logits."""
        from surrogate.model.stage1 import residual_bypass

        base = torch.randn(8, 16)
        correction = torch.randn(8, 16)

        # alpha ≈ 0: output ≈ base
        out_zero = residual_bypass(base, correction, torch.tensor([-100.0]))
        assert torch.allclose(out_zero, base, atol=1e-4)

        # alpha ≈ 1: output ≈ base + correction
        out_one = residual_bypass(base, correction, torch.tensor([100.0]))
        assert torch.allclose(out_one, base + correction, atol=1e-4)

    def test_stage1_compress(self):
        """_compress() produces correct shape and matches forward() output."""
        from surrogate.model.stage1 import IonicStage1

        model = IonicStage1()
        carried = torch.randn(8, 20)
        Vm = torch.randn(8)

        with torch.no_grad():
            cond_direct = model._compress(carried)
            _, cond_fwd, _, _, _ = model(carried, Vm)

        assert cond_direct.shape == (8, 8)
        assert torch.allclose(cond_direct, cond_fwd)


# ---------------------------------------------------------------------------
# v3 Phase 1: IonicStage2 tests (cross-attention readout)
# ---------------------------------------------------------------------------

class TestStage2:
    """Tests for the v3 Stage 2 cross-attention current readout."""

    def test_stage2_shapes(self):
        """Batched (32, 8) + (32, 9) -> (32,) scalar I_ion."""
        from surrogate.model.stage2 import IonicStage2

        stage2 = IonicStage2(cond_dim=8, n_env=9, attn_dim=4, d_v=1, mlp_hidden=4)
        cond_lat = torch.randn(32, 8)
        env_norm = torch.randn(32, 9)
        I_ion = stage2(cond_lat, env_norm)
        assert I_ion.shape == (32,)
        assert I_ion.dtype == torch.float32

    def test_stage2_unbatched(self):
        """Unbatched (8,) + (9,) -> scalar."""
        from surrogate.model.stage2 import IonicStage2

        stage2 = IonicStage2(cond_dim=8, n_env=9, attn_dim=4, d_v=1, mlp_hidden=4)
        cond_lat = torch.randn(8)
        env_norm = torch.randn(9)
        I_ion = stage2(cond_lat, env_norm)
        assert I_ion.shape == ()
        assert I_ion.dtype == torch.float32

    def test_stage2_zero_cond(self):
        """Zero conductance -> I_ion approx zero (MLP bias is zero-init).

        Zero conductance -> Q=0 -> scores=0 -> attended=0.
        MLP(0) = W2 @ GELU(W1 @ 0 + b1) + b2 = W2 @ GELU(0) + 0 = 0
        since b1=0, b2=0 (zero-init), GELU(0)=0.
        """
        from surrogate.model.stage2 import IonicStage2

        stage2 = IonicStage2(cond_dim=8, n_env=9, attn_dim=4, d_v=1, mlp_hidden=4)
        cond_lat = torch.zeros(16, 8)
        env_norm = torch.randn(16, 9)
        I_ion = stage2(cond_lat, env_norm)
        assert torch.allclose(I_ion, torch.zeros_like(I_ion), atol=1e-7)

    def test_stage2_param_count(self):
        """e_q(8x4) + e_k(9x4) + e_v(9x1) + W1(8x4+4) + W2(4x1+1) = 118."""
        from surrogate.model.stage2 import IonicStage2

        stage2 = IonicStage2(cond_dim=8, n_env=9, attn_dim=4, d_v=1, mlp_hidden=4)
        expected = (
            8 * 4       # e_q
            + 9 * 4     # e_k
            + 9 * 1     # e_v
            + 8 * 4 + 4 # mlp_w1 weight + bias
            + 4 * 1 + 1 # mlp_w2 weight + bias
        )
        assert expected == 118
        total = sum(p.numel() for p in stage2.parameters())
        assert total == expected

    def test_stage2_gradient_flow(self):
        """Backward through cross-attention + MLP: no NaN gradients."""
        from surrogate.model.stage2 import IonicStage2

        stage2 = IonicStage2(cond_dim=8, n_env=9, attn_dim=4, d_v=1, mlp_hidden=4)
        cond_lat = torch.randn(16, 8, requires_grad=True)
        env_norm = torch.randn(16, 9, requires_grad=True)
        I_ion = stage2(cond_lat, env_norm)
        loss = I_ion.pow(2).mean()
        loss.backward()

        # All model parameters have gradients
        for name, p in stage2.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad).all(), f"NaN/Inf grad in {name}"

        # Input tensors also have gradients
        assert cond_lat.grad is not None
        assert torch.isfinite(cond_lat.grad).all()
        assert env_norm.grad is not None
        assert torch.isfinite(env_norm.grad).all()

    def test_stage2_negative_scores(self):
        """Attention scores can be negative (no softmax)."""
        from surrogate.model.stage2 import IonicStage2

        stage2 = IonicStage2(cond_dim=8, n_env=9, attn_dim=4, d_v=1, mlp_hidden=4)
        # Use large random inputs to ensure some scores are negative
        torch.manual_seed(0)
        cond_lat = torch.randn(64, 8) * 5.0
        env_norm = torch.randn(64, 9) * 5.0

        with torch.no_grad():
            attn = stage2.conductance_attention
            Q = torch.einsum('ij,jk->ijk', cond_lat, attn.e_q)
            K = torch.einsum('il,lk->ilk', env_norm, attn.e_k)
            scores = torch.einsum('ijk,ilk->ijl', Q, K) * attn.scale
        assert (scores < 0).any(), "Expected some negative attention scores"


# ---------------------------------------------------------------------------
# v3 Phase 1: IonicSurrogateV3 orchestrator tests
# ---------------------------------------------------------------------------

class TestV3:
    """Tests for the v3 orchestrator combining Stage 1 + Nernst + Stage 2."""

    def _make_inputs(self, B=32, seed=42):
        """Helper: create standard batched inputs for V3 forward."""
        torch.manual_seed(seed)
        carried = torch.randn(B, 20)          # carried_dim = 16 + 4
        Vm = torch.randn(B)
        dt = torch.full((B,), 0.01)
        cond_lat_prev = torch.randn(B, 8)     # cond_dim
        # Physiological concentrations: [Na_i, K_i, Ca_i, Ca_ss]
        conc_prev = torch.stack([
            torch.rand(B) * 16 + 4,            # Na_i: [4, 20]
            torch.rand(B) * 15 + 130,           # K_i: [130, 145]
            torch.rand(B) * 0.002 + 5e-5,       # Ca_i: [5e-5, 2.05e-3]
            torch.rand(B) * 0.002 + 5e-5,       # Ca_ss: [5e-5, 2.05e-3]
        ], dim=-1)
        return carried, Vm, dt, cond_lat_prev, conc_prev

    def test_v3_shapes(self):
        """Full forward pass produces correct output shapes."""
        from surrogate.model.ionic_surrogate_v3 import IonicSurrogateV3

        model = IonicSurrogateV3()
        B = 32
        carried, Vm, dt, cond_lat_prev, conc_prev = self._make_inputs(B)

        out = model(carried, Vm, dt, cond_lat_prev, conc_prev)

        assert out["carried_state"].shape == (B, 20)
        assert out["conductance_latent"].shape == (B, 8)
        assert out["concentrations"].shape == (B, 4)
        assert out["I_ion"].shape == (B,)
        assert out["ionic_state_pred"].shape == (B, 14)
        assert out["conductance_pred"].shape == (B, 5)
        # All float32
        for key in ["carried_state", "conductance_latent", "concentrations", "I_ion"]:
            assert out[key].dtype == torch.float32, f"{key} dtype: {out[key].dtype}"

    def test_v3_autoregressive(self):
        """5-step Euler rollout via dzdt: outputs feed back, backward no NaN."""
        from surrogate.model.ionic_surrogate_v3 import IonicSurrogateV3

        model = IonicSurrogateV3()
        B = 8
        carried, Vm, dt, cond_lat_prev, conc_prev = self._make_inputs(B, seed=0)

        I_ions = []
        ionic_state_list = []
        conductance_list = []
        cond_list = []
        for step in range(5):
            # Advance state via dzdt (exercises attention + MLP + mixing logit)
            dz = model.stage1.dzdt(carried, Vm)
            carried = carried + 0.01 * dz  # Euler step

            # Compression + scaffold + readout via V3 forward
            out = model(carried, Vm, dt, cond_lat_prev, conc_prev)
            cond_lat_prev = out["conductance_latent"]
            conc_prev = out["concentrations"]
            I_ions.append(out["I_ion"])
            ionic_state_list.append(out["ionic_state_pred"])
            conductance_list.append(out["conductance_pred"])
            cond_list.append(out["conductance_latent"])
            # Simple Vm update for gradient flow
            Vm = Vm + dt * (-out["I_ion"])

        # Include all outputs so all params get gradients
        loss = (
            torch.stack(I_ions).pow(2).mean()
            + torch.stack(ionic_state_list).pow(2).mean()
            + torch.stack(conductance_list).pow(2).mean()
            + torch.stack(cond_list).pow(2).mean()
        )
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad).all(), f"NaN/Inf grad in {name}"

    def test_v3_stage2_reads_old(self):
        """I_ion depends on cond_lat_prev, NOT on the current step's conductance.

        Changing cond_lat_prev changes I_ion; changing carried_state (which
        changes the NEW conductance) should NOT change I_ion for that step.
        """
        from surrogate.model.ionic_surrogate_v3 import IonicSurrogateV3

        model = IonicSurrogateV3()
        model.eval()

        carried, Vm, dt, cond_lat_prev, conc_prev = self._make_inputs(B=8, seed=99)

        with torch.no_grad():
            out1 = model(carried, Vm, dt, cond_lat_prev, conc_prev)

        # Change cond_lat_prev -> I_ion should change
        cond_lat_prev2 = cond_lat_prev + 10.0
        with torch.no_grad():
            out2 = model(carried, Vm, dt, cond_lat_prev2, conc_prev)
        assert not torch.allclose(out1["I_ion"], out2["I_ion"]), (
            "I_ion should change when cond_lat_prev changes"
        )

        # Change carried_state (affects NEW cond, not prev) -> I_ion should NOT change
        carried2 = carried + 10.0
        with torch.no_grad():
            out3 = model(carried2, Vm, dt, cond_lat_prev, conc_prev)
        assert torch.allclose(out1["I_ion"], out3["I_ion"]), (
            "I_ion should NOT change when only carried_state changes "
            f"(max diff: {(out1['I_ion'] - out3['I_ion']).abs().max():.2e})"
        )

    def test_v3_nernst_uses_prev(self):
        """Nernst uses prev concentrations, not new.

        Change prev conc -> I_ion changes. Change new conc (via carried_state
        conc dims) -> I_ion unchanged.
        """
        from surrogate.model.ionic_surrogate_v3 import IonicSurrogateV3

        model = IonicSurrogateV3()
        model.eval()

        carried, Vm, dt, cond_lat_prev, conc_prev = self._make_inputs(B=8, seed=77)

        with torch.no_grad():
            out1 = model(carried, Vm, dt, cond_lat_prev, conc_prev)

        # Change conc_prev -> I_ion should change (Nernst changes -> env changes)
        conc_prev2 = conc_prev.clone()
        conc_prev2[:, 0] = conc_prev2[:, 0] * 2.0  # double Na_i
        with torch.no_grad():
            out2 = model(carried, Vm, dt, cond_lat_prev, conc_prev2)
        assert not torch.allclose(out1["I_ion"], out2["I_ion"]), (
            "I_ion should change when conc_prev changes"
        )

        # Change carried_state conc dims (affects NEW conc, not prev) -> I_ion unchanged
        carried2 = carried.clone()
        carried2[:, 16:] = carried2[:, 16:] + 100.0  # modify conc dims in carried
        with torch.no_grad():
            out3 = model(carried2, Vm, dt, cond_lat_prev, conc_prev)
        assert torch.allclose(out1["I_ion"], out3["I_ion"]), (
            "I_ion should NOT change when only carried conc dims change "
            f"(max diff: {(out1['I_ion'] - out3['I_ion']).abs().max():.2e})"
        )

    def test_v3_remove_scaffold(self):
        """remove_scaffold() delegates to stage1. Idempotent."""
        from surrogate.model.ionic_surrogate_v3 import IonicSurrogateV3

        model = IonicSurrogateV3(scaffold=True)
        assert hasattr(model.stage1, "ionic_state_decoder")
        assert hasattr(model.stage1, "gate_conductance_decoder")

        model.remove_scaffold()
        assert not hasattr(model.stage1, "ionic_state_decoder")
        assert not hasattr(model.stage1, "gate_conductance_decoder")

        # Idempotent -- no error on second call
        model.remove_scaffold()
        assert not hasattr(model.stage1, "ionic_state_decoder")

        # Forward still works after scaffold removal
        carried, Vm, dt, cond_lat_prev, conc_prev = self._make_inputs(B=4, seed=0)
        with torch.no_grad():
            out = model(carried, Vm, dt, cond_lat_prev, conc_prev)
        assert out["ionic_state_pred"] is None
        assert out["conductance_pred"] is None
        assert out["I_ion"].shape == (4,)

    def test_v3_param_count(self):
        """inference_param_count = stage1 + stage2 (nernst has 0 learned params)."""
        from surrogate.model.ionic_surrogate_v3 import IonicSurrogateV3

        model = IonicSurrogateV3()

        # Nernst has zero learned parameters (only buffers)
        nernst_params = sum(p.numel() for p in model.nernst.parameters())
        assert nernst_params == 0, f"Nernst has {nernst_params} params, expected 0"

        # inference_param_count = stage1.inference + stage2 total
        expected = model.stage1.inference_param_count() + sum(
            p.numel() for p in model.stage2.parameters()
        )
        assert model.inference_param_count() == expected

        # Verify known values: stage1=1408 (dt removed: W_k/W_v 1*4 not 2*4), stage2=118
        assert model.stage1.inference_param_count() == 1408
        stage2_params = sum(p.numel() for p in model.stage2.parameters())
        assert stage2_params == 118
        assert model.inference_param_count() == 1408 + 118  # = 1526

    def test_v3_no_import_cascade(self):
        """v3 model import doesn't break existing data imports."""
        from surrogate.data.storage import TraceStorage  # noqa: F401
        from surrogate.model import IonicSurrogateV3  # noqa: F401
        from surrogate.model.ionic_surrogate_v3 import IonicSurrogateV3 as V3  # noqa: F401
