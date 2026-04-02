#!/usr/bin/env python3
"""Generate Ionic Surrogate v3 architecture diagram using schemdraw.

Publication-quality box-and-arrow flowchart. Two columns:
  Left: Stage 1 (state evolution, off critical path)
  Right: Stage 2 (current readout, ON critical path)

Usage: python generate_v3_schemdraw.py
"""

import schemdraw
from schemdraw import flow

# === TUNABLE PARAMETERS ===
FONTSIZE = 11
BOX_W = 4.0          # standard box width
BOX_H = 0.9          # standard box height
TALL_H = 1.6         # tall box (MLP stack)
ARROW_GAP = 0.5      # gap between boxes
FORK_OFFSET = 2.5    # horizontal offset for fork branches
COL_GAP = 10.0       # gap between left and right columns
DPI = 250

# Colors
C_INPUT = ('mediumpurple', 'lavender')
C_ATTN = ('steelblue', 'lightblue')
C_MLP = ('darkorange', 'moccasin')
C_COMP = ('seagreen', 'honeydew')
C_NERNST = ('purple', 'thistle')
C_READOUT = ('crimson', 'mistyrose')
C_SCAFFOLD = ('gray', 'whitesmoke')


def make_box(label, w=BOX_W, h=BOX_H, color='black', fill=None):
    b = flow.Box(w=w, h=h).label(label).color(color)
    if fill:
        b = b.fill(fill)
    return b


def main():
    with schemdraw.Drawing() as d:
        d.config(fontsize=FONTSIZE)

        # ================================================================
        # STAGE 1 (LEFT COLUMN)
        # ================================================================
        d += make_box('Stage 1: State Evolution\n(off critical path · 1,416 params)',
                      w=6, h=0.8, color=C_ATTN[0], fill='aliceblue')

        # --- Inputs ---
        d += flow.Arrow().down(ARROW_GAP)
        d += (cs := make_box('carried_state(t)\n[ionic(16) | conc(4)] = 20ᵀ',
                             w=5, color=C_INPUT[0], fill=C_INPUT[1]).anchor('N'))

        # --- Attention ---
        d += flow.Arrow().down(ARROW_GAP)
        d += (attn := make_box('n×1 Cross-Attention\n20 dims attend to [Vm, dt], d=4\nσ(Q·Kᵀ/√4) gate · contractive',
                               w=5.5, h=1.3, color=C_ATTN[0], fill=C_ATTN[1]).anchor('N'))

        # Vm/dt injection label
        d += flow.Box(w=2, h=0.5).at(attn.W, dx=-1.5).anchor('E').label('Vm, dt').color(C_INPUT[0]).fill(C_INPUT[1])
        d += flow.Arrow().at(attn.W, dx=-0.3).to(attn.W)

        # --- Split ---
        d += flow.Arrow().at(attn.S).down(ARROW_GAP)
        d += (split_bar := flow.Box(w=5.5, h=0.4).anchor('N').label('SPLIT after attention').color('gray').fill('lightyellow'))

        # Left fork: ionic_mid
        d += flow.Arrow().at(split_bar.S, dx=-FORK_OFFSET).down(ARROW_GAP)
        d += (ionic := make_box('ionic_mid\n16ᵀ', w=3, color=C_ATTN[0], fill=C_ATTN[1]).anchor('N'))

        # Right fork: conc
        d += flow.Arrow().at(split_bar.S, dx=FORK_OFFSET).down(ARROW_GAP)
        d += (conc := make_box('conc(t+1)\n4ᵀ  [Na⁺,K⁺,Ca²⁺,Ca_ss]', w=3.5, color=C_NERNST[0], fill=C_NERNST[1]).anchor('N'))
        d += flow.Box(w=2.2, h=0.4).at(conc.E, dx=0.5).anchor('W').label('attention only').color(C_NERNST[0])

        # --- MLP Stack ---
        d += flow.Arrow().at(ionic.S).down(ARROW_GAP)
        d += (mlp := make_box('Markov MLP\nPre-RMSNorm → W₁(16,16) → GELU → W₂(16,16)',
                              w=4.5, h=TALL_H, color=C_MLP[0], fill=C_MLP[1]).anchor('N'))

        # --- Alpha Mixing ---
        d += flow.Arrow().at(mlp.S).down(ARROW_GAP * 0.6)
        d += (amix := make_box('Learned α Mixing\n(1−α)·z_mid + α·corr, 16 params',
                               w=4.5, h=0.8, color=C_MLP[0], fill=C_MLP[1]).anchor('N'))

        # Skip connection label (z_mid)
        d += flow.Line().at(ionic.W).left(0.8).color(C_MLP[0])
        d += flow.Arrow().down().toy(amix.W).color(C_MLP[0])
        d += flow.Arrow().to(amix.W).color(C_MLP[0])

        # --- ionic_state(t+1) ---
        d += flow.Arrow().at(amix.S).down(ARROW_GAP)
        d += (ionicout := make_box('ionic_state(t+1)\n16ᵀ', w=3.5, color=C_MLP[0]).anchor('N'))

        # --- Compression ---
        d += flow.Arrow().at(ionicout.S).down(ARROW_GAP)
        d += (comp := make_box('Compression MLP\n20→12→12→8  (2×GELU)\nfull carried_state input',
                               w=4.5, h=1.2, color=C_COMP[0], fill=C_COMP[1]).anchor('N'))

        # --- Beta Mixing ---
        d += flow.Arrow().at(comp.S).down(ARROW_GAP * 0.6)
        d += (bmix := make_box('Learned β Mixing\nlin + nonlin, 8 params',
                               w=4, h=0.7, color=C_COMP[0], fill=C_COMP[1]).anchor('N'))

        # Linear bypass label
        d += flow.Line().at(ionicout.E).right(1.0).color(C_COMP[0])
        d += flow.Arrow().down().toy(bmix.E).color(C_COMP[0])
        d += flow.Arrow().to(bmix.E).color(C_COMP[0])

        # --- cond_lat(t+1) ---
        d += flow.Arrow().at(bmix.S).down(ARROW_GAP)
        d += (condout := make_box('cond_lat(t+1)\n8ᵀ', w=3, color=C_COMP[0]).anchor('N'))

        # --- Nernst Branch ---
        nernst_y_target = comp.S[1]  # align with compression
        d += flow.Arrow().at(conc.S).down().toy(nernst_y_target)
        d += (nernst := make_box('Nernst Equation\nfixed physics, 0 params',
                                 w=3.5, h=0.9, color=C_NERNST[0], fill=C_NERNST[1]).anchor('N'))

        d += flow.Arrow().at(nernst.S).down(ARROW_GAP)
        d += (nstout := make_box('nernst_st(t+1)\n[E_Na,E_K,E_Ca,E_Ks,conc×4] = 8ᵀ',
                                 w=4, h=0.8, color=C_NERNST[0]).anchor('N'))

        # --- Scaffolds (dashed) ---
        d += flow.Arrow().at(ionicout.W).left(2).color('gray')
        d += flow.Box(w=2.5, h=0.5).anchor('E').label('gate dec\n16→12').color('gray').fill(C_SCAFFOLD[1])

        d += flow.Arrow().at(condout.W).left(2).color('gray')
        d += flow.Box(w=2.5, h=0.5).anchor('E').label('gate dec\n8→12').color('gray').fill(C_SCAFFOLD[1])

        # ================================================================
        # STAGE 2 (RIGHT COLUMN)
        # ================================================================
        stage2_x = cs.center[0] + COL_GAP

        d += make_box('Stage 2: Current Readout\n(ON critical path · 118 params)',
                      w=5.5, h=0.8, color=C_READOUT[0], fill='lavenderblush').at((stage2_x, cs.N[1] + ARROW_GAP + 0.8))

        # Inputs
        d += flow.Arrow().down(ARROW_GAP)
        d += (condR := make_box('cond_lat(t)  8ᵀ', w=3, color=C_COMP[0]).anchor('N'))

        d += make_box('nernst_st(t)  8ᵀ', w=3, color=C_NERNST[0]).at(condR.E, dx=2).anchor('W')

        # Normalize
        d += flow.Arrow().at(condR.S).down(ARROW_GAP)
        loc_norm = d.here
        d += (norm := make_box('Normalize\nnernst_st + Vm → env 9ᵀ',
                               w=4.5, h=0.9, color=C_READOUT[0], fill=C_READOUT[1]).anchor('N'))

        # Embeddings
        d += flow.Arrow().at(norm.S).down(ARROW_GAP)
        d += (embed := make_box('Embeddings\ne_q(8,4)  e_k(9,4)  e_v(9,1)',
                                w=5, h=0.9, color=C_READOUT[0], fill=C_READOUT[1]).anchor('N'))

        # Scores
        d += flow.Arrow().at(embed.S).down(ARROW_GAP)
        d += (scores := make_box('QKᵀ/√4  scores (8,9)\nNO softmax',
                                 w=4, h=0.9, color=C_READOUT[0], fill=C_READOUT[1]).anchor('N'))

        # Attended
        d += flow.Arrow().at(scores.S).down(ARROW_GAP)
        d += (attended := make_box('scores × V → attended (8,)',
                                   w=4, h=0.7, color=C_READOUT[0], fill=C_READOUT[1]).anchor('N'))

        # Output MLP
        d += flow.Arrow().at(attended.S).down(ARROW_GAP)
        d += (omlp := make_box('Output MLP\nLin(8,4) → GELU → Lin(4,1)',
                               w=4, h=0.9, color=C_READOUT[0], fill=C_READOUT[1]).anchor('N'))

        # I_ion
        d += flow.Arrow().at(omlp.S).down(ARROW_GAP)
        d += make_box('I_ion(t)\nscalar', w=2.5, h=0.7, color=C_READOUT[0]).anchor('N')

        d.save('/home/norepinephrine/Documents/Heart-Conduction/Images/ionic_v3_schemdraw.png', dpi=DPI)
        print(f'Saved at {DPI} DPI')


if __name__ == '__main__':
    main()
