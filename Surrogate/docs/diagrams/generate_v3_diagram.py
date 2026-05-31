#!/usr/bin/env python3
"""Generate the Ionic Surrogate v3 TikZ architecture diagram.

All layout parameters are defined as variables at the top. Node positions
are computed from these parameters -- no hardcoded coordinates in the TikZ
body. Run this script to produce .tex, .pdf, and .jpg outputs.

Usage:
    python generate_v3_diagram.py
"""

import os
import subprocess
import sys

# =====================================================================
# LAYOUT PARAMETERS (tune these, everything else computes)
# =====================================================================

# Vertical gaps
VGAP = 1.4           # standard vertical gap between rows
VGAP_SMALL = 0.8     # small gap (within sub-stages)
VGAP_SECTION = 1.6   # gap between sections (attention -> split, etc.)
VGAP_MERGE = 1.0     # gap from node to merge bar

# Horizontal parameters
LEFT_CENTER = 0.0    # left column center x
RIGHT_CENTER = 15.5  # right column center x
BOX_WIDTH = 2.2      # standard box width (cm)
BOX_HEIGHT = 2.4     # standard box height (em)
LINE_WIDTH = 1.0     # main flow line width (pt)
ROUND_CORNER = 4     # corner rounding (pt)

# Left column horizontal offsets (from LEFT_CENTER)
L_VM_OFFSET = -4.5
L_DT_OFFSET = -2.8
L_CS_OFFSET = 2.8
L_SKIP_OFFSET = 3.5
L_WV_OFFSET = -4.5
L_WK_OFFSET = -2.5
L_WQ_OFFSET = 2.8
L_SIG_OFFSET = -0.5
L_WOUT_OFFSET = -4.5
L_HAD_OFFSET = 1.0
L_ADD_OFFSET = 1.0
L_IONIC_OFFSET = -2.5
L_CONC_OFFSET = 3.5

# Right column horizontal offsets (from RIGHT_CENTER)
R_COND_OFFSET = -3.0
R_NST_OFFSET = 3.0
R_EQ_OFFSET = -3.0
R_EKV_OFFSET = 1.0
R_EV_OFFSET = 4.5

# =====================================================================
# DERIVED POSITIONS
# =====================================================================

# Left column x-positions
L_VM = LEFT_CENTER + L_VM_OFFSET
L_DT = LEFT_CENTER + L_DT_OFFSET
L_CS = LEFT_CENTER + L_CS_OFFSET
L_SKIP = LEFT_CENTER + L_SKIP_OFFSET
L_WV = LEFT_CENTER + L_WV_OFFSET
L_WK = LEFT_CENTER + L_WK_OFFSET
L_WQ = LEFT_CENTER + L_WQ_OFFSET
L_SIG = LEFT_CENTER + L_SIG_OFFSET
L_WOUT = LEFT_CENTER + L_WOUT_OFFSET
L_HAD = LEFT_CENTER + L_HAD_OFFSET
L_ADD = LEFT_CENTER + L_ADD_OFFSET
L_IONIC = LEFT_CENTER + L_IONIC_OFFSET
L_CONC = LEFT_CENTER + L_CONC_OFFSET

# Stack midpoint (between Vm and dt)
L_STACK_MID = (L_VM + L_DT) / 2.0

# Right column x-positions
R_COND = RIGHT_CENTER + R_COND_OFFSET
R_NST = RIGHT_CENTER + R_NST_OFFSET
R_EQ = RIGHT_CENTER + R_EQ_OFFSET
R_EKV = RIGHT_CENTER + R_EKV_OFFSET
R_EV = RIGHT_CENTER + R_EV_OFFSET

# Left column y-positions (top to bottom, computed from VGAP)
Y_INPUT = 0.0
Y_MERGE_VMDT = Y_INPUT - VGAP_MERGE         # merge bar for Vm+dt
Y_STACK = Y_MERGE_VMDT - VGAP_SMALL          # stack operator
Y_STACK_SPLIT = Y_STACK - VGAP_SMALL         # where stack trunk splits to Wv/Wk
Y_ROW_A = Y_STACK_SPLIT - VGAP              # Wv, Wk, Wq row
Y_QK_BAR = Y_ROW_A - VGAP_MERGE             # Q+K merge bar
Y_ROW_B = Y_QK_BAR - VGAP_MERGE             # sigma gate + Wout
Y_GT_BAR = Y_ROW_B - VGAP_MERGE             # gate+target merge bar
Y_ROW_C = Y_GT_BAR - VGAP_SMALL             # Hadamard (gate x diff)
Y_ROW_D = Y_ROW_C - VGAP                    # residual add
Y_ZMID = Y_ROW_D - VGAP_SMALL               # z_mid label
Y_SPLIT_BAR = Y_ZMID - 0.4                  # split merge bar
Y_SPLIT = Y_SPLIT_BAR - 0.6                 # split outputs (ionic_mid, conc)

# MLP stack
Y_MLP_TOP = Y_SPLIT - VGAP_SECTION          # Pre-RMSNorm (top of MLP stack)
# (MLP blocks stack adjacently, ~4 * 1.8em ~ 2.8cm total height)
Y_AMIX = Y_MLP_TOP - 4.2                    # learned alpha mixing
Y_IONIC_OUT = Y_AMIX - VGAP                 # ionic_state(t+1) output

# Compression
Y_COMP_TOP = Y_IONIC_OUT - VGAP_SECTION     # compression MLP
Y_BMIX = Y_COMP_TOP - VGAP                  # beta mixing
Y_COND_OUT = Y_BMIX - VGAP                  # cond_lat(t+1) output

# Nernst branch (right of conc output)
Y_NERNST = Y_COMP_TOP                        # Nernst equation (aligned with comp)
Y_NST_OUT = Y_NERNST - VGAP                 # nernst_st(t+1) output

# Right column y-positions
Y_R_NORM = Y_INPUT - VGAP * 1.7             # normalize block
Y_R_EMBED = Y_R_NORM - VGAP * 1.3           # e_q, e_k, e_v
Y_R_QK_BAR = Y_R_EMBED - VGAP_MERGE * 0.8  # Q+K merge bar
Y_R_SCORES = Y_R_QK_BAR - VGAP              # scores
Y_R_SV_BAR = Y_R_SCORES - VGAP_MERGE * 0.8 # scores+V merge bar
Y_R_ATTN = Y_R_SV_BAR - VGAP_SMALL          # attended
Y_R_MLP = Y_R_ATTN - VGAP                   # output MLP
Y_R_IION = Y_R_MLP - VGAP                   # I_ion output

# Bottom y (for annotations + legend)
Y_BOTTOM = min(Y_COND_OUT, Y_NST_OUT) - 1.5

# Scaffold positions
Y_SCAFF_FULL = Y_IONIC_OUT                   # gate decoder (full)
Y_SCAFF_COMP = Y_COND_OUT                    # gate decoder (compressed)
X_SCAFF = L_IONIC - 4.3                      # scaffold x position
X_SCAFF_MSE = L_CONC + 4.0                   # direct MSE scaffold

# Title and header y
Y_TITLE = 2.5
Y_SUBTITLE = 1.8
Y_COL_HEADER = 1.0
Y_COL_SUB = 0.6

# Title x (midpoint of left and right columns)
X_TITLE = (LEFT_CENTER + RIGHT_CENTER) / 2.0

# Vm routing to Stage 2 y-clearance
Y_VM_ROUTE = Y_INPUT + 0.15
X_VM_ROUTE_RIGHT = R_EV + 1.5

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def coord(x, y):
    """Format a TikZ coordinate."""
    return f"({x:.4g},{y:.4g})"


def node_at(x, y):
    """Format 'at (x,y)' for a TikZ node."""
    return f"at {coord(x, y)}"


def connect_down(x, y1, y2):
    """Vertical line segment from (x,y1) down to (x,y2)."""
    return f"\\draw[trunk] {coord(x, y1)} -- {coord(x, y2)};"


def connect_horiz(x1, x2, y):
    """Horizontal line segment from (x1,y) to (x2,y)."""
    return f"\\draw[trunk] {coord(x1, y)} -- {coord(x2, y)};"


def connect_flow_down(x, y1, y2):
    """Arrowed vertical flow from (x,y1) down to (x,y2)."""
    return f"\\draw[flow] {coord(x, y1)} -- {coord(x, y2)};"


def connect_L_right(x1, y1, x2, y2, style="flow"):
    """L-shaped: down from (x1,y1) to y2, then right to (x2,y2)."""
    return f"\\draw[{style}] {coord(x1, y1)} -- {coord(x1, y2)} -- {coord(x2, y2)};"


def connect_L_left(x1, y1, x2, y2, style="flow"):
    """L-shaped: down from (x1,y1) to y2, then left to (x2,y2)."""
    return f"\\draw[{style}] {coord(x1, y1)} -- {coord(x1, y2)} -- {coord(x2, y2)};"


def merge_bar(x1, x2, y):
    """Horizontal merge bar at height y between x1 and x2."""
    return f"\\draw[trunk] {coord(x1, y)} -- {coord(x2, y)};"


def flow_from_bar(x, y_bar, y_target):
    """Arrow from a merge bar down to a target y."""
    return f"\\draw[flow] {coord(x, y_bar)} -- {coord(x, y_target)};"


# =====================================================================
# TEX GENERATION
# =====================================================================

def generate_tex():
    """Build the complete .tex string."""

    # Collect lines
    lines = []
    def emit(s=""):
        lines.append(s)

    # --- Preamble ---
    emit(r"\documentclass[border=12pt]{standalone}")
    emit(r"\usepackage{tikz}")
    emit(r"\usepackage{amsmath,amssymb}")
    emit(r"\usetikzlibrary{positioning, calc, fit, backgrounds, arrows.meta}")
    emit()
    emit(r"% === COLORS ===")
    emit(r"\definecolor{attnblue}{HTML}{4A90D9}")
    emit(r"\definecolor{mlporange}{HTML}{E8913A}")
    emit(r"\definecolor{compgreen}{HTML}{2EAD6B}")
    emit(r"\definecolor{nernstpurp}{HTML}{8E6CC0}")
    emit(r"\definecolor{readoutred}{HTML}{D94A5E}")
    emit(r"\definecolor{scaffold}{HTML}{999999}")
    emit(r"\definecolor{inputcol}{HTML}{6C5CE7}")
    emit()
    emit(r"\newcommand{\dimtag}{\tiny\color{black!50}}")
    emit()
    emit(r"\begin{document}")
    emit(r"\begin{tikzpicture}[")
    emit(r"    >=Stealth,")
    emit(f"    every path/.append style={{rounded corners={ROUND_CORNER}pt}},")
    emit(f"    flow/.style={{->, line width={LINE_WIDTH}pt}},")
    emit(f"    trunk/.style={{line width={LINE_WIDTH}pt}},")
    emit(r"    scaffoldflow/.style={->, line width=0.7pt, dashed, scaffold},")
    emit(r"    layer/.style={")
    emit(f"        rectangle, draw=#1!70, rounded corners={ROUND_CORNER}pt,")
    emit(f"        fill=#1!12, minimum height={BOX_HEIGHT}em, minimum width={BOX_WIDTH}cm,")
    emit(r"        align=center, font=\footnotesize, inner sep=4pt")
    emit(r"    },")
    emit(r"    layer/.default=black,")
    emit(r"    mlpblock/.style={")
    emit(r"        rectangle, draw=mlporange!70, rounded corners=2pt,")
    emit(r"        fill=mlporange!12, minimum height=1.8em, minimum width=3.0cm,")
    emit(r"        align=center, font=\footnotesize, inner sep=2pt,")
    emit(r"        outer sep=0pt")
    emit(r"    },")
    emit(r"    inputnode/.style={")
    emit(f"        rectangle, draw=inputcol!70, rounded corners={ROUND_CORNER}pt,")
    emit(r"        fill=inputcol!12, minimum height=2.2em, minimum width=2.0cm,")
    emit(r"        align=center, font=\footnotesize\bfseries, inner sep=4pt")
    emit(r"    },")
    emit(r"    outputnode/.style={")
    emit(f"        rectangle, draw=#1!70, rounded corners={ROUND_CORNER}pt,")
    emit(r"        fill=#1!10, minimum height=2.2em, minimum width=2.0cm,")
    emit(r"        align=center, font=\footnotesize\bfseries, inner sep=4pt")
    emit(r"    },")
    emit(r"    outputnode/.default=black,")
    emit(r"    op/.style={")
    emit(r"        circle, draw=#1!70, fill=#1!18, inner sep=0pt,")
    emit(r"        minimum size=1.8em, font=\small\bfseries")
    emit(r"    },")
    emit(r"    op/.default=black,")
    emit(r"    scaffbox/.style={")
    emit(f"        rectangle, draw=scaffold!50, rounded corners={ROUND_CORNER}pt,")
    emit(r"        fill=scaffold!6, minimum height=2.0em, minimum width=2.0cm,")
    emit(r"        align=center, font=\scriptsize\color{scaffold}, inner sep=3pt")
    emit(r"    },")
    emit(r"    seclabel/.style={font=\scriptsize\bfseries, text=#1!80},")
    emit(r"]")
    emit()

    # =====================================================================
    # TITLE
    # =====================================================================
    emit(r"% === TITLE ===")
    emit(f"\\node[font=\\Large\\bfseries] {node_at(X_TITLE, Y_TITLE)} "
         r"{Ionic Surrogate v3};")
    emit(f"\\node[font=\\small, text=black!45] {node_at(X_TITLE, Y_SUBTITLE)}")
    emit(r"    {1,454 inference params $\cdot$ 970 critical-path FLOPs "
         r"$\cdot$ 3,292 background FLOPs};")
    emit()

    # =====================================================================
    # COLUMN HEADERS
    # =====================================================================
    emit(r"% === COLUMN HEADERS ===")
    emit(f"\\node[font=\\small\\bfseries, text=attnblue] "
         f"{node_at(LEFT_CENTER, Y_COL_HEADER)}")
    emit(r"    {Stage 1: State Evolution};")
    emit(f"\\node[font=\\tiny, text=black!40] "
         f"{node_at(LEFT_CENTER, Y_COL_SUB)}")
    emit(r"    {off critical path $\cdot$ 1,336 params};")
    emit()
    emit(f"\\node[font=\\small\\bfseries, text=readoutred] "
         f"{node_at(RIGHT_CENTER, Y_COL_HEADER)}")
    emit(r"    {Stage 2: Current Readout};")
    emit(f"\\node[font=\\tiny, text=black!40] "
         f"{node_at(RIGHT_CENTER, Y_COL_SUB)}")
    emit(r"    {ON critical path $\cdot$ 118 params};")
    emit()

    # =====================================================================
    # INPUTS
    # =====================================================================
    emit(r"% === INPUTS ===")
    emit(f"\\node[inputnode, minimum width=1.2cm] (vm) "
         f"{node_at(L_VM, Y_INPUT)}")
    emit(r"    {$V_m$\\[-1pt]{\dimtag scalar}};")
    emit(f"\\node[inputnode, minimum width=1.2cm] (dt) "
         f"{node_at(L_DT, Y_INPUT)}")
    emit(r"    {$\Delta t$\\[-1pt]{\dimtag scalar}};")
    emit(f"\\node[inputnode, minimum width=2.4cm] (cs) "
         f"{node_at(L_CS, Y_INPUT)}")
    emit(r"    {carried\_state$(t)$\\[-1pt]{\dimtag $20^\top$}};")
    emit()
    emit(f"\\node[inputnode, minimum width=2.2cm] (condR) "
         f"{node_at(R_COND, Y_INPUT)}")
    emit(r"    {cond\_lat$(t)$\\[-1pt]{\dimtag $8^\top$}};")
    emit(f"\\node[inputnode, minimum width=2.2cm] (nstR) "
         f"{node_at(R_NST, Y_INPUT)}")
    emit(r"    {nernst\_st$(t)$\\[-1pt]{\dimtag $8^\top$}};")
    emit()

    # =====================================================================
    # STAGE 1: CROSS-ATTENTION
    # =====================================================================
    emit(r"% === STAGE 1: CROSS-ATTENTION ===")
    Y_SEC_LABEL = (Y_INPUT + Y_MERGE_VMDT) / 2.0
    emit(f"\\node[seclabel=attnblue, anchor=west] "
         f"{node_at(L_VM - 0.7, Y_SEC_LABEL)} {{ATTENTION}};")
    emit()

    # Vm + dt merge bar -> stack
    emit(r"% --- Vm + dt merge bar -> stack ---")
    emit(connect_down(L_VM, Y_INPUT - 0.35, Y_MERGE_VMDT))
    emit(connect_down(L_DT, Y_INPUT - 0.35, Y_MERGE_VMDT))
    emit(merge_bar(L_VM, L_DT, Y_MERGE_VMDT))
    emit(f"\\node[op=black, minimum size=1.5em] (stack) "
         f"{node_at(L_STACK_MID, Y_STACK)} {{\\tiny$\\circledast$}};")
    emit(connect_flow_down(L_STACK_MID, Y_MERGE_VMDT, Y_STACK + 0.35))
    emit()

    # ROW A: W_v, W_k, W_q
    emit(r"% --- ROW A: W_v, W_k, W_q ---")
    emit(f"\\node[layer=attnblue, minimum width=1.8cm] (wv) "
         f"{node_at(L_WV, Y_ROW_A)}")
    emit(r"    {$\mathbf{W}_v$\\[-1pt]{\dimtag Lin$(2,4)$}};")
    emit(f"\\node[layer=attnblue, minimum width=1.8cm] (wk) "
         f"{node_at(L_WK, Y_ROW_A)}")
    emit(r"    {$\mathbf{W}_k$\\[-1pt]{\dimtag Lin$(2,4)$}};")
    emit(f"\\node[layer=attnblue, minimum width=2.4cm] (wq) "
         f"{node_at(L_WQ, Y_ROW_A)}")
    emit(r"    {$\mathbf{W}_q$\\[-1pt]{\dimtag $(20,4)$ per-dim}};")
    emit()

    # stack splits to W_v and W_k
    emit(r"% stack splits to W_v and W_k")
    emit(connect_down(L_STACK_MID, Y_STACK - 0.35, Y_STACK_SPLIT))
    emit(f"\\draw[flow] {coord(L_STACK_MID, Y_STACK_SPLIT)} "
         f"-- {coord(L_WV, Y_STACK_SPLIT)} -- (wv.north);")
    emit(f"\\draw[flow] {coord(L_STACK_MID, Y_STACK_SPLIT)} "
         f"-- {coord(L_WK, Y_STACK_SPLIT)} -- (wk.north);")
    emit()

    # carried_state -> W_q
    Y_CS_BRANCH = Y_INPUT - 0.5 * (Y_INPUT - Y_MERGE_VMDT)
    emit(r"% carried_state -> W_q (main trunk)")
    emit(f"\\coordinate (csbranch) {node_at(L_CS, Y_CS_BRANCH)};")
    emit(r"\draw[trunk] (cs.south) -- (csbranch);")
    emit(r"\draw[flow] (csbranch) -- (wq.north);")
    emit()

    # Skip connection
    emit(r"% Skip connection: carried_state -> Hadamard and Add")
    emit(f"\\draw[trunk, attnblue!40] (csbranch) -- {coord(L_SKIP, Y_CS_BRANCH)};")
    emit(f"\\draw[trunk, attnblue!40] {coord(L_SKIP, Y_CS_BRANCH)} "
         f"-- {coord(L_SKIP, Y_ROW_C)};")
    emit(f"\\draw[flow, attnblue!40] {coord(L_SKIP, Y_ROW_C)} "
         f"-- {coord(L_HAD, Y_ROW_C)};")
    emit(f"\\draw[trunk, attnblue!40] {coord(L_SKIP, Y_ROW_C)} "
         f"-- {coord(L_SKIP, Y_ROW_D)};")
    emit(f"\\draw[flow, attnblue!40] {coord(L_SKIP, Y_ROW_D)} "
         f"-- {coord(L_ADD, Y_ROW_D)};")
    emit()

    # Q+K merge bar
    emit(r"% --- Q+K merge bar ---")
    emit(connect_down(L_WK, Y_ROW_A - 0.35, Y_QK_BAR))
    emit(connect_down(L_WQ, Y_ROW_A - 0.35, Y_QK_BAR))
    emit(merge_bar(L_WK, L_WQ, Y_QK_BAR))
    QK_MID_X = L_SIG
    emit(flow_from_bar(QK_MID_X, Y_QK_BAR, Y_ROW_B + 0.35))
    emit()

    # ROW B: sigma gate + W_out
    emit(r"% --- ROW B: sigma gate + W_out ---")
    emit(f"\\node[layer=attnblue, minimum width=3.0cm] (siggate) "
         f"{node_at(L_SIG, Y_ROW_B)}")
    emit(r"    {$\sigma(Q\!\cdot\!K^\top\!/\!\sqrt{4})$\\[-1pt]"
         r"{\dimtag gate $20^\top$}};")
    emit(f"\\node[layer=attnblue, minimum width=1.8cm] (wout) "
         f"{node_at(L_WOUT, Y_ROW_B)}")
    emit(r"    {$\mathbf{W}_{out}$\\[-1pt]{\dimtag $(4,20)$}};")
    emit(r"\draw[flow] (wv.south) -- (wout.north);")
    emit()

    # gate + target merge bar
    emit(r"% --- gate + target merge bar ---")
    emit(connect_down(L_SIG, Y_ROW_B - 0.35, Y_GT_BAR))
    emit(connect_down(L_WOUT, Y_ROW_B - 0.35, Y_GT_BAR))
    emit(merge_bar(L_WOUT, L_SIG, Y_GT_BAR))
    GT_MID_X = (L_WOUT + L_SIG) / 2.0
    emit()

    # ROW C: Hadamard
    emit(r"% --- ROW C: Hadamard ---")
    emit(f"\\node[op=attnblue] (had1) {node_at(L_HAD, Y_ROW_C)} "
         r"{$\otimes$};")
    emit(f"\\draw[flow] {coord(GT_MID_X, Y_GT_BAR)} "
         f"-- {coord(GT_MID_X, Y_ROW_C)} -- (had1.west);")
    emit(f"\\node[font=\\tiny, text=black!50, anchor=west] "
         f"{node_at(L_HAD + 1.2, Y_ROW_C)}")
    emit(r"    {gate $\otimes$ (target $-$ prev)};")
    emit()

    # ROW D: residual add
    emit(r"% --- ROW D: residual add ---")
    emit(f"\\node[op=attnblue] (add1) {node_at(L_ADD, Y_ROW_D)} "
         r"{$\oplus$};")
    emit(r"\draw[flow] (had1.south) -- (add1.north);")
    emit(f"\\node[font=\\tiny, text=black!50, anchor=west] "
         f"{node_at(L_ADD + 1.2, Y_ROW_D)}")
    emit(r"    {residual};")
    emit()

    # z_mid label — next to the trunk between add1 and fork
    emit(r"% --- z_mid label ---")
    emit(f"\\node[font=\\tiny, text=black!50, anchor=west] "
         f"{node_at(L_ADD + 0.5, Y_ZMID)}")
    emit(r"    {$z_{\text{mid}}$ $20^\top$};")
    emit()

    # =====================================================================
    # POST-ATTENTION SPLIT
    # =====================================================================
    emit(r"% === POST-ATTENTION SPLIT ===")
    emit(f"\\node[outputnode=attnblue, minimum width=2.4cm] (ionicmid) "
         f"{node_at(L_IONIC, Y_SPLIT)}")
    emit(r"    {ionic\_mid\\[-1pt]{\dimtag $16^\top$}};")
    emit(f"\\node[outputnode=nernstpurp, minimum width=2.4cm] (concout) "
         f"{node_at(L_CONC, Y_SPLIT)}")
    emit(r"    {conc$(t\!+\!1)$\\[-1pt]{\dimtag $4^\top$}};")
    emit()

    # Trunk from add -> straight down -> horizontal fork to ionic_mid and conc
    # add1 is at x=L_ADD. Go straight down to fork level, then fork left/right.
    FORK_X = L_ADD  # fork from same x as add1 — no horizontal jog
    emit(f"% Split: add1 straight down, then fork left (ionic) and right (conc)")
    emit(f"\\draw[trunk] (add1.south) -- {coord(FORK_X, Y_SPLIT_BAR)};")
    emit(f"\\draw[flow] {coord(FORK_X, Y_SPLIT_BAR)} "
         f"-- {coord(L_IONIC, Y_SPLIT_BAR)} -- (ionicmid.north);")
    emit(f"\\draw[flow] {coord(FORK_X, Y_SPLIT_BAR)} "
         f"-- {coord(L_CONC, Y_SPLIT_BAR)} -- (concout.north);")
    emit()

    emit(f"\\node[font=\\tiny, text=nernstpurp!70, anchor=west] "
         f"{node_at(L_CONC + 1.5, Y_SPLIT)}")
    emit(r"    {DONE (attention-only)};")
    emit()

    # =====================================================================
    # MLP STACK
    # =====================================================================
    emit(r"% === MLP STACK ===")
    emit(f"\\node[seclabel=mlporange, anchor=west] "
         f"{node_at(L_IONIC - 1.8, Y_MLP_TOP + 0.5)}")
    emit(r"    {MLP STACK};")
    emit()

    emit(f"\\node[mlpblock] (mlp1) {node_at(L_IONIC, Y_MLP_TOP)}")
    emit(r"    {Pre-RMSNorm\\[-1pt]{\dimtag 0 params}};")
    emit(r"\node[mlpblock, anchor=north] (mlp2) at (mlp1.south)")
    emit(r"    {$\mathbf{W}_1$ Lin$(16,16)$};")
    emit(r"\node[mlpblock, anchor=north] (mlp3) at (mlp2.south)")
    emit(r"    {GELU};")
    emit(r"\node[mlpblock, anchor=north] (mlp4) at (mlp3.south)")
    emit(r"    {$\mathbf{W}_2$ Lin$(16,16)$};")
    emit()
    emit(r"\draw[flow] (ionicmid.south) -- (mlp1.north);")
    emit()

    # =====================================================================
    # LEARNED ALPHA MIXING
    # =====================================================================
    emit(r"% === LEARNED ALPHA MIXING ===")
    emit(f"\\node[layer=mlporange, minimum width=3.6cm] (amix) "
         f"{node_at(L_IONIC, Y_AMIX)}")
    emit(r"    {Learned $\alpha$ Mixing\\[-1pt]"
         r"{\dimtag $(1\!-\!\alpha)\!\cdot\!z_{\text{mid}} + \alpha\!\cdot\!\text{corr}$, 16 params}};")
    emit(r"\draw[flow] (mlp4.south) -- (amix.north);")
    emit()

    # z_mid residual skip -> alpha mixing
    SKIP_X = L_IONIC - 1.5
    Y_SKIP_MID = (Y_SPLIT + Y_AMIX) / 2.0
    emit(r"% z_mid residual skip -> alpha mixing")
    emit(f"\\draw[flow, mlporange!50] (ionicmid.west) "
         f"-- {coord(SKIP_X, Y_SPLIT)} "
         f"-- {coord(SKIP_X, Y_AMIX)} -- (amix.west);")
    emit(f"\\node[font=\\tiny, text=mlporange!50, anchor=east] "
         f"{node_at(SKIP_X - 0.2, Y_SKIP_MID)}")
    emit(r"    {$z_{\text{mid}}$ skip};")
    emit()

    # =====================================================================
    # IONIC STATE OUTPUT
    # =====================================================================
    emit(r"% === IONIC STATE OUTPUT ===")
    emit(f"\\node[outputnode=mlporange, minimum width=2.6cm] (ionicout) "
         f"{node_at(L_IONIC, Y_IONIC_OUT)}")
    emit(r"    {ionic\_state$(t\!+\!1)$\\[-1pt]{\dimtag $16^\top$}};")
    emit(r"\draw[flow] (amix.south) -- (ionicout.north);")
    emit()

    # =====================================================================
    # COMPRESSION
    # =====================================================================
    emit(r"% === COMPRESSION ===")
    emit(f"\\node[seclabel=compgreen, anchor=west] "
         f"{node_at(L_IONIC - 1.8, Y_COMP_TOP + 0.5)}")
    emit(r"    {COMPRESSION};")
    emit()

    emit(f"\\node[layer=compgreen, minimum width=3.2cm] (comp) "
         f"{node_at(L_IONIC, Y_COMP_TOP)}")
    emit(r"    {Compression MLP\\[-1pt]"
         r"{\dimtag $16\!\to\!12\!\to\!12\!\to\!8$ (2$\times$GELU)}};")
    emit(r"\draw[flow] (ionicout.south) -- (comp.north);")
    emit()

    emit(f"\\node[layer=compgreen, minimum width=3.2cm] (bmix) "
         f"{node_at(L_IONIC, Y_BMIX)}")
    emit(r"    {Learned $\beta$ Mixing\\[-1pt]"
         r"{\dimtag lin $+$ nonlin, 8 params}};")
    emit(r"\draw[flow] (comp.south) -- (bmix.north);")
    emit()

    # Linear bypass skip
    BYPASS_X = L_IONIC + 1.8
    Y_BYPASS_MID = (Y_IONIC_OUT + Y_BMIX) / 2.0
    emit(r"% Linear bypass skip")
    emit(f"\\draw[flow, compgreen!50] (ionicout.east) "
         f"-- {coord(BYPASS_X, Y_IONIC_OUT)} "
         f"-- {coord(BYPASS_X, Y_BMIX)} -- (bmix.east);")
    emit(f"\\node[font=\\tiny, text=compgreen!50, anchor=west] "
         f"{node_at(BYPASS_X + 0.2, Y_BYPASS_MID)}")
    emit(r"    {$W_{\text{lin}}$ $(16\!\to\!8)$};")
    emit()

    emit(f"\\node[outputnode=compgreen, minimum width=2.6cm] (condout) "
         f"{node_at(L_IONIC, Y_COND_OUT)}")
    emit(r"    {cond\_lat$(t\!+\!1)$\\[-1pt]{\dimtag $8^\top$}};")
    emit(r"\draw[flow] (bmix.south) -- (condout.north);")
    emit()

    # =====================================================================
    # NERNST BRANCH
    # =====================================================================
    emit(r"% === NERNST BRANCH ===")
    emit(f"\\node[seclabel=nernstpurp, anchor=west] "
         f"{node_at(L_CONC - 1.5, Y_NERNST + 0.5)}")
    emit(r"    {NERNST};")
    emit()

    emit(f"\\node[layer=nernstpurp, minimum width=2.8cm] (nernst) "
         f"{node_at(L_CONC, Y_NERNST)}")
    emit(r"    {Nernst Equation\\[-1pt]{\dimtag fixed physics, 0 params}};")
    emit(r"\draw[flow] (concout.south) -- (nernst.north);")
    emit()

    emit(f"\\node[outputnode=nernstpurp, minimum width=2.6cm] (nstout) "
         f"{node_at(L_CONC, Y_NST_OUT)}")
    emit(r"    {nernst\_st$(t\!+\!1)$\\[-1pt]{\dimtag $8^\top$}};")
    emit(r"\draw[flow] (nernst.south) -- (nstout.north);")
    emit()

    # =====================================================================
    # STAGE 2: CROSS-ATTENTION READOUT
    # =====================================================================
    emit(r"% === STAGE 2: CROSS-ATTENTION READOUT ===")
    Y_S2_LABEL = (Y_INPUT + Y_R_NORM) / 2.0
    emit(f"\\node[seclabel=readoutred, anchor=west] "
         f"{node_at(R_COND - 1.7, Y_S2_LABEL)}")
    emit(r"    {CROSS-ATTENTION (no softmax)};")
    emit()

    # Normalize block
    emit(r"% --- Normalize ---")
    emit(f"\\node[layer=readoutred, minimum width=4.0cm] (norm2) "
         f"{node_at(R_NST, Y_R_NORM)}")
    emit(r"    {normalize\\[-1pt]"
         r"{\dimtag nernst\_st$(t)$ $+$ $V_m$ $\to$ env $9^\top$}};")
    emit(r"\draw[flow] (nstR.south) -- (norm2.north);")
    emit()

    # Vm -> normalize: route above inputs
    emit(r"% Vm -> normalize route")
    emit(f"\\draw[flow, inputcol!45]")
    emit(f"    (vm.north) -- ++(0, 0.15) "
         f"-- {coord(X_VM_ROUTE_RIGHT, Y_VM_ROUTE)} "
         f"-- {coord(X_VM_ROUTE_RIGHT, Y_R_NORM)} -- (norm2.east);")
    VM_LABEL_X = (LEFT_CENTER + RIGHT_CENTER) / 2.0
    emit(f"\\node[font=\\tiny, text=inputcol!55, anchor=south] "
         f"{node_at(VM_LABEL_X, Y_VM_ROUTE + 0.05)} {{$V_m$}};")
    emit()

    # Embedding row: e_q, e_k, e_v
    emit(r"% --- Embedding row ---")
    emit(f"\\node[layer=readoutred, minimum width=2.2cm] (eq) "
         f"{node_at(R_EQ, Y_R_EMBED)}")
    emit(r"    {$\mathbf{e}_q$\\[-1pt]{\dimtag $(8,4)$}};")
    emit(f"\\node[layer=readoutred, minimum width=2.2cm] (ek) "
         f"{node_at(R_EKV, Y_R_EMBED)}")
    emit(r"    {$\mathbf{e}_k$\\[-1pt]{\dimtag $(9,4)$}};")
    emit(f"\\node[layer=readoutred, minimum width=2.0cm] (ev) "
         f"{node_at(R_EV, Y_R_EMBED)}")
    emit(r"    {$\mathbf{e}_v$\\[-1pt]{\dimtag $(9,1)$}};")
    emit()

    # cond_lat(t) -> e_q
    emit(r"\draw[flow] (condR.south) -- (eq.north);")
    emit()

    # normalize -> e_k and e_v (split)
    Y_NORM_SPLIT = (Y_R_NORM + Y_R_EMBED) / 2.0
    emit(r"% normalize -> e_k and e_v")
    emit(f"\\coordinate (normsplit) {node_at(R_NST, Y_NORM_SPLIT)};")
    emit(r"\draw[trunk] (norm2.south) -- (normsplit);")
    emit(f"\\draw[flow] (normsplit) -- {coord(R_EKV, Y_NORM_SPLIT)} -- (ek.north);")
    emit(f"\\draw[flow] (normsplit) -- {coord(R_EV, Y_NORM_SPLIT)} -- (ev.north);")
    emit()

    # Dimension labels
    emit(r"% Dimension labels under embeddings")
    Y_DIM_LABEL = Y_R_EMBED - 0.6
    emit(f"\\node[font=\\tiny, text=black!50, anchor=north] "
         f"{node_at(R_EQ, Y_DIM_LABEL)} {{Q $(8,4)$}};")
    emit(f"\\node[font=\\tiny, text=black!50, anchor=north] "
         f"{node_at(R_EKV, Y_DIM_LABEL)} {{K $(9,4)$}};")
    emit(f"\\node[font=\\tiny, text=black!50, anchor=north] "
         f"{node_at(R_EV, Y_DIM_LABEL)} {{V $(9,1)$}};")
    emit()

    # Q + K merge bar -> scores
    emit(r"% --- Q + K merge bar -> scores ---")
    emit(connect_down(R_EQ, Y_R_EMBED - 0.35, Y_R_QK_BAR))
    emit(connect_down(R_EKV, Y_R_EMBED - 0.35, Y_R_QK_BAR))
    emit(merge_bar(R_EQ, R_EKV, Y_R_QK_BAR))
    R_QK_MID = (R_EQ + R_EKV) / 2.0
    emit(flow_from_bar(R_QK_MID, Y_R_QK_BAR, Y_R_SCORES + 0.35))
    emit()

    emit(f"\\node[layer=readoutred, minimum width=3.4cm] (scores) "
         f"{node_at(R_QK_MID, Y_R_SCORES)}")
    emit(r"    {$Q K^\top / \sqrt{4}$\\[-1pt]"
         r"{\dimtag scores $(8,9)$ -- no softmax}};")
    emit()

    # scores + V merge bar -> attended
    emit(r"% --- scores + V merge bar -> attended ---")
    emit(connect_down(R_QK_MID, Y_R_SCORES - 0.35, Y_R_SV_BAR))
    emit(connect_down(R_EV, Y_R_EMBED - 0.35, Y_R_SV_BAR))
    emit(merge_bar(R_QK_MID, R_EV, Y_R_SV_BAR))
    R_SV_MID = R_EKV
    emit(flow_from_bar(R_SV_MID, Y_R_SV_BAR, Y_R_ATTN + 0.35))
    emit()

    emit(f"\\node[layer=readoutred, minimum width=3.0cm] (attn) "
         f"{node_at(R_SV_MID, Y_R_ATTN)}")
    emit(r"    {scores $\times$ V\\[-1pt]{\dimtag attended $(8,)$}};")
    emit()

    # Output MLP
    emit(r"% --- Output MLP ---")
    emit(f"\\node[layer=readoutred, minimum width=4.2cm] (omlp) "
         f"{node_at(R_SV_MID, Y_R_MLP)}")
    emit(r"    {$W_1$ Lin$(8,4)$ $\to$ GELU $\to$ $W_2$ Lin$(4,1)$\\[-1pt]"
         r"{\dimtag output MLP}};")
    emit(r"\draw[flow] (attn.south) -- (omlp.north);")
    emit()

    # I_ion output
    emit(r"% --- I_ion output ---")
    emit(f"\\node[outputnode=readoutred, minimum width=2.2cm] (iion) "
         f"{node_at(R_SV_MID, Y_R_IION)}")
    emit(r"    {$I_{\text{ion}}(t)$\\[-1pt]{\dimtag scalar}};")
    emit(r"\draw[flow] (omlp.south) -- (iion.north);")
    emit()

    # =====================================================================
    # SCAFFOLD DECODERS
    # =====================================================================
    emit(r"% === SCAFFOLD DECODERS ===")
    emit(f"\\node[font=\\tiny\\itshape, text=scaffold] "
         f"{node_at(X_SCAFF, Y_SCAFF_FULL + 0.5)}")
    emit(r"    {scaffolds (training only)};")
    emit()

    emit(f"\\node[scaffbox] (sdec1) {node_at(X_SCAFF, Y_SCAFF_FULL)}")
    emit(r"    {gate decoder\\[-1pt]{\tiny $16\!\to\!12$}};")
    emit(r"\draw[scaffoldflow] (ionicout.west) -- (sdec1.east);")
    emit()

    emit(f"\\node[scaffbox] (sdec2) {node_at(X_SCAFF, Y_SCAFF_COMP)}")
    emit(r"    {gate decoder\\[-1pt]{\tiny $8\!\to\!12$}};")
    emit(r"\draw[scaffoldflow] (condout.west) -- (sdec2.east);")
    emit()

    emit(f"\\node[scaffbox] (smse) {node_at(X_SCAFF_MSE, Y_SPLIT)}")
    emit(r"    {direct MSE\\[-1pt]{\tiny vs true conc}};")
    emit(r"\draw[scaffoldflow] (concout.east) -- (smse.west);")
    emit()

    # =====================================================================
    # STAGE BACKGROUNDS
    # =====================================================================
    emit(r"% === STAGE BACKGROUNDS ===")
    emit(r"\begin{scope}[on background layer]")

    # Stage 1: Attention region
    ATTN_LEFT = L_VM - 1.3
    ATTN_RIGHT = L_CONC + 2.5
    ATTN_TOP = Y_INPUT + 0.3
    ATTN_BOT = Y_SPLIT_BAR - 0.3
    emit(f"    \\node[fit={{({ATTN_LEFT:.4g}, {ATTN_TOP:.4g})"
         f"({ATTN_RIGHT:.4g}, {ATTN_BOT:.4g})}},")
    emit(r"          fill=attnblue!4, draw=attnblue!18,")
    emit(r"          rounded corners=8pt, inner sep=0pt] {};")

    # Split region
    SPLIT_LEFT = L_IONIC - 2.0
    SPLIT_RIGHT = L_CONC + 2.0
    SPLIT_TOP = Y_SPLIT_BAR + 0.2
    SPLIT_BOT = Y_SPLIT - 0.6
    emit(f"    \\node[fit={{({SPLIT_LEFT:.4g}, {SPLIT_TOP:.4g})"
         f"({SPLIT_RIGHT:.4g}, {SPLIT_BOT:.4g})}},")
    emit(r"          fill=attnblue!3, draw=attnblue!12,")
    emit(r"          rounded corners=6pt, inner sep=0pt] {};")

    # MLP + alpha mixing
    MLP_LEFT = L_IONIC - 2.0
    MLP_RIGHT = L_IONIC + 2.2
    MLP_TOP = Y_MLP_TOP + 0.8
    MLP_BOT = Y_IONIC_OUT - 0.5
    emit(f"    \\node[fit={{({MLP_LEFT:.4g}, {MLP_TOP:.4g})"
         f"({MLP_RIGHT:.4g}, {MLP_BOT:.4g})}},")
    emit(r"          fill=mlporange!5, draw=mlporange!18,")
    emit(r"          rounded corners=8pt, inner sep=0pt] {};")

    # Compression
    COMP_LEFT = L_IONIC - 2.0
    COMP_RIGHT = L_IONIC + 2.2
    COMP_TOP = Y_COMP_TOP + 0.8
    COMP_BOT = Y_COND_OUT - 0.5
    emit(f"    \\node[fit={{({COMP_LEFT:.4g}, {COMP_TOP:.4g})"
         f"({COMP_RIGHT:.4g}, {COMP_BOT:.4g})}},")
    emit(r"          fill=compgreen!5, draw=compgreen!18,")
    emit(r"          rounded corners=8pt, inner sep=0pt] {};")

    # Nernst
    NERNST_LEFT = L_CONC - 1.8
    NERNST_RIGHT = L_CONC + 1.8
    NERNST_TOP = Y_NERNST + 0.8
    NERNST_BOT = Y_NST_OUT - 0.5
    emit(f"    \\node[fit={{({NERNST_LEFT:.4g}, {NERNST_TOP:.4g})"
         f"({NERNST_RIGHT:.4g}, {NERNST_BOT:.4g})}},")
    emit(r"          fill=nernstpurp!5, draw=nernstpurp!18,")
    emit(r"          rounded corners=8pt, inner sep=0pt] {};")

    # Stage 2
    S2_LEFT = R_COND - 1.7
    S2_RIGHT = R_EV + 1.3
    S2_TOP = Y_INPUT + 0.3
    S2_BOT = Y_R_IION - 0.6
    emit(f"    \\node[fit={{({S2_LEFT:.4g}, {S2_TOP:.4g})"
         f"({S2_RIGHT:.4g}, {S2_BOT:.4g})}},")
    emit(r"          fill=readoutred!4, draw=readoutred!18,")
    emit(r"          rounded corners=8pt, inner sep=0pt] {};")

    emit(r"\end{scope}")
    emit()

    # =====================================================================
    # ANNOTATIONS
    # =====================================================================
    emit(r"% === ANNOTATIONS ===")
    ANNOT_LEFT = ATTN_LEFT - 2.8
    emit(f"\\node[font=\\tiny, text=black!40, anchor=north west, text width=16cm]")
    emit(f"    {node_at(ANNOT_LEFT, Y_BOTTOM - 0.5)}")
    emit(r"    {External: $V_m^{n+1} = V_m + \Delta t(-I_{\text{ion}} "
         r"+ I_{\text{stim}})/C_m$")
    emit(r"    \quad---\quad $I_{\text{stim}}$ applied in operator splitting, "
         r"not a model input.")
    emit(r"    \quad Stage 1 runs on separate CUDA stream during diffusion solve.};")
    emit()

    # =====================================================================
    # LEGEND
    # =====================================================================
    emit(r"% === LEGEND ===")
    Y_LEGEND_TITLE = Y_BOTTOM - 1.5
    Y_LEGEND_ROW = Y_BOTTOM - 1.9
    emit(f"\\node[font=\\scriptsize\\bfseries, anchor=north west] "
         f"{node_at(ANNOT_LEFT, Y_LEGEND_TITLE)} {{Legend:}};")

    # Legend items -- spread across bottom
    LX = ANNOT_LEFT + 2.0
    emit(f"\\node[op=attnblue, minimum size=1.4em, font=\\scriptsize\\bfseries]")
    emit(f"    {node_at(LX, Y_LEGEND_ROW)} {{$\\oplus$}};")
    emit(f"\\node[font=\\tiny, anchor=west] "
         f"{node_at(LX + 0.7, Y_LEGEND_ROW)} {{elem.\\ add}};")

    LX2 = LX + 3.0
    emit(f"\\node[op=attnblue, minimum size=1.4em, font=\\scriptsize\\bfseries]")
    emit(f"    {node_at(LX2, Y_LEGEND_ROW)} {{$\\otimes$}};")
    emit(f"\\node[font=\\tiny, anchor=west] "
         f"{node_at(LX2 + 0.7, Y_LEGEND_ROW)} "
         r"    {gate $\otimes$ (target $-$ prev)};")

    LX3 = LX2 + 5.5
    emit(f"\\node[op=black, minimum size=1.4em, font=\\tiny\\bfseries]")
    emit(f"    {node_at(LX3, Y_LEGEND_ROW)} {{$\\circledast$}};")
    emit(f"\\node[font=\\tiny, anchor=west] "
         f"{node_at(LX3 + 0.7, Y_LEGEND_ROW)} {{stack}};")

    LX4 = LX3 + 2.5
    emit(f"\\draw[flow] {coord(LX4, Y_LEGEND_ROW)} -- "
         f"++(0.7, 0);")
    emit(f"\\node[font=\\tiny, anchor=west] "
         f"{node_at(LX4 + 0.8, Y_LEGEND_ROW)} {{data flow ({LINE_WIDTH}pt)}};")

    LX5 = LX4 + 4.0
    emit(f"\\draw[scaffoldflow] {coord(LX5, Y_LEGEND_ROW)} -- "
         f"++(0.7, 0);")
    emit(f"\\node[font=\\tiny, anchor=west] "
         f"{node_at(LX5 + 0.8, Y_LEGEND_ROW)} {{training only (0.7pt)}};")

    LX6 = LX5 + 4.0
    Y_BAR_TOP = Y_LEGEND_ROW + 0.3
    Y_BAR_BOT = Y_LEGEND_ROW - 0.3
    emit(f"\\draw[trunk] {coord(LX6, Y_BAR_TOP)} -- {coord(LX6, Y_BAR_BOT)};")
    emit(f"\\draw[trunk] {coord(LX6, Y_LEGEND_ROW)} -- {coord(LX6 + 1.0, Y_LEGEND_ROW)};")
    emit(f"\\node[font=\\tiny, anchor=west] "
         f"{node_at(LX6 + 1.2, Y_LEGEND_ROW)} {{merge/split bar}};")
    emit()

    # Close
    emit(r"\end{tikzpicture}")
    emit(r"\end{document}")

    return "\n".join(lines)


# =====================================================================
# MAIN: generate .tex, compile, convert
# =====================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_name = "ionic_surrogate_v3"
    tex_path = os.path.join(script_dir, f"{base_name}.tex")
    pdf_path = os.path.join(script_dir, f"{base_name}.pdf")
    jpg_base = os.path.join(script_dir, base_name)

    # 1. Generate .tex
    tex_content = generate_tex()
    with open(tex_path, "w") as f:
        f.write(tex_content)
    print(f"[1/3] Generated {tex_path}")

    # 2. Compile with pdflatex
    print(f"[2/3] Compiling with pdflatex...")
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory", script_dir,
         tex_path],
        capture_output=True, text=True, cwd=script_dir, timeout=60
    )
    if result.returncode != 0:
        print("pdflatex FAILED. Log output:")
        # Print the last 40 lines of the log for diagnosis
        log_lines = result.stdout.split("\n")
        for line in log_lines[-40:]:
            print(f"  {line}")
        sys.exit(1)

    if not os.path.exists(pdf_path):
        print(f"ERROR: {pdf_path} was not created")
        sys.exit(1)
    print(f"  -> {pdf_path}")

    # 3. Convert to JPG with pdftoppm
    print(f"[3/3] Converting to JPG with pdftoppm...")
    result = subprocess.run(
        ["pdftoppm", "-jpeg", "-r", "300", "-singlefile",
         pdf_path, jpg_base],
        capture_output=True, text=True, cwd=script_dir, timeout=30
    )
    jpg_path = f"{jpg_base}.jpg"
    if not os.path.exists(jpg_path):
        print(f"ERROR: {jpg_path} was not created")
        sys.exit(1)
    print(f"  -> {jpg_path}")

    print(f"\nDone. Files:")
    print(f"  .tex: {tex_path}")
    print(f"  .pdf: {pdf_path}")
    print(f"  .jpg: {jpg_path}")


if __name__ == "__main__":
    main()
