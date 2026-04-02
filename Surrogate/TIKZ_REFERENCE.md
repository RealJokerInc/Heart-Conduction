# TikZ Reference: Neural Network Architecture Diagrams

Collected reference material for drawing transformer-style architecture diagrams with
attention blocks, skip connections, element-wise operations, and normalization annotations.

---

## Table of Contents

1. [Package/Library Cheat Sheet](#1-packagelibrary-cheat-sheet)
2. [Style Definition Patterns](#2-style-definition-patterns)
3. [NNTikZ: Full Transformer (Vaswani)](#3-nntikz-full-transformer-vaswani)
4. [NNTikZ: Scaled Dot-Product Attention](#4-nntikz-scaled-dot-product-attention)
5. [NNTikZ: Multi-Head Attention](#5-nntikz-multi-head-attention)
6. [NNTikZ: Encoder-Only Transformer](#6-nntikz-encoder-only-transformer)
7. [NNTikZ: Decoder-Only Transformer](#7-nntikz-decoder-only-transformer)
8. [NNTikZ: LSTM Cell (Gate Operations)](#8-nntikz-lstm-cell-gate-operations)
9. [NNTikZ: GRU Cell (Gate Operations)](#9-nntikz-gru-cell-gate-operations)
10. [Petar Velickovic: Self-Attention](#10-petar-velickovic-self-attention)
11. [Skip Connection / Residual Block](#11-skip-connection--residual-block)
12. [Common Patterns Reference](#12-common-patterns-reference)
13. [PlotNeuralNet Style Primitives](#13-plotneuralnet-style-primitives)
14. [Sources](#14-sources)

---

## 1. Package/Library Cheat Sheet

### Core TikZ Libraries (union across all sources)

```latex
\usepackage{tikz}
\usetikzlibrary{
    positioning,          % relative node placement (right=of, above=of)
    calc,                 % coordinate arithmetic $(A)!0.5!(B)$
    fit,                  % bounding boxes around groups of nodes
    backgrounds,          % draw behind existing nodes (on background layer)
    arrows.meta,          % modern arrow tips (LaTeX, Stealth, Hooks)
    chains,               % sequential node placement (start chain)
    shapes.geometric,     % ellipse, diamond, regular polygon, etc.
    shapes,               % additional shape library
    decorations.pathreplacing,  % braces, ticks
    fadings,              % path fading for "omitted layers"
    quotes,               % edge["label"] shorthand
    ext.paths.ortho,      % orthogonal path routing (nntikz multihead)
}
```

### Optional but useful

```latex
\usepackage{pgfplots}              % for activation function plots
\usepackage{neuralnetwork}          % battlesnake/neural CTAN package
\usepackage{listofitems}            % array-based layer definitions (tikz.net style)
```

---

## 2. Style Definition Patterns

### NNTikZ Master Style Set (transformer family)

```latex
\begin{tikzpicture}[
    >=LaTeX,
    very thick,
    % --- Arrows ---
    arrow/.style={-latex, very thick, rounded corners=0.2cm},
    % --- Rectangular layer block ---
    layer/.style={
        rectangle, fill=white!10, rounded corners=1mm,
        inner xsep=0em, inner ysep=0.25em,
        minimum height=1.4em, align=center,
        text width=2.5cm, draw, very thick
    },
    % --- Enclosing block (encoder/decoder) ---
    block/.style={
        rectangle, fill=gray!10, rounded corners=3mm,
        draw, very thick
    },
    % --- Circular I/O node ---
    input/.style={
        circle, minimum width=2.25em, draw, fill=gray!10, thick
    },
    % --- Positional encoding sine wave inside circle ---
    do path picture/.style={
        path picture={
          \pgfpointdiff{\pgfpointanchor{path picture bounding box}{south west}}
            {\pgfpointanchor{path picture bounding box}{north east}}
          \pgfgetlastxy\x\y
          \tikzset{x=\x/2,y=\y/2}
          #1
        }
    },
    sin wave/.style={do path picture={
        \draw [line cap=round] (-3/4,0)
        sin (-3/8,1/2) cos (0,0) sin (3/8,-1/2) cos (3/4,0);
    }}
]
```

### NNTikZ Gate/Cell Style Set (LSTM/GRU family)

```latex
\begin{tikzpicture}[
    >=LaTeX,
    % --- RNN cell container ---
    cell/.style={
        rectangle, rounded corners=5mm, draw, very thick,
        minimum height=4cm, minimum width=6cm
    },
    % --- Function boxes (sigma, tanh) ---
    func/.style={rectangle, draw, inner sep=2pt, minimum height=0.4cm},
    % --- Operator circles (x, +) ---
    op/.style={circle, draw, inner sep=-0.5pt, minimum height=0.4cm},
    % --- Pointwise operation fill ---
    pointwise/.style={fill=gray!50},
    % --- Arrow variants ---
    arrow/.style={-latex, very thick},
    arrowc1/.style={arrow, rounded corners=.25cm},
    arrowc2/.style={arrow, rounded corners=.5cm},
    % --- Dashed training-only paths ---
    backprop/.style={arrow, dashed, gray},
]
```

### David Stutz Color-Coded Layer Styles

```latex
\definecolor{fc}{HTML}{1E90FF}      % fully-connected: blue
\definecolor{conv}{HTML}{FFA500}    % convolution: orange
\definecolor{pool}{HTML}{B22222}    % pooling: red
\definecolor{bn}{HTML}{FFD700}      % batch norm: gold

\tikzset{
    fc/.style={black, draw=black, fill=fc, rectangle, minimum height=1cm},
    conv/.style={black, draw=black, fill=conv, rectangle, minimum height=1cm},
    pool/.style={black, draw=black, fill=pool, rectangle, minimum height=1cm},
}
```

---

## 3. NNTikZ: Full Transformer (Vaswani)

Source: https://github.com/fraserlove/nntikz (MIT License)

```latex
\documentclass{standalone}

\usepackage{tikz}

\usetikzlibrary{positioning, chains, shapes.geometric, fit, shapes, arrows.meta, calc, backgrounds}

\begin{document}

\begin{tikzpicture}[
    >=LaTeX,
    very thick,
    arrow/.style={
        -latex,
        very thick,
        rounded corners=0.2cm
    },
    block/.style={
        rectangle,
        fill=gray!10,
        rounded corners=3mm,
        draw,
        very thick
    },
    layer/.style={
        rectangle,
        fill=white!10,
        rounded corners=1mm,
        inner xsep=0em,
        inner ysep=0.25em,
        minimum height=1.4em,
        align=center,
        text width=2.5cm,
        draw,
        very thick
    },
    input/.style={
        circle,
        minimum width=2.25em,
        draw,
        fill=gray!10,
        thick
    },
    do path picture/.style={%
        path picture={%
          \pgfpointdiff{\pgfpointanchor{path picture bounding box}{south west}}%
            {\pgfpointanchor{path picture bounding box}{north east}}%
          \pgfgetlastxy\x\y%
          \tikzset{x=\x/2,y=\y/2}%
          #1
        }
    },
    sin wave/.style={do path picture={
        \draw [line cap=round] (-3/4,0)
        sin (-3/8,1/2) cos (0,0) sin (3/8,-1/2) cos (3/4,0);
        }
    }
]

    \node[input] (iemb) at (1.25,-0.45) {$\mathbf{x}$};
    \node[input] (oemb) at (5.25,-0.45) {$\mathbf{y}$};

    \node[circle, draw, minimum size=0.25em, inner sep=0pt, above=1em of iemb] (sum1) {$\mathbf{+}$};
    \node[circle, draw, minimum size=0.25em, inner sep=0pt, above=1em of oemb] (sum2) {$\mathbf{+}$};
    \node [circle, draw, sin wave, minimum size=2em, left=0.8em of sum1] (pe1) {};
    \node [circle, draw, sin wave, minimum size=2em, right=0.8em of sum2] (pe2) {};

    \node[layer] (add1) at (1.25,3.43) {Add \& Norm};
    \node[layer] (attn1) at (1.25,2.58) {Multi-Head \vspace{-0.05cm} \linebreak Attention};
    \draw[] (attn1) -- (add1);
    \node[layer] (add4) at (1.25,5.98) {Add \& Norm};
    \node[layer] (ff1) at (1.25,5.13) {Feed \vspace{-0.05cm} \linebreak Forward};
    \draw[] (ff1) -- (add4);

    \node[layer] (add3) at (5.25,3.83) {Add \& Norm};
    \node[layer] (attn3) at (5.25,2.78) {Masked \vspace{-0.05cm} \linebreak Multi-Head \vspace{-0.05cm} \linebreak Attention};
    \draw[] (attn3) -- (add3);
    \node[layer] (add2) at (5.25,6.38) {Add \& Norm};
    \node[layer] (attn2) at (5.25,5.53) {Multi-Head \vspace{-0.05cm} \linebreak Attention};
    \draw[] (attn2) -- (add2);
    \node[layer] (add5) at (5.25,8.93) {Add \& Norm};
    \node[layer] (ff2) at (5.25,8.08) {Feed \vspace{-0.05cm} \linebreak Forward};
    \draw[] (ff2) -- (add5);

    \coordinate (d1) at ($(attn3.south east) + (0.75,-0.7)$);
    \coordinate (d2) at ($(add5.north west) + (-0.15,0.05)$);
    \coordinate (e1) at ($(attn1.south east) + (0.15,-0.7)$);
    \coordinate (e2) at ($(add4.north west) + (-0.75,0.05)$);
    \begin{scope}[on background layer]
        \node[block, fit=(d1) (d2)] (decoder) {};
        \node[block, fit=(e1) (e2)] (encoder) {};
    \end{scope}

    \node[layer, above=1.5em of add5] (linear) {Linear};
    \node[layer, above=1em of linear] (softmax) {Softmax};
    \node[input, above=1em of softmax] (probs) {$\hat{\mathbf{y}}$};

    \draw[arrow] (add1) -- (ff1);
    \draw[arrow] (add2) -- (ff2);
    \draw[arrow] (add5) -- (linear);
    \draw[arrow] (linear) -- (softmax);
    \draw[arrow] (iemb) -- (sum1);
    \draw[arrow] (sum1) -- (attn1);
    \draw[arrow] (oemb) -- (sum2);
    \draw[arrow] (sum2) -- (attn3);
    \draw[] (sum1) -- (pe1);
    \draw[] (sum2) -- (pe2);

    % Skip connections (encoder)
    \draw[arrow] (ff1.south)++(0, -0.6) -| ($(add4.west) + (-0.6,-0.5)$) |- (add4.west);
    \draw[arrow] (attn1.south)++(0, -0.6) -| ($(add1.west) + (-0.6,-0.5)$) |- (add1.west);

    % Skip connections (decoder)
    \draw[arrow] (attn3.south)++(0, -0.6) -| ($(add3.east) + (0.6,-0.5)$) |- (add3.east);
    \draw[arrow] (attn2.south)++(0, -0.6) -| ($(add2.east) + (0.6,-0.5)$) |- (add2.east);
    \draw[arrow] (ff2.south)++(0, -0.6) -| ($(add5.east) + (0.6,-0.5)$) |- (add5.east);

    % Q, K, V fan-out (encoder self-attention)
    \draw[arrow] (attn1.south)++(0, -0.4) -| ($(attn1.south) + (-1,0)$);
    \draw[arrow] (attn1.south)++(0, -0.4) -| ($(attn1.south) + (1,0)$);

    % Q, K, V fan-out (decoder masked self-attention)
    \draw[arrow] (attn3.south)++(0, -0.4) -| ($(attn3.south) + (-1,0)$);
    \draw[arrow] (attn3.south)++(0, -0.4) -| ($(attn3.south) + (1,0)$);

    % Cross-attention: encoder output -> decoder K, V
    \draw[arrow] (add4.north) |- ($(add4.north) + (1,1)$) -| ($(add4.north) + (2,-1)$) |- ($(attn2.south) + (-1,-0.4)$) -| ($(attn2.south) + (-1,0)$);
    \draw[arrow] (add4.north) |- ($(add4.north) + (1,1)$) -| ($(add4.north) + (2,-1)$) |- ($(attn2.south) + (0,-0.4)$) -| ($(attn2.south)$);
    \draw[arrow] (add3.north) |- ($(attn2.south) + (0.5,-0.4)$) -| ($(attn2.south) + (1,0)$);

    \node[] at ($(oemb.east) + (1.4,0)$) {(shifted right)};
    \node[] at ($(pe1.west) + (-0.3,0)$) {$\mathbf{P}$};
    \node[] at ($(pe2.east) + (0.3,0)$) {$\mathbf{P}$};

    \draw[arrow] (iemb) -- (sum1);
    \draw[arrow] (oemb) -- (sum2);
    \draw[arrow] (softmax) -- (probs);

    \node[anchor=east] at ($(encoder.west) + (-0.2,0)$) {$N\times$};
    \node[anchor=west] at ($(decoder.east) + (0.2,0)$) {$N\times$};
\end{tikzpicture}

\end{document}
```

---

## 4. NNTikZ: Scaled Dot-Product Attention

Source: https://github.com/fraserlove/nntikz (MIT License)

```latex
\documentclass{standalone}

\usepackage{tikz}

\usetikzlibrary{positioning, chains, shapes.geometric, fit, shapes, arrows.meta, calc}

\begin{document}

\begin{tikzpicture}[
    >=LaTeX,
    layer/.style={
        rectangle, fill=white!10, rounded corners=2mm,
        minimum height=2em, minimum width=3em, draw, thick
    },
    arrow/.style={-latex, very thick}
]

    % Inputs
    \node[] (Q) {$\mathbf{Q}$};
    \node[, right=1em of Q] (K) {$\mathbf{K}$};
    \node[, right=1em of K] (V) {$\mathbf{V}$};

    % Processing layers
    \node[layer, above=2em of $(Q)!0.5!(K)$] (matmul) {MatMul};
    \node[layer, above=1em of matmul] (scale) {Scale};
    \node[layer, above=1em of scale] (mask) {Mask (opt.)};
    \node[layer, above=1em of mask] (softmax) {Softmax};
    \node[layer, above=14.5em of $(Q)!0.8!(V)$] (matmul2) {MatMul};
    \node[above=1.4em of matmul2] (result) {Scaled Dot-Product Attention};

    % Connections
    \draw[arrow] (Q) -- (matmul);
    \draw[arrow] (K) -- (matmul);
    \draw[arrow] (matmul) -- (scale);
    \draw[arrow] (scale) -- (mask);
    \draw[arrow] (mask) -- (softmax);
    \draw[arrow] (softmax) -- (matmul2);
    \draw[arrow] (matmul2) -- (result);
    \path[draw, arrow] (V) -- ++(0, 14.5em);
\end{tikzpicture}

\end{document}
```

---

## 5. NNTikZ: Multi-Head Attention

Source: https://github.com/fraserlove/nntikz (MIT License)

Uses advanced features: `ext.paths.ortho`, shadowed/stacked nodes, transparency groups.

```latex
\documentclass{standalone}

\usepackage{tikz}

\usetikzlibrary{positioning, chains, shapes.geometric, fit, shapes, arrows.meta, calc, decorations.pathreplacing, ext.paths.ortho, quotes}

\begin{document}

\makeatletter
\tikzset{phantom node/.code=\tikz@addoption{\expandafter\let\csname pgf@sh@boxes@\tikz@shape\endcsname\pgfutil@empty}}
\makeatother

% Custom TikZ settings for shadowed (stacked) nodes
\tikzset{
  shadowed node xshift/.initial=1.5ex,
  shadowed node yshift/.initial=1ex,
  shadowed node list/.initial={2, 1},
  pics/shadowed node/.default=\pgfkeysvalueof{/tikz/shadowed node list},
  shadowed node/.pic={
    \foreach[expand list] \elem in {#1} {
      \scoped[transparency group, shadowed node calculation={\elem}]
        \node[style/.expand once=\tikzpictextoptions, phantom node,
              xshift={\elem*\pgfkeysvalueof{/tikz/shadowed node xshift}},
              yshift={\elem*\pgfkeysvalueof{/tikz/shadowed node yshift}}]
              (-\elem) {\tikzpictext};
    }
    \node[alias=-0, style/.expand once=\tikzpictextoptions] () {\tikzpictext};
  },
  set shadowed node calculation parameter/.style={
    shadowed node calculation/.style={opacity={(#1-##1+1)/(#1+1)}}
  },
  set shadowed node calculation parameter=2,
  overshoot line to/.style={
    to path={($(\tikztostart)!-(#1)!(\tikztotarget)$)--($(\tikztotarget)!-(#1)!(\tikztostart)$)\tikztonodes}
  },
  edges have transparency group/.style={
    execute at begin to={\scope[transparency group,#1]},
    execute at end to=\endscope
  }
}

\begin{tikzpicture}[
    >=LaTeX, thick, x=2cm, node distance=1.75em,
    cell/.style={
        rectangle, fill=white!10, rounded corners=2mm,
        minimum height=2em, minimum width=2.5em, draw, thick
    },
    arrow/.style={-latex, very thick}
]

    \node foreach[count=\i] \t in {$\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$} (VKQ-\i) at (\i, 0) {\t}
    pic foreach \i in {1, 2, 3}
        ["Linear" cell=red, above=of VKQ-\i] (Linear-\i) {shadowed node}
    pic["Scaled Dot-Product Attention" cell=red, above = of Linear-2]
        (Attention) {shadowed node}
    node[above = of Attention, cell=red] (Concat) {Concat}
    node[above = of Concat, cell=red] (Linear) {Linear}
    node[above = of Linear] (MHAtt) {Multi-Head Attention}
    [arrow, ortho/install shortcuts]
    foreach \i in {1, 2, 3}{
        (VKQ-\i) edge coordinate [pos=.2] (@) (Linear-\i)
        (Linear-\i) edge[|*] (Attention)
        foreach \j in {2, 1}{
        [
            edges have transparency group={shadowed node calculation=\j},
            behind path
        ]
        (@) edge[out=90, in=-90] (Linear-\i-\j)
        [in front of path]
        (Linear-\i-\j) edge[path only, |*] coordinate (@@) (Attention-\j)
                        edge[-] (@@)
        [behind path] (@@) edge[|*] (Attention-\j)
        }
    }
    foreach \i in {0, 1, 2}{
        [edges have transparency group={shadowed node calculation=\i}]
        (Attention-\i) edge[|*] (Concat)
    }
    (Concat) edge (Linear)
    (Linear) edge (MHAtt);

    % Brace label for H (number of heads)
    \draw[
        arrows={[arc=135]}, arrows={Hooks[left]-Hooks[right]},
        s/.style={shift={(3mm,-3.5mm)}}
    ]
    ([s]Attention-2.east) to[overshoot line to=1mm] coordinate(@) ([s]Attention.east);
    \path node[right=3mm] at (@) {$H$};
\end{tikzpicture}

\end{document}
```

---

## 6. NNTikZ: Encoder-Only Transformer

Source: https://github.com/fraserlove/nntikz (MIT License)

```latex
\documentclass{standalone}

\usepackage{tikz}

\usetikzlibrary{positioning, chains, shapes.geometric, fit, shapes, arrows.meta, calc, backgrounds}

\begin{document}

\begin{tikzpicture}[
    >=LaTeX,
    very thick,
    arrow/.style={-latex, very thick, rounded corners=0.2cm},
    block/.style={rectangle, fill=gray!10, rounded corners=3mm, draw, very thick},
    layer/.style={
        rectangle, fill=white!10, rounded corners=1mm,
        inner xsep=0em, inner ysep=0.25em,
        minimum height=1.4em, align=center,
        text width=2.5cm, draw, very thick
    },
    input/.style={circle, minimum width=2.25em, draw, fill=gray!10, thick},
    do path picture/.style={
        path picture={
          \pgfpointdiff{\pgfpointanchor{path picture bounding box}{south west}}
            {\pgfpointanchor{path picture bounding box}{north east}}
          \pgfgetlastxy\x\y
          \tikzset{x=\x/2,y=\y/2}
          #1
        }
    },
    sin wave/.style={do path picture={
        \draw [line cap=round] (-3/4,0)
        sin (-3/8,1/2) cos (0,0) sin (3/8,-1/2) cos (3/4,0);
    }}
]

    \node[input] (iemb) at (1.25,-0.45) {$\mathbf{x}$};

    \node[circle, draw, minimum size=0.25em, inner sep=0pt, above=1em of iemb] (sum1) {$\mathbf{+}$};
    \node [circle, draw, sin wave, minimum size=2em, left=0.8em of sum1] (pe1) {};

    \node[layer] (add1) at (1.25,3.43) {Add \& Norm};
    \node[layer] (attn1) at (1.25,2.58) {Multi-Head \vspace{-0.05cm} \linebreak Attention};
    \draw[] (attn1) -- (add1);
    \node[layer] (add4) at (1.25,5.98) {Add \& Norm};
    \node[layer] (ff1) at (1.25,5.13) {Feed \vspace{-0.05cm} \linebreak Forward};
    \draw[] (ff1) -- (add4);

    \coordinate (e1) at ($(attn1.south east) + (0.15,-0.7)$);
    \coordinate (e2) at ($(add4.north west) + (-0.75,0.05)$);
    \begin{scope}[on background layer]
        \node[block, fit=(e1) (e2)] (encoder) {};
    \end{scope}

    \draw[arrow] (add1) -- (ff1);
    \draw[arrow] (iemb) -- (sum1);
    \draw[arrow] (sum1) -- (attn1);
    \draw[] (sum1) -- (pe1);

    \node[input, above=1.5em of add4] (z) {$\mathbf{z}$};
    \draw[arrow] (add4) -- (z);

    % Skip connections
    \draw[arrow] (ff1.south)++(0, -0.6) -| ($(add4.west) + (-0.6,-0.5)$) |- (add4.west);
    \draw[arrow] (attn1.south)++(0, -0.6) -| ($(add1.west) + (-0.6,-0.5)$) |- (add1.west);

    % Q, K, V fan-out
    \draw[arrow] (attn1.south)++(0, -0.4) -| ($(attn1.south) + (-1,0)$);
    \draw[arrow] (attn1.south)++(0, -0.4) -| ($(attn1.south) + (1,0)$);

    \node[] at ($(pe1.west) + (-0.3,0)$) {$\mathbf{P}$};
    \draw[arrow] (iemb) -- (sum1);

    \node[anchor=east] at ($(encoder.west) + (-0.2,0)$) {$N\times$};
\end{tikzpicture}

\end{document}
```

---

## 7. NNTikZ: Decoder-Only Transformer

Source: https://github.com/fraserlove/nntikz (MIT License)

```latex
\documentclass{standalone}

\usepackage{tikz}

\usetikzlibrary{positioning, chains, shapes.geometric, fit, shapes, arrows.meta, calc, backgrounds}

\begin{document}

\begin{tikzpicture}[
    >=LaTeX,
    very thick,
    arrow/.style={-latex, very thick, rounded corners=0.2cm},
    block/.style={rectangle, fill=gray!10, rounded corners=3mm, draw, very thick},
    layer/.style={
        rectangle, fill=white!10, rounded corners=1mm,
        inner xsep=0em, inner ysep=0.25em,
        minimum height=1.4em, align=center,
        text width=2.5cm, draw, very thick
    },
    input/.style={circle, minimum width=2.25em, draw, fill=gray!10, thick},
    do path picture/.style={
        path picture={
          \pgfpointdiff{\pgfpointanchor{path picture bounding box}{south west}}
            {\pgfpointanchor{path picture bounding box}{north east}}
          \pgfgetlastxy\x\y
          \tikzset{x=\x/2,y=\y/2}
          #1
        }
    },
    sin wave/.style={do path picture={
        \draw [line cap=round] (-3/4,0)
        sin (-3/8,1/2) cos (0,0) sin (3/8,-1/2) cos (3/4,0);
    }}
]

    \node[input] (oemb) at (5.25,-0.45) {$\mathbf{y}$};

    \node[circle, draw, minimum size=0.25em, inner sep=0pt, above=1em of oemb] (sum2) {$\mathbf{+}$};
    \node [circle, draw, sin wave, minimum size=2em, right=0.8em of sum2] (pe2) {};

    \node[layer] (add3) at (5.25,3.83) {Add \& Norm};
    \node[layer] (attn3) at (5.25,2.78) {Masked \vspace{-0.05cm} \linebreak Multi-Head \vspace{-0.05cm} \linebreak Attention};
    \draw[] (attn3) -- (add3);
    \node[layer] (add5) at (5.25,6.38) {Add \& Norm};
    \node[layer] (ff2) at (5.25,5.53) {Feed \vspace{-0.05cm} \linebreak Forward};
    \draw[] (ff2) -- (add5);

    \coordinate (d1) at ($(attn3.south east) + (0.75,-0.7)$);
    \coordinate (d2) at ($(add5.north west) + (-0.15,0.05)$);
    \begin{scope}[on background layer]
        \node[block, fit=(d1) (d2)] (decoder) {};
    \end{scope}

    \node[layer, above=1.5em of add5] (linear) {Linear};
    \node[layer, above=1em of linear] (softmax) {Softmax};
    \node[input, above=1em of softmax] (probs) {$\hat{\mathbf{y}}$};

    \draw[arrow] (add3) -- (ff2);
    \draw[arrow] (add5) -- (linear);
    \draw[arrow] (linear) -- (softmax);
    \draw[arrow] (oemb) -- (sum2);
    \draw[arrow] (sum2) -- (attn3);
    \draw[] (sum2) -- (pe2);

    % Skip connections (right side, via east)
    \draw[arrow] (attn3.south)++(0, -0.6) -| ($(add3.east) + (0.6,-0.5)$) |- (add3.east);
    \draw[arrow] (ff2.south)++(0, -0.6) -| ($(add5.east) + (0.6,-0.5)$) |- (add5.east);

    % Q, K, V fan-out
    \draw[arrow] (attn3.south)++(0, -0.4) -| ($(attn3.south) + (-1,0)$);
    \draw[arrow] (attn3.south)++(0, -0.4) -| ($(attn3.south) + (1,0)$);

    \node[] at ($(oemb.east) + (1.4,0)$) {(shifted right)};
    \node[] at ($(pe2.east) + (0.3,0)$) {$\mathbf{P}$};

    \draw[arrow] (oemb) -- (sum2);
    \draw[arrow] (softmax) -- (probs);

    \node[anchor=west] at ($(decoder.east) + (0.2,0)$) {$N\times$};
\end{tikzpicture}

\end{document}
```

---

## 8. NNTikZ: LSTM Cell (Gate Operations)

Source: https://github.com/fraserlove/nntikz (MIT License)

Shows: sigma gates, tanh, element-wise multiply, element-wise add, pointwise styling.

```latex
\documentclass{standalone}

\usepackage{tikz}

\usetikzlibrary{positioning, chains, shapes.geometric, fit, shapes, arrows.meta, calc}

\begin{document}

\begin{tikzpicture}[
    >=LaTeX,
    cell/.style={
        rectangle, rounded corners=5mm, draw, very thick,
        minimum height=4cm, minimum width=6cm
    },
    input/.style={circle, minimum width=2.25em, draw, fill=gray!10, thick},
    hidden/.style={},
    func/.style={rectangle, draw, inner sep=2pt, minimum height=0.4cm},
    op/.style={circle, draw, inner sep=-0.5pt, minimum height=0.4cm},
    pointwise/.style={fill=gray!50},
    arrow/.style={-latex, very thick},
    arrowc1/.style={arrow, rounded corners=.25cm},
    arrowc2/.style={arrow, rounded corners=.5cm},
    backprop/.style={arrow, dashed, gray}
]

    % LSTM cell container
    \node [cell] at (0,0){};

    % Internal functions and operators
    \node [func] (sigf) at (-2,-0.75) {$\sigma$};
    \node [func] (sigi) at (-1.4,-0.75) {$\sigma$};
    \node [func] (tanhi) at (-0.5,-0.75) {$\tanh$};
    \node [func] (sigo) at (0.5,-0.75) {$\sigma$};
    \node [func, pointwise] (tanho) at (1.5,0.75) {$\tanh$};
    \node [op, pointwise] (mulf) at (-2,1.5) {$\times$};
    \node [op, pointwise] (add) at (-0.5,1.5) {+};
    \node [op, pointwise] (muli) at (-0.5,0) {$\times$};
    \node [op, pointwise] (mulo) at (1.5,0) {$\times$};

    % Inputs, outputs and hidden states
    \node[hidden] (ct-1) at (-4,1.5) {$\mathbf{c}_{t-1}$};
    \node[hidden] (ht-1) at (-4,-1.5) {$\mathbf{h}_{t-1}$};
    \node[input] (x) at (-2.5,-3) {$\mathbf{x}_{t}$};
    \node[hidden] (ct) at (4,1.5) {$\mathbf{c}_{t}$};
    \node[hidden] (ht) at (4,-1.5) {$\mathbf{h}_{t}$};
    \node[input] (y) at (2.5,3) {$\hat{\mathbf{y}}_{t}$};

    % Connections
    \draw [arrowc1, -] (x) -- (x |- ht-1) -| (tanhi);
    \draw [arrowc2, -] (ht-1) -| (sigo);
    \draw [arrowc1, -] (ht-1 -| sigf)++(-0.5,0) -| (sigf);
    \draw [arrowc1, -] (ht-1 -| sigi)++(-0.5,0) -| (sigi);
    \draw [arrowc1, -] (ht-1 -| tanhi)++(-0.5,0) -| (tanhi);
    \draw [arrow] (ct-1) -- (mulf) -- (add) -- (ct);

    % Gate operations
    \draw [arrow] (sigf) -- node[midway, left] {$\mathbf{f}_t$} (mulf);
    \draw [arrowc2] (sigi) |- node[midway, xshift=-1.5, yshift=1.5] {$\mathbf{i}_t$} (muli);
    \draw [arrow] (tanhi) -- node[midway, xshift=12, yshift=1.5] {$\tilde{\mathbf{c}}_t$} (muli);
    \draw [arrowc2] (sigo) |- node[midway, xshift=-1.5, yshift=1.5] {$\mathbf{o}_t$} (mulo);
    \draw [arrow] (muli) -- (add);
    \draw [arrowc1] (add -| tanho) ++(-0.5,0) -| (tanho);
    \draw [arrow] (tanho) -- (mulo);

    % Output
    \draw [arrowc2] (mulo) |- (ht);
    \draw (ct -| y) ++(0,-0.1) coordinate (i1);
    \draw [arrowc2, -] (ht -| y) ++(-0.5,0) -| (i1);
    \draw [arrow] (i1) ++(0,0.2) -- (y);
\end{tikzpicture}

\end{document}
```

---

## 9. NNTikZ: GRU Cell (Gate Operations)

Source: https://github.com/fraserlove/nntikz (MIT License)

```latex
\documentclass{standalone}

\usepackage{tikz}

\usetikzlibrary{positioning, chains, shapes.geometric, fit, shapes, arrows.meta, calc}

\begin{document}

\begin{tikzpicture}[
    >=LaTeX,
    cell/.style={
        rectangle, rounded corners=5mm, draw, very thick,
        minimum height=4cm, minimum width=6cm
    },
    input/.style={circle, minimum width=2.25em, draw, fill=gray!10, thick},
    hidden/.style={},
    func/.style={rectangle, draw, inner sep=2pt, minimum height=0.4cm},
    op/.style={circle, draw, inner sep=-0.5pt, minimum height=0.4cm},
    pointwise/.style={fill=gray!50},
    arrow/.style={-latex, very thick},
    arrowc1/.style={arrow, rounded corners=.25cm},
    arrowc2/.style={arrow, rounded corners=.5cm},
    backprop/.style={arrow, dashed, gray}
]

    \node [cell] at (0,0){};

    % Internal functions and operators
    \node [func] (sigr) at (-0.75,-0.75) {$\sigma$};
    \node [func] (sigz) at (0,-0.75) {$\sigma$};
    \node [func] (tanh) at (2,-0.75) {$\tanh$};
    \node [func, pointwise] (minus) at (0,0.5) {$1-$};
    \node [op, pointwise] (mulr) at (-1.5,0) {$\times$};
    \node [op, pointwise] (mulz) at (0,1.5) {$\times$};
    \node [op, pointwise] (add) at (2,1.5) {$+$};
    \node [op, pointwise] (mulh) at (2,0) {$\times$};

    % I/O
    \node[hidden] (ht-1) at (-4,1.5) {$\mathbf{h}_{t-1}$};
    \node[input] (x) at (-2.5,-3) {$\mathbf{x}_{t}$};
    \node[hidden] (ht) at (4,1.5) {$\mathbf{h}_{t}$};
    \node[input] (y) at (2.5,3) {$\hat{\mathbf{y}}_{t}$};

    % Connections
    \draw [arrowc1] (ht-1) -- (mulz) -- (add) -- (ht);
    \draw [arrowc1, -] (x) |- (-2.25, -1.4) ++(sigr);
    \draw [arrowc1, -] (x) |- (-2.25, -1.75) ++(sigr);
    \draw [arrowc1, -] (x -| sigz) ++(-1, 1.6) -| (sigz);
    \draw [arrowc1, -] (x -| sigr) ++(-1.5, 1.6) -| (sigr);
    \draw [arrowc1, -] (ht-1.east) -| (mulr);
    \draw [arrowc1, -] (ht-1) -| (-2.5,0) ++(x);
    \draw [arrowc1, -] (x) ++(0,3) |- (-1,-1.4) ++(tanh);
    \draw [arrowc2, -] (x -| tanh) ++(-4.25, 1.25) -| (tanh);

    % Gate operations
    \draw [arrowc1, -] (mulr) -- ++(0,-1.3);
    \draw [arrowc1, -] (mulr) ++(0,-1.55) |- (0,-1.75) ++(tanh);
    \draw [arrowc1] (sigr) |- node[midway, shift={(-2pt,4pt)}] {$\mathbf{r}_t$} (mulr.east);
    \draw [arrowc2] (sigz) |- (mulh);
    \draw [arrowc2] (tanh) -- node[midway, shift={(12pt,1.5pt)}] {$\tilde{\mathbf{h}}_t$} (mulh);
    \draw [arrowc2] (minus) -- (mulz);
    \draw [arrowc2] (sigz) -- node[midway, left] {$\mathbf{z}_t$} (minus);
    \draw [arrowc2] (mulh) -- (add);

    \draw [arrowc1] (add) coordinate[auto] -|(y);
\end{tikzpicture}

\end{document}
```

---

## 10. Petar Velickovic: Self-Attention

Source: https://github.com/PetarV-/TikZ (MIT License)

Shows: attention coefficients with variable opacity, element-wise multiply circles,
summation, function boxes on edges, white-overprint technique for crossing arrows.

```latex
\documentclass[crop, tikz]{standalone}
\usepackage{tikz}

\usetikzlibrary{positioning}

\begin{document}
\begin{tikzpicture}

    \node (X1) {$\vec{e}_{1}$};
    \node[rectangle, right= 0.5em of X1] (x_dots_1) {$\dots$};
    \node[right=0.5em of x_dots_1] (Xj) {$\vec{e}_{j}$};
    \node[rectangle, right= 1em of Xj] (x_dots_2) {$\dots$};
    \node[right=1em of x_dots_2] (Xn) {$\vec{e}_{n}$};

    % Attention function boxes
    \node[rectangle, draw, ultra thick, above=of X1] (attn1) {\large $a_\phi$};
    \node[rectangle, draw, ultra thick, above=of Xj] (attnj) {\large $a_\phi$};
    \node[rectangle, draw, ultra thick, above=of Xn] (attnn) {\large $a_\phi$};

    % Pairwise attention inputs
    \draw[-stealth, thick] (X1) -- (attn1);
    \draw[-stealth, thick] (Xj) -- (attn1);
    \draw[-stealth, thick] (Xj) -- (attnj);
    \draw[-stealth, thick] ([xshift=3em]Xj) -- (attnj);
    \draw[-stealth, thick] (Xj) -- (attnn);
    \draw[-stealth, thick] (Xn) -- (attnn);

    % Attention coefficients (opacity = attention weight)
    \node[above= of attn1, opacity=0.2] (alpha1j) {$\alpha_{1,j}$};
    \node[above= of attnj, opacity=1] (alphajj) {$\alpha_{j,j}$};
    \node[above= of attnn, opacity=0.6] (alphanj) {$\alpha_{n,j}$};

    % Element-wise multiply
    \node[circle, draw, above=of alpha1j] (times1) {$\times$};
    \node[circle, draw, above=of alphajj] (timesj) {$\times$};
    \node[circle, draw, above=of alphanj] (timesn) {$\times$};

    % Summation
    \node[rectangle, draw, above=of timesj] (sum) {$\Sigma$};
    \node[above=1em of sum] (x_tprim) {$\vec{e}_j'$};

    % White-overprint technique for clean arrow crossings
    \draw[-stealth, line width=1.5mm, white] (attn1) -- (alpha1j);
    \draw[-stealth, thick, opacity=0.2] (attn1) -- (alpha1j);
    \draw[-stealth, line width=1.5mm, white] (attnj) -- (alphajj);
    \draw[-stealth, thick, opacity=1] (attnj) -- (alphajj);
    \draw[-stealth, line width=1.5mm, white] (attnn) -- (alphanj);
    \draw[-stealth, thick, opacity=0.6] (attnn) -- (alphanj);

    % Feature transform on skip path (function box on edge)
    \draw[-stealth, white, line width=1.5mm] (X1) edge[bend right=30] (times1);
    \draw[-stealth, thick] (X1) edge[bend right=30] node[rectangle, draw, fill=white, midway] {$f_\psi$} (times1);
    \draw[-stealth, white, line width=1.5mm] (Xj) edge[bend right=30] (timesj);
    \draw[-stealth, thick] (Xj) edge[bend right=30] node[rectangle, draw, fill=white, midway] {$f_\psi$} (timesj);
    \draw[-stealth, thick] (Xn) edge[bend right=30] node[rectangle, draw, fill=white, midway] {$f_\psi$} (timesn);

    % Aggregate
    \draw[-, line width=1.5mm, white] (times1) -- (sum);
    \draw[-stealth, thick] (times1) -- (sum);
    \draw[-, line width=1.5mm, white] (timesj) -- (sum);
    \draw[-stealth, thick] (timesj) -- (sum);
    \draw[-stealth, thick] (timesn) -- (sum);
    \draw[-stealth, thick] (times1) -- (sum);

    % Weighted coefficients to multiply
    \draw[-stealth, line width=1.5mm, white] (alpha1j) -- (times1);
    \draw[-stealth, thick, opacity=0.2] (alpha1j) -- (times1);
    \draw[-stealth, line width=1.5mm, white] (alphajj) -- (timesj);
    \draw[-stealth, thick, opacity=1] (alphajj) -- (timesj);
    \draw[-stealth, line width=1.5mm, white] (alphanj) -- (timesn);
    \draw[-stealth, thick, opacity=0.6] (alphanj) -- (timesn);

    \draw[-stealth, thick] (sum) -- (x_tprim);

\end{tikzpicture}
\end{document}
```

---

## 11. Skip Connection / Residual Block

Source: https://tikz.janosh.dev + https://tikz.net/skip-connection/ (CC-BY-SA 4.0)

```latex
\documentclass[tikz]{standalone}

\usetikzlibrary{positioning, calc, decorations.pathreplacing}

\begin{document}
\begin{tikzpicture}

  \node[fill=orange!50] (l1) {layer 1};
  \node[blue!50!black, right=of l1, label={below:activation}] (act1) {$a(\vec x)$};
  \node[fill=teal!50, right=of act1] (l2) {layer 2};
  \node[right=of l2, font=\Large, label={below:add}, inner sep=0,
        pin={60:$\mathcal F(\vec x) + \vec x$}] (add) {$\oplus$};
  \node[blue!50!black, right=of add, label={below:activation}] (act2) {$a(\vec x)$};

  % Main path
  \draw[->] (l1) -- (act1);
  \draw[->] (act1) -- (l2);
  \draw[<-] (l1) -- ++(-2,0) node[below, pos=0.8] {$\vec x$};
  \draw[->] (l2) -- (act2) node[above, pos=0.8] {};

  % Skip connection (identity bypass)
  \draw[->] ($(l1)-(1.5,0)$) to[out=90, in=90]
    node[below=1ex, midway, align=center] {skip connection\\(identity)}
    node[above, midway] {$\vec x$} (add);

  % Brace showing F(x)
  \draw[decorate, decoration={brace, amplitude=1ex, raise=1cm}]
    (l2.east) -- node[midway, below=1.2cm] {$\mathcal F(\vec x)$} (l1.west);

\end{tikzpicture}
\end{document}
```

---

## 12. Common Patterns Reference

### Pattern: Circled Element-wise Operator

```latex
% Addition (used in residual connections)
\node[circle, draw, inner sep=0pt, minimum size=1.5em] (add) {$\oplus$};

% Multiplication (used in gating)
\node[circle, draw, inner sep=0pt, minimum size=1.5em] (mul) {$\otimes$};

% With pointwise fill (NNTikZ convention)
\node[circle, draw, fill=gray!50, inner sep=-0.5pt, minimum height=0.4cm] (mul) {$\times$};
```

### Pattern: Activation Functions in Boxes

```latex
% Sigmoid gate
\node[rectangle, draw, inner sep=2pt, minimum height=0.4cm] (sig) {$\sigma$};

% GELU (custom text)
\node[rectangle, draw, rounded corners=1mm, minimum height=1.4em,
      minimum width=2cm, align=center] (gelu) {GELU};

% Tanh
\node[rectangle, draw, inner sep=2pt, minimum height=0.4cm] (th) {$\tanh$};
```

### Pattern: Skip Connection (Vertical, as in Transformer)

```latex
% Input below the attention layer, skip to Add&Norm above
% This routes: down from attention input, left, up, right into Add&Norm
\draw[arrow] (attn.south)++(0, -0.6)
    -| ($(addnorm.west) + (-0.6,-0.5)$)
    |- (addnorm.west);
```

### Pattern: Skip Connection (Horizontal, as in ResNet)

```latex
% Bypass over multiple layers via top arc
\draw[->] ($(input)-(1.5,0)$) to[out=90, in=90]
    node[above, midway] {identity} (add_node);
```

### Pattern: Q/K/V Fan-out (Input splits to 3 destinations)

```latex
% Single input fans out to three sub-inputs of attention
\draw[arrow] (input.south)++(0, -0.4)
    -| ($(attn.south) + (-1,0)$);  % Q
\draw[arrow] (input.south)++(0, -0.4)
    -| ($(attn.south) + (0,0)$);   % K
\draw[arrow] (input.south)++(0, -0.4)
    -| ($(attn.south) + (1,0)$);   % V
```

### Pattern: Stacked/Repeated Blocks (Nx notation)

```latex
% Background block with fit
\begin{scope}[on background layer]
    \node[rectangle, fill=gray!10, rounded corners=3mm, draw, very thick,
          fit=(bottom_node) (top_node)] (block) {};
\end{scope}
\node[anchor=east] at ($(block.west) + (-0.2,0)$) {$N\times$};
```

### Pattern: Annotation Labels (Normalization, etc.)

```latex
% Small annotation above or beside a layer
\node[font=\scriptsize, above right=0.1em of layer.north east,
      text=gray] {SN};  % Spectral Normalization

% Or as a pin
\node[layer, pin={[pin distance=0.5cm, font=\scriptsize]right:LayerNorm}]
    (ln) {Linear};

% Or as a label
\node[layer, label={[font=\scriptsize, text=gray]above:SpectralNorm}]
    (sn_layer) {Conv 1x1};
```

### Pattern: Dashed Training-Only Path

```latex
% Dropout or auxiliary loss branch visible only during training
\draw[-latex, very thick, dashed, gray] (layer.east)
    -- ++(2,0) node[right, text=black] {aux loss};
```

### Pattern: White-Overprint for Arrow Crossings

```latex
% Draw a thick white line first, then the actual arrow on top
\draw[-, line width=1.5mm, white] (A) -- (B);
\draw[-stealth, thick] (A) -- (B);
```

### Pattern: Colored Layer Blocks by Type

```latex
\definecolor{attn_color}{HTML}{4ECDC4}    % attention: teal
\definecolor{ff_color}{HTML}{FF6B6B}      % feedforward: coral
\definecolor{norm_color}{HTML}{FFE66D}    % normalization: yellow
\definecolor{embed_color}{HTML}{95E1D3}   % embedding: mint

\tikzset{
    attn_layer/.style={layer, fill=attn_color!30},
    ff_layer/.style={layer, fill=ff_color!30},
    norm_layer/.style={layer, fill=norm_color!30},
}
```

### Pattern: Positional Encoding (Sine Wave in Circle)

```latex
% From NNTikZ: draws a small sine curve inside a circle
\node [circle, draw, sin wave, minimum size=2em] (pe) {};
\node[] at ($(pe.west) + (-0.3,0)$) {$\mathbf{P}$};
% Requires the "sin wave" and "do path picture" styles from section 2
```

---

## 13. PlotNeuralNet Style Primitives

Source: https://github.com/HarisIqbal88/PlotNeuralNet

PlotNeuralNet uses three .sty files defining TikZ `\pic` objects:

### init.tex (entry point)
```latex
\usetikzlibrary{quotes, arrows.meta, positioning}
% Imports: Ball.sty, Box.sty, RightBandedBox.sty
```

### Ball.sty
- Draws shaded spheres for element-wise/reduction operations
- Default logo: `$\Sigma$`
- Parameters: `radius`, `scale`, `fill`, `opacity`, `logo`, `caption`, `name`
- Anchors: `anchor`, `east`, `west`, `north`, `south`

### Box.sty
- Draws 3D rectangular blocks (conv layers, FC layers)
- Parameters: `width`, `height`, `depth`, `scale`, `fill`, `opacity`
- Labels: `xlabel`, `ylabel`, `zlabel`, `caption`
- 24+ named coordinates including cardinal + 3D variants (near/far/corners)

### RightBandedBox.sty
- 3D block with a colored right-side band (e.g., conv+BN)
- Dual fill: `fill` + `bandfill` with independent opacity
- Same coordinate system as Box

### Python API (generates TikZ)
```python
# Key functions in pycore/tikzeng.py:
to_Conv(name, s_filter, n_filter, offset, to, width, height, depth, caption)
to_Pool(name, offset, to, width, height, depth, opacity, caption)
to_SoftMax(name, s_filter, offset, to, width, height, depth, caption)
to_connection(of, to)
```

---

## 14. Sources

Primary repositories with complete source code:

- **NNTikZ** (MIT): https://github.com/fraserlove/nntikz
  - 17 diagrams: transformer, multihead_attention, dot_product_attention, encoder_only, decoder_only, lstm, gru, attention (Bahdanau), rnn, neural_network, neuron, dropout, and more

- **Petar Velickovic TikZ** (MIT): https://github.com/PetarV-/TikZ
  - Self-attention, GAT layer, LSTM, progressive neural network, and 20+ more

- **PlotNeuralNet**: https://github.com/HarisIqbal88/PlotNeuralNet
  - Python-to-TikZ generator for 3D CNN/architecture diagrams

- **transformer-verif**: https://github.com/pzeina/transformer-verif
  - Data-driven transformer diagram template with CSV-populated cells

- **Janosh TikZ / Diagrams**: https://github.com/janosh/diagrams
  - Skip connection, normalizing flow, and 200+ physics/ML/chem diagrams
  - Browse at: https://janosh.dev/diagrams

- **tikz.net**: https://tikz.net/neural_networks/
  - Neural network examples with downloadable .tex (CC-BY-SA 4.0)

- **David Stutz blog**: https://davidstutz.de/illustrating-convolutional-neural-networks-in-latex-with-tikz/
  - Color-coded layer style definitions, fading connections

- **neuralnetwork CTAN package**: https://github.com/battlesnake/neural
  - High-level `\inputlayer`, `\hiddenlayer`, `\outputlayer`, `\linklayers` commands
