---
paper: ziarelli_2025_ecg_forward
title: "Towards Deep Learning Surrogate for the Forward Problem in Electrocardiology: A Scalable Alternative to Physics-Based Models"
authors: "Ogbomo-Harmitt S, Magnetti C, Spota C, Grzelak J, Aslanidi O"
year: 2025
journal: "CinC 2025 (Computing in Cardiology)"
doi: "arxiv:2512.13765"
pmid: ""
pdf: ../papers/ziarelli_2025_ecg_forward.pdf
questions: [surrogate_pipeline]
---

> Filename retained as `ziarelli_2025_ecg_forward.md` for continuity with earlier search references; true authorship is Ogbomo-Harmitt et al.

## Key Findings
- **Deep learning replacement for the forward electrocardiology problem**: map cardiac voltage (V_m) propagation maps directly to body-surface ECG signals.
- **Mean R² = 0.99 ± 0.01** across healthy, fibrotic, and remodeled 2D tissue conditions. Per-lead error not specified in the abstract.
- **Time-dependent attention-based seq2seq architecture** with convolutional encoders for the spatial V_m maps and attention-based decoder for the temporal ECG leads.
- **Hybrid loss**: Huber + spectral entropy. Huber gives robust MSE-alternative; spectral entropy term pressures frequency-domain consistency, important for ECG waveform shape.
- **Critical meta-finding for our pivot**: the paper **does NOT learn φ_e as a field**. It maps V_m → body-surface ECG directly, treating the physics of extracellular potential as something to be short-circuited by learning, not something the model reconstructs. This is the status quo of the cardiac-ML field that our hybrid bidomain pivot breaks from.

## Method
- **Architecture**: convolutional encoder over V_m spatial maps → attention-based seq2seq → 12-lead ECG time series.
- **Training**: supervised on simulator-generated (V_m map → ECG) pairs across three tissue conditions (healthy, fibrotic, remodeled). 2D tissue simulations.
- **Loss**: Huber + spectral entropy.
- **Input**: cardiac voltage propagation maps (V_m(x, y, t)).
- **Output**: ECG time series (leads).
- **Accuracy**: R² = 0.99 ± 0.01 mean across conditions.

## Connections to Our Models

### Relevant Engine Components
**Zero direct architectural overlap** — they skip φ_e as a learned field and jump from V_m to ECG via lead-field-style integration baked into the network. Our hybrid bidomain surrogate does the opposite: we learn φ_e as a full-field intermediate and let the rest of the bidomain loop remain classical.

### Agreements
- **Supervised training from simulator output** for cardiac surrogate work is a viable recipe — at least for their forward-ECG scope.
- **Attention + conv encoder hybrid** architecture pattern is proven for spatiotemporal cardiac signals.
- **Huber + spectral loss**: spectral-domain terms may be worth considering for the dual-tower elliptic surrogate if autoregressive rollout produces frequency drift in φ_e (wavefront position artifacts).

### Disagreements or Gaps
- **They skip φ_e entirely**. The forward ECG problem classically goes `V_m → extracellular φ_e via bidomain elliptic → body-surface ECG via lead-field integral`. Their NN collapses both transforms into one mapping. This works for forward-ECG accuracy but **loses the spatial φ_e field**, which is exactly what we need for the bidomain rollout loop.
- **Output is 1D (12 ECG leads over time), not 2D spatial field**. Our elliptic surrogate must produce a 2D φ_e field. Different task.
- **Not autoregressive**: they predict ECG from V_m maps in one shot. Our surrogate must work inside the bidomain time loop where φ_e feeds back into V_m dynamics. Long-horizon stability story is different.
- **2D tissue only**: for forward ECG, 2D is a severe simplification (real torsos are 3D). For our bidomain-elliptic purposes, 2D is adequate (Bidomain V1 is 2D).
- **Monodomain-like input**: their V_m maps almost certainly come from monodomain simulations (paper doesn't discuss bidomain). Further evidence that bidomain φ_e is treated as an afterthought across the field.

### Actionable Insights
- **HIGH — Cite this paper in the motivation section.** It's the cleanest demonstration that the field takes V_m → ECG as a direct mapping and skips φ_e. Our pivot's value proposition ("nobody solves for φ_e as a learned field") is directly supported by this paper.
- **LOW — Huber + spectral-entropy hybrid loss**: potentially useful for the dual-tower surrogate if we find pure MSE overshoots on high-frequency φ_e gradient structures at wavefronts. Low priority, only revisit if we see such artifacts.
- **LOW — Architecture template**: attention-based seq2seq is *not* what our spatial-field surrogate should look like. Their architecture is scalarized (time series), ours is fully spatial.
- **LOW — Benchmark comparability**: their 2D tissue setup may overlap enough with our Bidomain V1 benchmarks to make a side-by-side possible, but the output spaces are too different to be a fair comparison.

## Limitations / Caveats
- **2D only**: severely limits applicability to real ECG inverse-problem scenarios, where torso geometry matters.
- **R² = 0.99 on lead waveforms** doesn't translate to our metric space (CV, APD, Kleber boundary ratio). No cross-comparable numbers for us.
- **Conference paper (CinC 2025)** — abstract and typical 4–6 page format; architecture details incomplete without fetching the PDF. Full method details may be under-specified.
- **Forward problem only, not inverse**: the paper's scope is V_m → ECG, not ECG → V_m. Our pivot is elliptic (V_m → φ_e → feedback), different direction entirely.
- **Author attribution correction**: earlier project references tagged this as "Ziarelli 2025." The actual authorship is **Ogbomo-Harmitt, Magnetti, Spota, Grzelak, Aslanidi** (King's College London group). The filename is kept consistent with prior references but the frontmatter is authoritative.
