# Rohr, Kucera, Fast & Kléber 1997 — Paradoxical Improvement by Partial Uncoupling

**Citation**: Rohr S, Kucera JP, Fast VG, Kléber AG. "Paradoxical improvement of impulse conduction in cardiac tissue by partial cellular uncoupling." *Science* 275(5301):841-844, 1997.
**PubMed**: 9012353

## Key Contribution
Demonstrated that **reducing gap junction coupling** can paradoxically **restore** conduction that was blocked at geometric expansions — a counterintuitive result with major implications for geometry-induced pacemaking.

## Experimental Setup
- **Geometry**: Cell strands (25-70 µm wide) expanding into large monolayers (2.2 × 2.2 mm)
- **Cells**: Neonatal rat heart cells, patterned cultures
- **Uncoupling agent**: Palmitoleic acid (partial gap junction blockade)
- **Measurement**: Optical mapping with voltage-sensitive dyes

## The Paradox
1. Under normal coupling: unidirectional conduction block at strand-to-expansion
2. After partial uncoupling: conduction **restored** through the same geometry

## Mechanism: Asymmetric Effect on Source vs Sink
- Reducing coupling in the large mass (sink) decreases its current demand **more** than it reduces the source's current supply
- The source (narrow strand) has few neighbors → less affected by uncoupling
- The sink (large mass) has many neighbors → greatly reduced collective current drain
- Net effect: improved source-to-sink current ratio

## Implications for Geometry-Induced Pacemaking
1. Coupling strength is not just a dial — its effect depends on geometry
2. In our simulations, we should expect a **non-monotonic** relationship between coupling (D) and pacemaking success
3. There may be an optimal coupling window: too strong → suppresses automaticity; too weak → can't drive tissue; intermediate → pacing works
4. This parallels the SAN design: low internal coupling (Cx45) is a feature, not a bug

## Relevance to Our Work
Critical for understanding parameter space. When sweeping diffusion coefficient D in our PHAS13 tissue simulations, we should look for this paradoxical regime.
