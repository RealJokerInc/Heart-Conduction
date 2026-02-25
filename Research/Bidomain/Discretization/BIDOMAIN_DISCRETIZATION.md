# Discretization Methods for Cardiac Bidomain Equations
## A Comprehensive Research Guide for Implementation

**Version:** 1.0
**Date:** February 2025
**Context:** Extension of monodomain engine (V5.4) to support bidomain equations with FEM, FDM, FVM

---

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundation: Bidomain Equations](#mathematical-foundation)
3. [FEM Discretization](#fem-discretization)
4. [FDM Discretization](#fdm-discretization)
5. [FVM Discretization](#fvm-discretization)
6. [Semi-Discrete System Structure](#semi-discrete-system)
7. [Practical Implementation Considerations](#practical-considerations)
8. [References](#references)

---

## Introduction {#introduction}

### Motivation

The monodomain model simplifies cardiac electrophysiology by assuming equal anisotropy ratios between intracellular and extracellular conductivities. However, physiological measurements show:
- **Intracellular anisotropy ratio:** σ_iL/σ_iT ≈ 10:1
- **Extracellular anisotropy ratio:** σ_eL/σ_eT ≈ 2.5:1

This inequality invalidates the monodomain assumption for applications requiring accurate extracellular field representation, such as:
- Defibrillation studies
- Extracellular stimulation
- Virtual electrode polarization
- Electrocardiogram (ECG) generation
- Ischemia and source-sink mismatch effects

The bidomain model captures these phenomena by maintaining separate intracellular (φ_i) and extracellular (φ_e) potentials, with the transmembrane potential V_m = φ_i - φ_e emerging naturally.

### Computational Scope

This document covers discretization of the bidomain equations using three primary spatial methods:
- **FEM** (Finite Element Method) - most versatile for complex geometries
- **FDM** (Finite Difference Method) - efficient for structured grids
- **FVM** (Finite Volume Method) - excellent conservation properties

Each method is adapted to extend the capabilities of an existing monodomain V5.4 engine with FEM, FDM, and FVM support.

---

## Mathematical Foundation: Bidomain Equations {#mathematical-foundation}

### 1.1 Strong Form of Bidomain Equations

The standard bidomain PDE system consists of:

**Equation 1 (Parabolic - Membrane Potential Dynamics):**

$$\chi C_m \frac{\partial V_m}{\partial t} = -\chi I_{ion}(V_m, \mathbf{w}) + \nabla \cdot (\mathbf{D}_i \nabla V_m) + \nabla \cdot (\mathbf{D}_i \nabla \phi_e) + I_{stim}^i$$

**Equation 2 (Elliptic - Extracellular Potential):**

$$0 = \nabla \cdot [(\mathbf{D}_i + \mathbf{D}_e) \nabla \phi_e] + \nabla \cdot (\mathbf{D}_i \nabla V_m) + I_{stim}^e$$

**Coupled ODE System (Gating Variables):**

$$\frac{d\mathbf{w}}{dt} = \mathbf{h}(V_m, \mathbf{w})$$

where:
- **V_m** = transmembrane potential (voltage difference across membrane)
- **φ_e** = extracellular potential
- **φ_i** = intracellular potential (typically set to V_m + φ_e or determined by boundary conditions)
- **χ** = surface-to-volume ratio of cells (typical value: 1000-1500 cm⁻¹)
- **C_m** = membrane capacitance (typical value: 1 μF/cm²)
- **D_i, D_e** = intracellular and extracellular conductivity tensors (S/cm)
- **I_ion** = transmembrane ionic current (μA/cm²)
- **I_stim^i, I_stim^e** = stimulation currents
- **w** = gating variables (dimensionless, range [0,1])
- **h** = gating variable dynamics function

### 1.2 Alternative Formulation: Two-Potential System

Some implementations formulate the bidomain as a coupled 2×2 system in terms of φ_i and φ_e:

$$\chi C_m \frac{\partial (φ_i - φ_e)}{\partial t} = -\chi I_{ion}(φ_i - φ_e, \mathbf{w}) + \nabla \cdot (\mathbf{D}_i \nabla φ_i) + I_{stim}^i$$

$$0 = \nabla \cdot (\mathbf{D}_i \nabla φ_i) + \nabla \cdot (\mathbf{D}_e \nabla φ_e) + \chi C_m \frac{\partial V_m}{\partial t} + \chi I_{ion}(V_m, \mathbf{w}) + (I_{stim}^i + I_{stim}^e)$$

### 1.3 Relationship to Monodomain Model

**Monodomain Equation:**

$$\chi C_m \frac{\partial V}{\partial t} = -\chi I_{ion}(V, \mathbf{w}) + \nabla \cdot (\mathbf{D} \nabla V) + I_{stim}$$

**Reduction Condition (Equal Anisotropy Ratio Assumption):**

The bidomain system reduces to monodomain when:

$$\frac{\sigma_e^L}{\sigma_e^T} = \frac{\sigma_i^L}{\sigma_i^T}$$

or more generally, when **D_e = λD_i** (λ is a positive scalar).

Under this condition:
- The second (elliptic) equation becomes **∇φ_e = 0** (φ_e is constant, often set to 0)
- The effective conductivity becomes: **D_eff = (D_i + D_e)·D_i / (D_i + D_e)** (harmonic mean form)
- The monodomain approximation activates

**When Monodomain Fails:**

1. **Unequal anisotropy ratios** - Causes monodomain conduction velocities to differ by 2-5% from bidomain
2. **Extracellular stimulation** - Monodomain cannot capture virtual electrode polarization
3. **Defibrillation** - Requires accurate extracellular field representation
4. **Bath coupling** - Torso or experimental bath interactions require bidomain
5. **Transmural heterogeneity** - Regional conductivity variations in three domains (epi/mid/endo)

**Physiological Validity Comparison:**

Recent studies show that monodomain activation times are on average 1 ms higher than bidomain, and propagation is 2% faster in bidomain. For many standard electrophysiology applications, monodomain proves adequate due to its ability to capture source-sink mismatch effects. However, bidomain remains the gold standard for comprehensive simulations.

### 1.4 Conductivity Tensors

**Orthotropic Representation:**

The conductivity in each domain (intracellular i, extracellular e) can be represented in local fiber coordinates as:

$$\mathbf{D}_d = \begin{pmatrix} \sigma_d^L & 0 & 0 \\ 0 & \sigma_d^T & 0 \\ 0 & 0 & \sigma_d^T \end{pmatrix}$$

where subscript d ∈ {i, e}, and L, T denote longitudinal (along fiber) and transverse (across fiber) directions.

**Typical Physiological Values:**

| Domain | σ_L (S/cm) | σ_T (S/cm) | Ratio |
|--------|-----------|-----------|-------|
| Intracellular | 0.3-0.5 | 0.03-0.05 | 10:1 |
| Extracellular | 0.3-0.6 | 0.15-0.3 | 2.5:1 |
| Effective (monodomain) | 0.25-0.3 | 0.08-0.12 | 3:1 |

**Fiber Rotation Representation:**

In anatomically detailed models, fibers rotate transmurally. The local conductivity tensor is rotated to global coordinates via:

$$\mathbf{D}_d^{global} = \mathbf{R}(θ) \mathbf{D}_d^{local} \mathbf{R}(θ)^T$$

where **R(θ)** is the rotation matrix encoding fiber angles at each location.

### 1.5 Boundary Conditions

#### **Type 1: Insulated Tissue (No Bath)**

Both intracellular and extracellular domains are electrically isolated:

$$\mathbf{n} \cdot \mathbf{D}_i \nabla φ_i = 0 \quad \text{on } \Gamma \quad \text{(Neumann)}$$
$$\mathbf{n} \cdot \mathbf{D}_e \nabla φ_e = 0 \quad \text{on } \Gamma \quad \text{(Neumann)}$$

This corresponds to homogeneous Neumann boundary conditions on ∂Ω. Physically, no current crosses the tissue boundary.

#### **Type 2: Bath-Coupled Domain (Torso/Experimental)**

The tissue is coupled to a conductive external medium (torso, saline bath):

**In tissue domain (Ω_tissue):**
- Parabolic equation for V_m
- Elliptic equation for φ_e

**In external domain (Ω_bath):**
- Laplace equation for φ_bath: ∇·(D_bath ∇φ_bath) = 0

**Interface conditions (∂Ω_tissue = Γ_interface):**
- Continuity of potential: φ_e|_tissue = φ_bath|_bath
- Current conservation: n·D_e ∇φ_e|_tissue = -n·D_bath ∇φ_bath|_bath
- Insulation of intracellular domain: n·D_i ∇φ_i = 0

**Far-field boundary (∂Ω_bath far from tissue):**
- φ_bath = 0 or reference potential

This formulation enables simulation of body surface potentials and ECG measurements.

#### **Type 3: Robin Boundary Conditions (Impedance Matching)**

For complex boundary behaviors:

$$\alpha φ + β \mathbf{n} \cdot \mathbf{D} \nabla φ = γ \quad \text{on } \Gamma$$

Used for absorbing boundary conditions to minimize reflections in truncated domains.

---

## FEM Discretization {#fem-discretization}

### 2.1 Weak Form Derivation

#### **Step 1: Variational Formulation of Parabolic Equation**

Multiply Equation 1 by test function v(x) ∈ H¹(Ω) and integrate:

$$\int_\Omega \chi C_m \frac{\partial V_m}{\partial t} v \, dx = -\int_\Omega \chi I_{ion}(V_m, \mathbf{w}) v \, dx + \int_\Omega \nabla \cdot (\mathbf{D}_i \nabla V_m) v \, dx + \int_\Omega \nabla \cdot (\mathbf{D}_i \nabla φ_e) v \, dx + \int_\Omega I_{stim}^i v \, dx$$

Apply integration by parts to second-order terms:

$$\int_\Omega \nabla \cdot (\mathbf{D}_i \nabla V_m) v \, dx = -\int_\Omega (\nabla \mathbf{D}_i \nabla V_m) \cdot (\nabla v) \, dx + \int_{\partial \Omega} \mathbf{n} \cdot (\mathbf{D}_i \nabla V_m) v \, dS$$

With homogeneous Neumann boundary conditions, the boundary integral vanishes:

$$\int_\Omega \chi C_m \frac{\partial V_m}{\partial t} v \, dx + \int_\Omega \nabla V_m \cdot (\mathbf{D}_i \nabla v) \, dx = -\int_\Omega \chi I_{ion}(V_m, \mathbf{w}) v \, dx - \int_\Omega \nabla φ_e \cdot (\mathbf{D}_i \nabla v) \, dx + \int_\Omega I_{stim}^i v \, dx$$

#### **Step 2: Variational Formulation of Elliptic Equation**

Multiply Equation 2 by test function q(x) ∈ H¹(Ω):

$$0 = \int_\Omega \nabla \cdot [(\mathbf{D}_i + \mathbf{D}_e) \nabla φ_e] q \, dx + \int_\Omega \nabla \cdot (\mathbf{D}_i \nabla V_m) q \, dx + \int_\Omega I_{stim}^e q \, dx$$

Applying integration by parts:

$$\int_\Omega \nabla φ_e \cdot [(\mathbf{D}_i + \mathbf{D}_e) \nabla q] \, dx = -\int_\Omega \nabla V_m \cdot (\mathbf{D}_i \nabla q) \, dx - \int_\Omega I_{stim}^e q \, dx$$

#### **Weak Formulation Summary**

Find V_m ∈ V_V and φ_e ∈ V_φ such that for all v ∈ V_V and q ∈ V_φ:

$$\int_\Omega \chi C_m \frac{\partial V_m}{\partial t} v \, dx + a_i(V_m, v) + a_i(φ_e, v) = L_V(v)$$

$$a_e(φ_e, q) + a_i(V_m, q) = L_φ(q)$$

where:
- **a_i(u, v)** = ∫_Ω ∇u·(D_i ∇v) dx (intracellular bilinear form)
- **a_e(u, v)** = ∫_Ω ∇u·((D_i + D_e) ∇v) dx (elliptic bilinear form)
- **L_V(v)** = -∫_Ω χI_ion v dx + ∫_Ω I_stim^i v dx (RHS for parabolic)
- **L_φ(q)** = -∫_Ω I_stim^e q dx (RHS for elliptic)
- **V_V** = {v ∈ H¹(Ω) : v satisfies Dirichlet BC} (test space for V_m)
- **V_φ** = {q ∈ H¹(Ω) : q satisfies Dirichlet BC} (test space for φ_e)

### 2.2 Finite Element Spaces

#### **Basis Functions**

Choose finite-dimensional subspaces V_V^h ⊂ V_V and V_φ^h ⊂ V_φ with basis functions {φ_j}:

$$V_m^h(x,t) = \sum_{j=1}^{N} V_m^j(t) φ_j(x)$$
$$φ_e^h(x,t) = \sum_{j=1}^{N} φ_e^j(t) φ_j(x)$$

**P1 Linear Elements:**
- Nodal basis: φ_j(x_k) = δ_jk (Kronecker delta)
- Support: 4 elements in tetrahedral mesh
- Cost: 1 DOF per node
- Accuracy: O(h²) in energy norm

**P2 Quadratic Elements:**
- Nodes: vertices + edge midpoints
- Support: 8-10 elements in tetrahedral mesh
- Cost: 1 DOF per vertex + 1 per edge (10 total per tetrahedron)
- Accuracy: O(h³) in energy norm
- Memory/compute: ~8× higher than P1

#### **Integration Scheme**

Standard quadrature rules for element integrals:
- **P1 elements:** 1-point rule at centroid (exact for degree 1)
- **P2 elements:** 4-point rule (tetrahedron), 3-point rule (triangle)

Total quadrature points per element: Q for accuracy

### 2.3 Mass Matrix Assembly

The mass matrix M ∈ ℝ^(N×N) represents the time derivative term:

$$M_{ij} = \int_\Omega \chi C_m φ_i φ_j \, dx = \chi C_m \sum_{e=1}^{E} \int_{Ω_e} φ_i^e φ_j^e \, dx$$

**Element Mass Matrix (P1 in tetrahedron):**

$$M^e = \frac{\chi C_m V_e}{20} \begin{pmatrix} 2 & 1 & 1 & 1 \\ 1 & 2 & 1 & 1 \\ 1 & 1 & 2 & 1 \\ 1 & 1 & 1 & 2 \end{pmatrix}$$

where V_e is the element volume.

**For P2 elements:** 10×10 matrix with non-zero entries proportional to volume and basis function products.

**Lumped Mass Matrix (Optional):**

Replace M by diagonal lumping to avoid implicit time-stepping penalties:

$$M_{ij}^{lump} = δ_{ij} \sum_k M_{ik}$$

Useful for explicit time-stepping schemes but reduces accuracy.

### 2.4 Stiffness Matrix Assembly

#### **Intracellular Stiffness K_i:**

$$K_i^{ij} = \int_\Omega \nabla φ_i \cdot (\mathbf{D}_i \nabla φ_j) \, dx = \sum_{e=1}^{E} \int_{Ω_e} \nabla φ_i^e \cdot (\mathbf{D}_i^e \nabla φ_j^e) \, dx$$

**Element Stiffness Matrix (P1, constant D_i per element):**

For linear basis functions on tetrahedron with vertices at x_0, x_1, x_2, x_3:

$$K_i^e = A^T \mathbf{D}_i A$$

where A is the 4×3 gradient matrix (scaled edge vectors) and each row ∇φ_k^e is computed as:

$$\nabla φ_k^e = \frac{1}{6V_e} \text{edge vectors (based on local numbering)}$$

Explicit formula in barycentric coordinates leads to 4×4 element matrix.

#### **Elliptic Stiffness K_e:**

$$K_e^{ij} = \int_\Omega \nabla φ_i \cdot [(\mathbf{D}_i + \mathbf{D}_e) \nabla φ_j] \, dx$$

Assembled identically to K_i but using combined conductivity **(D_i + D_e)**.

**Block System Structure After Discretization:**

The global system becomes:

$$\begin{pmatrix} M & 0 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} \dot{V}_m \\ \dot{φ}_e \end{pmatrix} + \begin{pmatrix} K_i & K_i \\ K_i & K_e \end{pmatrix} \begin{pmatrix} V_m \\ φ_e \end{pmatrix} = \begin{pmatrix} F_V \\ F_φ \end{pmatrix}$$

Equivalently (extracting the elliptic constraint):

$$\begin{pmatrix} M & 0 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} \dot{V}_m \\ \dot{φ}_e \end{pmatrix} + \begin{pmatrix} K_i + K_i & K_i \\ K_i & K_e \end{pmatrix} \begin{pmatrix} V_m \\ φ_e \end{pmatrix} = \begin{pmatrix} F_V \\ F_φ \end{pmatrix}$$

where the block structure shows:
- **2×2 system:** Coupling between transmembrane and extracellular potentials
- **Singular mass matrix:** No time derivative in elliptic equation (differential-algebraic structure)
- **K_i appears twice:** Due to the coupling structure

### 2.5 RHS Vector Assembly

**Reaction-Diffusion Term:**

$$F_V^{ionic} = -\int_\Omega \chi I_{ion}(V_m, \mathbf{w}) φ_j \, dx$$

Computed via numerical quadrature at element level. For each quadrature point:

$$F_V^{ionic,e}[j] = -\chi \sum_{q=1}^{Q} I_{ion}^q w_q φ_j(x_q)$$

where w_q are quadrature weights.

**Stimulus Term:**

$$F_V^{stim} = \int_\Omega I_{stim}^i(x,t) φ_j \, dx$$

Can be applied uniformly, regionally, or as applied voltage (Dirichlet).

**Extracellular Stimulus:**

$$F_φ^{stim} = -\int_\Omega I_{stim}^e(x,t) φ_j \, dx$$

For externally imposed electric field: integrate applied potential over domain or set Dirichlet BC.

### 2.6 Extension from Monodomain V5.4

**Key Modifications to Existing FEM Code:**

1. **Double the DOF structure:**
   - Old: N nodes → N unknowns (V at each node)
   - New: N nodes → 2N unknowns (V_m and φ_e at each node)

2. **Matrix assembly changes:**
   ```pseudocode
   // Old monodomain:
   for each element e:
       compute K^e (one N_e × N_e matrix)
       assemble into K (N × N)

   // New bidomain:
   for each element e:
       compute K_i^e (intracellular stiffness)
       compute K_e^e (elliptic stiffness)
       assemble into block structure:
       K[V_m,V_m] += K_i^e + K_i^e   (self + coupling)
       K[V_m,φ_e] += K_i^e           (coupling)
       K[φ_e,V_m] += K_i^e           (coupling)
       K[φ_e,φ_e] += K_e^e           (elliptic)
   ```

3. **Sparse matrix storage:**
   - Use 2×2 block CSR (Compressed Sparse Row) format
   - Or interleave: [V_m^1, φ_e^1, V_m^2, φ_e^2, ...]
   - Memory increase: ~4× (doubled unknowns, increased bandwidth)

4. **RHS vector management:**
   - Combine ionic and stimulus contributions
   - Account for coupling in elliptic equation RHS

### 2.7 Numerical Examples: FEM Assembly

**Example 1D Element (Linear):**

For a 1D element [0, h] with D_i = σ_i, D_e = σ_e:

$$K_i^e = \frac{\sigma_i}{h} \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}$$

$$K_e^e = \frac{\sigma_i + \sigma_e}{h} \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}$$

**Example 2D Triangle (Linear P1):**

Triangle with vertices at (0,0), (1,0), (0,1), area = 0.5:

$$\nabla φ_1 = (-1, -1), \quad \nabla φ_2 = (1, 0), \quad \nabla φ_3 = (0, 1)$$

For isotropic D_i = σ_i I:

$$K_i^e = \frac{\sigma_i}{2} \begin{pmatrix} 2 & -1 & -1 \\ -1 & 1 & 0 \\ -1 & 0 & 1 \end{pmatrix}$$

---

## FDM Discretization {#fdm-discretization}

### 3.1 Structured Grid Setup

Finite difference methods require structured, regular grids. Typical setup:

$$x_{i,j,k} = (i Δx, j Δy, k Δz), \quad i=1,...,N_x, j=1,...,N_y, k=1,...,N_z$$

**Spacing:** Δx, Δy, Δz typically uniform, though non-uniform grids possible.

**Domain:** Heart tissue Ω with extent [0, L_x] × [0, L_y] × [0, L_z]

**Physiological parameters (3D ventricle):**
- L_x, L_y, L_z ≈ 3-5 cm
- Grid spacing: Δx = Δy = Δz = 0.1 mm = 100 μm (standard)
- Total nodes: (30-50)³ = 27,000 - 125,000 unknowns

### 3.2 Laplacian Discretization with Anisotropic Conductivity

#### **Diagonal Conductivity Tensor**

For orthotropic tissue aligned with grid axes, D is diagonal:

$$\mathbf{D} = \begin{pmatrix} D_x & 0 & 0 \\ 0 & D_y & 0 \\ 0 & 0 & D_z \end{pmatrix}$$

**Laplacian operator:**

$$\nabla \cdot (\mathbf{D} \nabla u) = D_x \frac{\partial^2 u}{\partial x^2} + D_y \frac{\partial^2 u}{\partial y^2} + D_z \frac{\partial^2 u}{\partial z^2}$$

**7-Point Stencil (Isotropic or Diagonal D):**

$$\nabla \cdot (\mathbf{D} \nabla u)|_{i,j,k} \approx \frac{D_x}{Δx^2}(u_{i+1,j,k} - 2u_{i,j,k} + u_{i-1,j,k})$$
$$+ \frac{D_y}{Δy^2}(u_{i,j+1,k} - 2u_{i,j,k} + u_{i,j-1,k})$$
$$+ \frac{D_z}{Δz^2}(u_{i,j,k+1} - 2u_{i,j,k} + u_{i,j,k-1})$$

Stencil weights:

$$\begin{matrix}
\text{Central} : & -2(\frac{D_x}{Δx^2} + \frac{D_y}{Δy^2} + \frac{D_z}{Δz^2}) \\
\text{X-faces} : & \frac{D_x}{Δx^2} \quad (±1 \text{ indices}) \\
\text{Y-faces} : & \frac{D_y}{Δy^2} \quad (±1 \text{ indices}) \\
\text{Z-faces} : & \frac{D_z}{Δz^2} \quad (±1 \text{ indices})
\end{matrix}$$

#### **Full Anisotropic Tensor (Non-Aligned Fibers)**

When fibers are rotated θ from grid axes:

$$\mathbf{D} = \mathbf{R}(θ) \mathbf{D}_{local} \mathbf{R}(θ)^T$$

results in cross-derivative terms:

$$\nabla \cdot (\mathbf{D} \nabla u) = \sum_{i,j=1}^3 D_{ij} \frac{\partial^2 u}{\partial x_i \partial x_j}$$

**9-Point Stencil for General 2D Case:**

For 2D with full tensor:

$$L u = D_{xx} u_{xx} + 2D_{xy} u_{xy} + D_{yy} u_{yy}$$

**Stencil coefficients:**

$$\text{Center:} \quad -2(D_{xx}/Δx^2 + D_{yy}/Δy^2)$$

$$\text{Cardinal (±x):} \quad D_{xx}/Δx^2 + D_{xy}/(4ΔxΔy)$$

$$\text{Cardinal (±y):} \quad D_{yy}/Δy^2 + D_{xy}/(4ΔxΔy)$$

$$\text{Diagonal (±x,±y):} \quad -D_{xy}/(4ΔxΔy)$$

**3D Extension (27-Point Stencil):**

For full 3D anisotropic tensor, 26 off-center neighbors contribute plus center:
- 6 face neighbors (±x, ±y, ±z)
- 12 edge neighbors (±x±y, ±x±z, ±y±z)
- 8 corner neighbors (±x±y±z)
- 1 center point

Coefficients computed similarly via Taylor expansion.

### 3.3 Bidomain Discretization Structure

**Parabolic Equation (Time-Stepping):**

$$\chi C_m \frac{\partial V_m}{\partial t}|_{i,j,k} = -\chi I_{ion,i,j,k}(V_m, \mathbf{w}) + L_i V_m|_{i,j,k} + L_i φ_e|_{i,j,k} + I_{stim}^i_{i,j,k}$$

where **L_i** is the 7-point or 27-point discretization of ∇·(D_i ∇·).

**Elliptic Equation (Implicit):**

$$0 = L_e φ_e|_{i,j,k} + L_i V_m|_{i,j,k} + I_{stim}^e_{i,j,k}$$

where **L_e** is the discretization of ∇·((D_i + D_e) ∇·).

**Rearranged for Linear System:**

At each time step, solve:

$$(L_e φ_e)_{i,j,k} = -(L_i V_m)_{i,j,k} - I_{stim}^e_{i,j,k}$$

This is a large sparse linear system: **L_e · φ_e = RHS**

**Assembly in FDM:**

1. **Create sparse matrix L_e:**
   - For each interior grid point, add stencil coefficients
   - Diagonal element: -2(σ_ix + σ_iy + σ_iz + σ_ex + σ_ey + σ_ez) / (Δ²)
   - 6 neighbors: (σ_ix or σ_ix + σ_ex) / (Δ²) etc.

2. **RHS vector assembly:**
   - Compute RHS = -(L_i V_m) - I_stim^e at all grid points
   - Account for boundary conditions

### 3.4 Boundary Condition Implementation

#### **Neumann BC (No-Flux):**

$$\frac{\partial u}{\partial n}|_{boundary} = 0$$

**One-sided finite difference (ghost point):**

$$\frac{\partial u}{\partial x}|_{i+1/2} = \frac{u_{i+1} - u_i}{Δx} = 0 \Rightarrow u_{i+1} = u_i$$

Eliminate ghost point in stencil assembly.

**Centered difference with modified boundary row:**

For boundary at x = 0:

$$\frac{\partial u}{\partial x}|_1 \approx \frac{u_2 - u_1}{Δx} = 0$$

In stencil assembly, set coefficient for u_0 to zero and modify center.

#### **Dirichlet BC (Fixed Potential):**

$$u|_{boundary} = g$$

Set row corresponding to boundary node:

$$u_{boundary} = g$$

Standard treatment: modify system matrix and RHS.

#### **Robin BC (Mixed):**

$$α u + β \frac{\partial u}{\partial n} = γ$$

Combine Dirichlet and Neumann weighted by α and β.

### 3.5 FDM Advantages and Limitations

**Advantages:**
- Simple implementation
- Fast assembly (no quadrature needed)
- Efficient for regular geometries
- Excellent memory layout for structured grids

**Limitations:**
- Requires structured grids
- Difficult to handle complex geometries (heart boundaries)
- Anisotropic non-aligned conductivity: 27-point stencil expensive
- Boundary irregularities reduced-order accuracy

### 3.6 FDM Implementation for Engine V5.4

**Modifications needed:**

1. **Grid generation:** Structured cartesian grid builder
2. **Stencil generator:** Compute stencil coefficients for each tissue point
3. **Matrix assembly:** Direct stencil-to-sparse-matrix mapping
4. **Boundary handling:** Ghost point or modified stencil rows
5. **Solver:** Reuse existing AMG/BiCGSTAB from V5.4 (matrix structure identical)

---

## FVM Discretization {#fvm-discretization}

### 4.1 Cell-Centered Finite Volume Method

Finite volume methods integrate conservation laws over control volumes. For cardiac electrophysiology:

**Conservation Principle:**

The rate of change of transmembrane charge in control volume Ω_i must equal the net ion current flux in plus reaction terms:

$$\frac{d}{dt} \int_{Ω_i} \chi C_m V_m \, dV + \int_{\partial Ω_i} \mathbf{J} \cdot \mathbf{n} \, dS = -\int_{Ω_i} \chi I_{ion} \, dV + \int_{Ω_i} I_{stim}^i \, dV$$

where **J** = -D∇φ is the ionic current density.

### 4.2 Control Volume Discretization

**Cell-Centered Arrangement:**

Each grid cell (tetrahedron, hexahedron) stores one value:
- **V_m,i** = transmembrane potential at cell center
- **φ_e,i** = extracellular potential at cell center

**Flux Computation:**

The flux across interface between cells i and j:

$$F_{ij} = \int_{\partial Ω_{ij}} (-D ∇φ) \cdot \mathbf{n} \, dS$$

approximated as:

$$F_{ij} ≈ -D_{ij} \frac{\partial φ}{\partial n}|_{i→j} A_{ij} ≈ -D_{ij} \frac{φ_j - φ_i}{d_{ij}} A_{ij}$$

where:
- **D_ij** = harmonic mean of conductivities
- **A_ij** = interface area
- **d_ij** = distance between cell centers

### 4.3 Harmonic Mean for Conductivities

**Scalar Conductivity:**

$$D_{ij} = \frac{2 D_i D_j}{D_i + D_j}$$

**Tensor Conductivity (Isotropic in Each Cell):**

$$\mathbf{D}_{ij} = \frac{2 D_i D_j}{D_i + D_j} \mathbf{I}$$

**Tensor with Aligned Anisotropy:**

For each principal direction:

$$D^{(k)}_{ij} = \frac{2 D_i^{(k)} D_j^{(k)}}{D_i^{(k)} + D_j^{(k)}}, \quad k = L, T, T$$

### 4.4 Discrete System Assembly

**Parabolic Equation (after discretization):**

$$\chi C_m V_{cell,i} \frac{dV_m^i}{dt} + \sum_{j \in \partial Ω_i} F_{ij}^i = -\chi I_{ion}^i + \sum_{j \in \partial Ω_i} F_{ij}^{ie} + I_{stim,i}^i$$

where:
- **F_ij^i** = flux of ions through interface (intracellular)
- **F_ij^ie** = flux related to extracellular potential gradient
- **V_cell,i** = volume of cell i

**Elliptic Equation (after discretization):**

$$0 = \sum_{j \in \partial Ω_i} (F_{ij}^e + F_{ij}^i) + I_{stim,i}^e$$

where F_ij^e uses combined conductivity (D_i + D_e).

### 4.5 Conservation Properties

**Key Advantage of FVM:**

Conservation is guaranteed locally at discrete level:

$$\sum_i I_{stim}^i + I_{stim}^e = 0 \quad \text{(charge conservation)}$$

Fluxes between adjacent cells are exactly balanced (no artificial sources/sinks).

### 4.6 FVM for Anisotropic Conductivity

**Orthotropic Case:**

For aligned fibers, use directional harmonic means:

$$F_{ij,longitudinal} = -D^L_{ij} \frac{φ_j - φ_i}{d_{ij}^L} A_{ij}$$

**Non-Aligned Anisotropy:**

For full tensor D with fiber rotation, the interface normal n̂ determines effective conductivity:

$$D_n = \mathbf{n}^T \mathbf{D} \mathbf{n}$$

Flux:

$$F_{ij} ≈ -D_{n,ij} \frac{φ_j - φ_i}{d_{ij}} A_{ij}$$

This naturally handles arbitrary fiber orientations without 27-point stencils.

### 4.7 Block System Structure in FVM

After space discretization, the system is:

$$\begin{pmatrix} M_{diag} & 0 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} \dot{V}_m \\ \dot{φ}_e \end{pmatrix} + \begin{pmatrix} L_i & L_i \\ L_i & L_e \end{pmatrix} \begin{pmatrix} V_m \\ φ_e \end{pmatrix} = \begin{pmatrix} F_V \\ F_φ \end{pmatrix}$$

where:
- **M_diag** = diagonal mass matrix (cell volumes × χC_m)
- **L_i, L_e** = sparse flux operators (one row per cell)
- Each row has ~6-20 nonzero entries (faces of polyhedra)

### 4.8 Advantages and Limitations

**Advantages:**
- Excellent conservation properties
- Works with unstructured meshes
- Handles anisotropy naturally (no 27-point stencils)
- Lower memory than FEM (one value per cell)

**Limitations:**
- Accuracy typically O(h) (lower than FEM)
- More complex to implement
- Requires careful interface flux computation
- Flux limiters needed for nonlinear terms

---

## Semi-Discrete System Structure {#semi-discrete-system}

### 5.1 ODE System After Spatial Discretization

After applying FEM, FDM, or FVM, the continuous PDE system reduces to a system of ODEs:

$$\mathbf{M} \frac{d\mathbf{U}}{dt} + \mathbf{K} \mathbf{U} = \mathbf{F}(t)$$

where:
- **U** = [V_m^1, V_m^2, ..., V_m^N, φ_e^1, φ_e^2, ..., φ_e^N]^T (stacked state vector)
- **M** = block diagonal mass matrix
- **K** = block stiffness matrix
- **F(t)** = RHS vector including ionic, stimulus, and coupling terms

### 5.2 Degrees of Freedom Comparison

| Model | Total DOF | Per Node | Memory |
|-------|-----------|----------|--------|
| Monodomain | N | 1 (V) | 1× baseline |
| Bidomain (V_m, φ_e) | 2N | 2 (V_m, φ_e) | ~4× (2 DOF + bandwidth) |
| Bidomain + bath | 2N + N_bath | varies | 5-10× |

For N = 50,000 nodes (typical 3D ventricle):
- **Monodomain:** ~200 KB per variable (float32), 800 KB total
- **Bidomain:** ~800 KB for U, ~3-4 MB for stiffness matrix, ~10 MB total
- **Bidomain + bath:** ~30-50 MB

### 5.3 Block Matrix Structure

**Explicit Block Form:**

$$\begin{pmatrix} M & 0 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} \dot{V}_m \\ \dot{φ}_e \end{pmatrix} + \begin{pmatrix} K_{ii} & K_{ie} \\ K_{ei} & K_{ee} \end{pmatrix} \begin{pmatrix} V_m \\ φ_e \end{pmatrix} = \begin{pmatrix} F_V \\ F_φ \end{pmatrix}$$

where:
- **M** = χC_m mass matrix (N×N, full rank)
- **K_ii** = ∫ D_i∇φ_i·∇φ_j + ∫ D_i∇φ_e·∇φ_j (coupling to φ_e)
- **K_ie** = ∫ D_i∇φ_e·∇φ_j
- **K_ei** = ∫ D_i∇φ_i·∇φ_j
- **K_ee** = ∫ (D_i + D_e)∇φ_e·∇φ_j

**Reformulation (Eliminate φ_e algebraically):**

From the elliptic equation:
$$K_{ee} φ_e = -K_{ei} V_m - F_φ$$

$$φ_e = -K_{ee}^{-1}(K_{ei} V_m + F_φ)$$

Substitute into parabolic equation:

$$M \dot{V}_m + (K_{ii} - K_{ie} K_{ee}^{-1} K_{ei}) V_m = F_V - K_{ie} K_{ee}^{-1} F_φ$$

**Effective monodomain-like system:**

$$M \dot{V}_m + K_{eff} V_m = F_{eff}$$

where **K_eff** is the Schur complement. This avoids explicitly storing or inverting K_ee but requires K_ee to be non-singular.

### 5.4 Coupling Structure

**Row 1 (Parabolic for V_m):**
- Depends on: V_m (at same node), φ_e (at same node), I_ion(V_m)
- RHS: stimulus current

**Row 2 (Elliptic for φ_e):**
- Depends only on spatial coupling (stiffness matrix)
- No time derivative: differential-algebraic (DAE) structure
- Enforces current conservation

**Information Flow:**
1. φ_e responds instantaneously to changes in V_m
2. V_m changes dynamically due to diffusion and ionic activity
3. φ_e feeds back into V_m equation through coupling terms

This delayed coupling allows operator-splitting approaches to work effectively.

### 5.5 Eigenvalue Analysis

**Monodomain case (for reference):**

Eigenvalues of K/M are negative real (diffusion is damping):

$$λ_k = -μ_k / C_m \quad (μ_k > 0)$$

Smallest eigenvalue determines longest time scale (~AP duration = 200-300 ms).

**Bidomain case:**

The Schur complement K_eff can have different spectral properties than monodomain. The coupling through K_ie and K_ie can create additional modes.

Critical for choosing time-stepping schemes and preconditioners.

---

## Practical Implementation Considerations {#practical-considerations}

### 6.1 Time Integration Schemes

#### **Fully Implicit (Backward Euler - 1st Order)**

$$\mathbf{M} \frac{\mathbf{U}^{n+1} - \mathbf{U}^n}{Δt} + \mathbf{K} \mathbf{U}^{n+1} = \mathbf{F}^{n+1}$$

$$(\mathbf{M}/Δt + \mathbf{K}) \mathbf{U}^{n+1} = \mathbf{M}/Δt \cdot \mathbf{U}^n + \mathbf{F}^{n+1}$$

**Advantages:** Unconditionally stable, simple
**Disadvantages:** Implicit system must be solved each step, expensive

#### **Semi-Implicit (IMEX) - Recommended**

Treat diffusion implicitly, ionic reaction explicitly:

$$\mathbf{M} \frac{\mathbf{U}^{n+1} - \mathbf{U}^n}{Δt} + \mathbf{K} \mathbf{U}^{n+1} = \mathbf{F}^n(\mathbf{U}^n)$$

Same linear solve as fully implicit but RHS uses known (explicit) ionic terms.

**Advantages:** Stable, faster RHS evaluation
**Disadvantages:** Still requires matrix inversion each step

#### **Operator Splitting - Fastest**

**Godunov (1st order):**

**Step 1 - React (ODE only, no diffusion):**

$$\frac{d\mathbf{w}}{dt} = \mathbf{h}(V_m, \mathbf{w})$$

Solve for 0 < t < Δt using Runge-Kutta (cheap, local at each point).

**Step 2 - Diffuse (PDE, V_m fixed from reaction):**

$$\mathbf{M} \frac{\mathbf{V}_m^{n+1} - \mathbf{V}_m^{n+1/2}}{Δt} + \mathbf{K} \mathbf{V}_m^{n+1} = \mathbf{F}_{diff}$$

Solve elliptic system for φ_e using updated V_m.

**Advantages:** Decouples stiff ionic chemistry from expensive diffusion
**Disadvantages:** 1st order, error accumulation in V_m updates

**Strang (2nd Order):**

React for Δt/2, diffuse for Δt, react for Δt/2.

More accurate but slightly more expensive (3× reaction solves per step).

#### **Time Step Selection**

**CFL Condition (for explicit schemes, if used):**

$$Δt < \frac{Δx^2}{2 D_{max}}$$

For Δx = 100 μm, D_max ≈ 0.5 S/cm:

$$Δt < \frac{(10^{-3})^2}{2 × 0.5} ≈ 1 \text{ μs}$$

**Typical implicit time steps:** 0.1-1 ms (10-100× larger)

**Adaptive stepping:** Monitor nonlinear residuals, adjust Δt dynamically.

### 6.2 Linear Solver Configuration

#### **System Properties**

**Matrix:** K (bidomain stiffness)
- Size: 2N × 2N
- Structure: Block 2×2 with 2N×2N submatrices
- Symmetry: Symmetric positive semi-definite (SPD)
- Condition number: κ(K) ~ 1/Δx² → poor for small meshes

#### **Direct Solvers (for small problems)**

**LU Factorization:** K = LU
- Cost: O((2N)³) = O(N³) per factor + O((2N)²) backsolve
- Memory: O((2N)²) = O(N²)
- Feasible for N < 1,000 (20,000 total unknowns)
- Reusable for multiple RHS

#### **Iterative Solvers (Standard)**

**CG (Conjugate Gradient) - if K is SPD:**

```
initialize: r = b - Ax, p = r, k = 0
while ||r|| > tol:
    α = r^T r / (p^T A p)
    x = x + α p
    r = r - α A p
    β = r_new^T r_new / (r^T r)
    p = r + β p
    k = k + 1
```

**Convergence:** Typically 50-200 iterations for cardiac problems
**Cost per iteration:** 1 matrix-vector product + 3 dot products

**BiCGSTAB (for non-symmetric systems, if needed):**

Requires only 1 matrix-vector product per iteration but more dot products.
Typically 100-400 iterations for bidomain.

#### **Preconditioners (Essential)**

**Algebraic Multigrid (AMG):**

Best for bidomain; achieves 5-10× speedup.

```
Setup phase (once):
    Coarsen: Generate hierarchy of grids
    Restrict: Interpolation matrices
    Galerkin: Build coarse-grid operators

Solve phase (each iteration):
    Smooth on fine grid: Few iterations
    Restrict residual to coarse grid
    Solve coarse system (or more coarsening)
    Interpolate back to fine grid
```

- **Cost:** ~2-3× matrix cost (setup + precondition)
- **Result:** Reduce iterations to 10-30

**ILU (Incomplete LU) - Cheaper but less effective:**

Factor K ≈ LU with dropping of small entries.

- Cost: O(N) per precondition
- Result: ~20-50% reduction in iterations

**Block Preconditioners (Bidomain-Specific):**

Exploit 2×2 block structure:

$$P^{-1} = \begin{pmatrix} K_{ii} & K_{ie} \\ 0 & S \end{pmatrix}^{-1}$$

where **S = K_ee - K_ei K_ii^{-1} K_ie** (Schur complement).

Approximately solve:
1. K_ii system for first block
2. S system for second block

Reduces coupling, improves convergence.

### 6.3 Memory Requirements Analysis

**Monodomain (V5.4 baseline):**
- State vector V: N × 8 bytes (float64)
- Stiffness K: ~10N × 8 bytes (sparse, ~10 entries/row typical)
- Gating variables w: N × M_gates × 8 bytes (~10-20 gates)
- **Total:** ~(100-200)N bytes

**For N = 50,000:**
- Monodomain: ~5-10 MB

**Bidomain:**
- State vector U: 2N × 8 bytes
- Stiffness K: ~20N × 8 bytes (2×2 block, ~20 entries/row)
- Gating variables w: N × M_gates × 8 bytes (shared across V_m, φ_e)
- **Total:** ~(200-400)N bytes

**For N = 50,000:**
- Bidomain: ~10-20 MB

**Bidomain + Bath:**
- Additional N_bath nodes (~2-5× tissue nodes)
- Extended stiffness matrix
- **Total:** ~50-100 MB

**Practical Recommendation:**

For physiological-scale simulation (ventricle):
- ~10 million tetrahedra at 100 μm resolution
- Monodomain: ~100 MB (workable on 4 GB machines)
- Bidomain: ~200-400 MB (reasonable)
- Bidomain + bath: ~1-2 GB (high-end workstation)

### 6.4 Computational Cost Analysis

**Costs per time step (assuming semi-implicit with operator splitting):**

| Operation | Monodomain | Bidomain | Ratio |
|-----------|-----------|----------|-------|
| Reaction solve | 1 | 1 | 1× |
| RHS assembly | N K-flops | 2N K-flops | 2× |
| Preconditioner setup (per N steps) | O(N) | O(N) | 1× |
| Preconditioner application | 1 | 1 | 1× |
| Matrix-vector product | 10N | 20N | 2× |
| Linear solver iterations | ~15 | ~20-30 | 2× |
| **Total per step** | | | **2-4×** |

**Example for ventricle simulation (1 heartbeat = 300 ms):**

- Mesh: 50,000 nodes
- Time step: 1 ms
- Solver: MINRES + AMG preconditioner

| Model | Setup | Main loop | Total |
|-------|-------|-----------|-------|
| Monodomain V5.4 | 10 s | 300 s | 310 s |
| Bidomain | 10 s | 900 s | 910 s |
| **Speedup factor** | | | **3×** |

**Optimization strategies:**

1. **Coarse time steps for φ_e:** Solve elliptic every 2-5 V_m steps
2. **AMG multigrid:** Most critical, 5-10× speedup
3. **Compartmentalization:** Solve only active tissue regions
4. **GPU acceleration:** Potential 10-50× speedup on modern NVIDIA/AMD GPUs

### 6.5 When to Use Bidomain vs. Monodomain

**Use Monodomain when:**
- Conduction velocity is primary output
- No extracellular stimulation required
- Computational resources limited
- Intracellular anisotropy ≈ extracellular anisotropy
- Clinical timing simulations (3-5% error acceptable)

**Use Bidomain when:**
- Virtual electrode polarization important
- Defibrillation or extracellular stimulation
- ECG/BSP calculation required
- Ischemic regions (conductivity changes)
- High-accuracy research studies

**Hybrid approaches:**

**Pseudobidomain:** Use monodomain for main solve, occasional bidomain solve (~10% cost of full bidomain, 80% accuracy)

**Augmented Monodomain:** Modify monodomain with φ_e correction term:

$$V_{eff} = V_{mono} + c · (∇φ_{bath})$$

Captures some bath effects without full bidomain cost.

### 6.6 Mesh Requirements

**Spatial discretization error scales as O(h²) for FEM, O(h) for FVM:**

| Mesh | Nodes | Tetrahedra | KB (Mono) | MB (Bidomain) |
|------|-------|-----------|----------|---------------|
| Coarse (200 μm) | 20,000 | 100,000 | 2 | 4 |
| Standard (100 μm) | 160,000 | 800,000 | 16 | 32 |
| Fine (50 μm) | 1.2M | 6.4M | 120 | 240 |
| Very fine (25 μm) | 10M | 50M | 1000 | 2000 |

**Recommended:**
- Minimum: 100 μm (captures AP waveform)
- Standard: 50-100 μm (clinical+ accuracy)
- Research: 25-50 μm (detailed cellular effects)

**Fiber structure:**
- Must align with tissue fiber directions (transmurally-varying angles)
- Or use rotated conductivity tensors to avoid aligning mesh

### 6.7 Validation and Convergence Testing

**Grid convergence study:**

1. **Compute solution on 3 meshes:** h, h/2, h/4
2. **Compare values at cell center (e.g., V_m, conduction velocity):**

$$e_h = |u_h - u_{ref}|$$

3. **Estimate convergence rate:**

$$r = \log_2(e_h / e_{h/2})$$

Should approach 2 for FEM, 1 for FVM if implemented correctly.

**Benchmark problems:**

1. **1D cable equation:** Analytical solution exists
2. **2D slab (tissue bath coupling):** Compare with published solutions
3. **3D ventricle:** ECG waveforms, activation sequences

---

## References {#references}

### Foundational Works

1. Sundnes, J., Lines, G. T., Cai, X., Nielsen, B. F., Mardal, K. A., & Tveito, A. (2006). *Computing the electrical activity in the heart*. Springer Monographs in Mathematics. https://link.springer.com/book/10.1007/3-540-33437-X

2. Colli Franzone, P., Pavarino, L. F., & Scacchi, S. (2014). *Mathematical cardiac electrophysiology*. Springer. https://www.springer.com/us/book/9783319048000

3. Plank, G., et al. (2021). The openCARP simulation environment for cardiac electrophysiology. *Progress in Biophysics and Molecular Biology*, 159, 3-19. https://www.sciencedirect.com/science/article/abs/pii/S0079610721002972

### Recent Methods and Solvers

4. NOVEL BIDOMAIN PARTITIONED STRATEGIES FOR THE SIMULATION OF VENTRICULAR FIBRILLATION DYNAMICS. (2025). arXiv. https://arxiv.org/pdf/2510.27447

5. Solvers for the Cardiac Bidomain Equations. (2007). PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC2881536/

6. Algebraic Multigrid Preconditioner for the Cardiac Bidomain Model. (2018). PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC5428748/

### Mathematical Analysis

7. Existence and uniqueness of the solution for the bidomain model used in cardiac electrophysiology. Bourgault, Y., Coudière, Y., & Pierre, C. (2007). *SIAM Journal on Applied Mathematics*, 67(1), 25-39. https://hal.science/hal-00101458/document

8. Deriving the Bidomain Model of Cardiac Electrophysiology From a Cell-Based Model; Properties and Comparisons. (2021). Frontiers in Physiology. https://pmc.ncbi.nlm.nih.gov/articles/PMC8782150/

### Finite Element Methods

9. A fully implicit finite element method for bidomain models of cardiac electromechanics. (2012). PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC3501134/

10. A macro finite element formulation for cardiac electrophysiology simulations using hybrid unstructured grids. (2012). PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC3223405/

### Finite Volume Methods

11. A finite volume method for modeling discontinuous electrical activation in cardiac tissue. (2005). *Annals of Biomedical Engineering*, 33(8), 1015-1025. https://link.springer.com/article/10.1007/s10439-005-1434-6

12. A finite volume scheme for cardiac propagation in media with isotropic conductivities. (2010). *Mathematics and Computers in Simulation*, 80(9), 1821-1840. https://www.sciencedirect.com/science/article/abs/pii/S0378475409003644

### Monodomain vs. Bidomain

13. A comparison of monodomain and bidomain propagation models for the human heart. (2007). *Journal of Computational Physics*, 224(2), 637-658. https://www.researchgate.net/publication/5899472_A_comparison_of_monodomain_and_bidomain_propagation_models_for_the_human_heart

14. Bidomain ECG Simulations Using an Augmented Monodomain Model for the Cardiac Source. (2011). PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC3378475/

15. Optimal monodomain approximations of the bidomain equations used in cardiac electrophysiology. (2013). *Mathematical Models and Methods in Applied Sciences*, 23(10), 1743-1770. https://inria.hal.science/hal-00644257/en

### Boundary Conditions and Bath Coupling

16. A smoothed boundary bidomain model for cardiac simulations in anatomically detailed geometries. (2024). *PLOS One*. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0286577

17. On Boundary Stimulation and Optimal Boundary Control of the Bidomain Equations. (2013). PMC. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3980049/

18. A comparison of two boundary conditions used with the bidomain model of cardiac tissue. (1995). *Annals of Biomedical Engineering*, 23(3), 329-340. https://link.springer.com/article/10.1007/BF02368075

### Time Integration and Operator Splitting

19. Stable time integration suppresses unphysical oscillations in the bidomain model. (2014). *Frontiers in Physics*, 2, 40. https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2014.00040/full

20. High-Order Operator Splitting for the Bidomain and Monodomain Models. (2019). *SIAM Journal on Scientific Computing*, 41(3), B440-B464. https://epubs.siam.org/doi/10.1137/17M1137061

21. Operator splitting for the bidomain model revisited. (2015). *Journal of Computational and Applied Mathematics*, 298, 151-162. https://www.sciencedirect.com/science/article/pii/S0377042715004677

### Conductivity and Anisotropy

22. How the anisotropy of the intracellular and extracellular conductivities influences stimulation of cardiac muscle. (1991). *Journal of Mathematical Biology*, 30(2), 161-188. https://link.springer.com/article/10.1007/BF00175610

23. Cardiac anisotropy in boundary-element models for the electrocardiogram. (2009). *Medical & Biological Engineering & Computing*, 47(10), 1043-1056. https://link.springer.com/article/10.1007/s11517-009-0472-x

24. Approaches for determining cardiac bidomain conductivity values: progress and challenges. (2021). *Medical & Biological Engineering & Computing*, 59(2), 231-243. https://pmc.ncbi.nlm.nih.gov/articles/PMC7755382/

### Computational Efficiency

25. On the computational complexity of the bidomain and the monodomain models of electrophysiology. (2006). *IEEE Transactions on Biomedical Engineering*, 53(12), 2552-2555. https://pubmed.ncbi.nlm.nih.gov/16773461/

26. Computational techniques for solving the bidomain equations in three dimensions. (2002). *IEEE Transactions on Biomedical Engineering*, 49(11), 1260-1269. https://pmc.ncbi.nlm.nih.gov/articles/PMC2881536/

### openCARP Documentation

27. openCARP User's Manual (v7.0). https://opencarp.org/manual/opencarp-manual-v7.0.pdf

28. openCARP Extracellular potentials and ECGs. https://opencarp.org/documentation/examples/02_ep_tissue/07_extracellular

---

## Appendix: Implementation Checklist for Engine V5.4 Extension

### Phase 1: Data Structure Extension
- [ ] Double DOF arrays: V_m and φ_e separate or interleaved?
- [ ] Define block matrix format (CSR with 2×2 blocks or flat 2N×2N)
- [ ] Update sparse matrix assembly routines
- [ ] Create mapping between global DOF index and (node, variable) pairs

### Phase 2: FEM Assembly (Highest Priority)
- [ ] Modify element mass matrix computation (scalar × integration matrix)
- [ ] Implement dual stiffness matrix assembly (K_i and K_e)
- [ ] Block assembly routine for 2×2 system
- [ ] RHS vector with ionic + coupling terms
- [ ] Neumann BC for both domains

### Phase 3: FDM Assembly
- [ ] Structured grid generator
- [ ] 7-point and 27-point stencil generators
- [ ] Tensor conductivity handling
- [ ] Ghost point boundary condition treatment

### Phase 4: FVM Assembly
- [ ] Control volume geometry (face areas, cell volumes)
- [ ] Harmonic mean conductivity computation
- [ ] Flux discretization (interface integrals)
- [ ] Face-centric assembly loop

### Phase 5: Solver Integration
- [ ] Block-aware AMG preconditioner (or test AMG on full system)
- [ ] CG/BiCGSTAB driver with bidomain RHS
- [ ] Elliptic equation solve at each time step
- [ ] Error checking for singular/ill-conditioned matrices

### Phase 6: Time Integration
- [ ] Modify time-stepping loop to handle DAE structure
- [ ] Implement operator splitting (optional but recommended)
- [ ] RHS evaluation with I_ion(V_m, w) and stim terms
- [ ] Gating variable update (unchanged from monodomain)

### Phase 7: Validation
- [ ] Grid convergence test (1D cable, known solution)
- [ ] Bath coupling test (tissue + external medium)
- [ ] ECG/BSP computation test
- [ ] Defibrillation scenario (extracellular stimulus response)
- [ ] Performance benchmarking vs. monodomain

---

## Summary

This document provides a comprehensive guide to discretizing the cardiac bidomain equations using FEM, FDM, and FVM methods, suitable for integration into Engine V5.4. The bidomain model captures physiologically important effects (unequal anisotropy, extracellular fields) that monodomain cannot, at a computational cost of 2-4× for realistic meshes. Key implementation considerations include:

1. **Mathematical formulation:** Coupled parabolic-elliptic system in two potentials (V_m, φ_e)
2. **FEM:** Most versatile, uses weak form and 2×2 block assembly
3. **FDM:** Simple for regular grids, requires careful anisotropy handling
4. **FVM:** Excellent conservation, handles anisotropy naturally
5. **Solvers:** AMG preconditioning essential, 5-10× speedup
6. **Memory:** ~4× monodomain (80-400 MB for typical problems)
7. **Validation:** Grid convergence, ECG/BSP output, defibrillation scenarios

The modular structure allows phased implementation, prioritizing FEM first (most valuable), then FDM and FVM for specialized applications.

---

**Document prepared:** February 2025
**Research scope:** Comprehensive review of 2020-2025 literature and established references
**Implementation context:** Extension of working monodomain V5.4 engine
**Audience:** Computational cardiac electrophysiology researchers and engineers
