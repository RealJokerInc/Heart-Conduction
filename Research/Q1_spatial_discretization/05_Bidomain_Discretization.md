# Spatial Discretization of the Bidomain Equations: Research Report

## 1. The Bidomain Equations

The bidomain model treats cardiac tissue as two interpenetrating continuous domains (intracellular, extracellular) separated by the cell membrane. In the `Vm/phi_e` formulation preferred for numerical work:

**Parabolic equation** (transmembrane potential evolution):
```
chi * Cm * dVm/dt = div(sigma_i * grad(Vm)) + div(sigma_i * grad(phi_e)) - chi*Iion(Vm,q) + Istim
```

**Elliptic equation** (extracellular potential):
```
div((sigma_i + sigma_e) * grad(phi_e)) = -div(sigma_i * grad(Vm))
```

Where `Vm = phi_i - phi_e`, `sigma_i` and `sigma_e` are the intracellular/extracellular conductivity tensors, `chi` is the surface-to-volume ratio (~1400 cm^-1), `Cm` is membrane capacitance (~1 uF/cm^2).

**Conductivity tensors** from fiber orientation (2D transversely isotropic):
```
sigma_a = sigma_a_l * (f x f^T) + sigma_a_t * (I - f x f^T)    for a in {i, e}
```
Expanding with fiber angle theta:
```
sigma_a_xx = sigma_a_l * cos^2(theta) + sigma_a_t * sin^2(theta)
sigma_a_yy = sigma_a_l * sin^2(theta) + sigma_a_t * cos^2(theta)
sigma_a_xy = (sigma_a_l - sigma_a_t) * cos(theta) * sin(theta)
```

Typical conductivity values (Clerc 1976, mS/cm): `sigma_i_l = 1.74`, `sigma_i_t = 0.19`, `sigma_e_l = 6.25`, `sigma_e_t = 2.36`.

**Key difference from monodomain**: The bidomain requires TWO spatial operators and solving TWO coupled PDEs per timestep. The monodomain is obtained under the equal anisotropy ratio assumption (`sigma_i = alpha * sigma_e`), yielding a single effective operator.

---

## 2. FDM for Bidomain

### 2.1 The Two Operators

The bidomain FDM requires two discrete Laplacian-type operators:

- **Ai**: discretization of `div(sigma_i * grad(.))` (intracellular operator)
- **Asum**: discretization of `div((sigma_i + sigma_e) * grad(.))` (sum operator)

Both are constructed using the same stencil machinery but with different conductivity tensors. The discrete system becomes:

**Parabolic** (after time discretization):
```
chi * Cm * (Vm^{n+1} - Vm^n) / dt = Ai * Vm^{n+1} + Ai * phi_e^{n+1} - chi * Iion^n + Istim
```

**Elliptic**:
```
Asum * phi_e^{n+1} = -Ai * Vm^{n+1}
```

### 2.2 Five-Point Stencil (Isotropic / Axis-Aligned Anisotropy)

When `sigma_xy = 0` (fibers aligned with grid axes), the 5-point stencil suffices. For the intracellular operator Ai at node (i,j):

```
(Ai * u)[i,j] = sigma_i_xx * (u[i+1,j] - 2*u[i,j] + u[i-1,j]) / dx^2
              + sigma_i_yy * (u[i,j+1] - 2*u[i,j] + u[i,j-1]) / dy^2
```

Stencil weights:
```
         [  0,              sigma_i_yy/dy^2,                        0            ]
Ai_5pt = [sigma_i_xx/dx^2,  -2*(sigma_i_xx/dx^2 + sigma_i_yy/dy^2),  sigma_i_xx/dx^2]
         [  0,              sigma_i_yy/dy^2,                        0            ]
```

The sum operator Asum has identical structure with `sigma_i + sigma_e` replacing `sigma_i`.

### 2.3 Nine-Point Stencil (Anisotropic with Fiber Rotation)

When `sigma_xy != 0`, the cross-derivative `d^2u/dxdy` requires corner neighbors:

```
d^2u/dxdy ~ (u[i+1,j+1] - u[i+1,j-1] - u[i-1,j+1] + u[i-1,j-1]) / (4*dx*dy)
```

Full 9-point stencil at node (i,j):
```
NW = -sigma_xy / (4*dx*dy)     N = sigma_yy / dy^2          NE = +sigma_xy / (4*dx*dy)
W  = sigma_xx / dx^2           C = -2*(sigma_xx/dx^2 + sigma_yy/dy^2)   E = sigma_xx / dx^2
SW = +sigma_xy / (4*dx*dy)     S = sigma_yy / dy^2          SE = -sigma_xy / (4*dx*dy)
```

**Note on the factor of 2 in Dxy**: The full expansion of `div(D * grad(u))` with constant D gives `Dxx * d^2u/dx^2 + 2*Dxy * d^2u/dxdy + Dyy * d^2u/dy^2`. The factor of 2 is embedded when sigma_xy comes from the symmetric tensor (the two cross-derivative terms `d/dx(sigma_xy * du/dy) + d/dy(sigma_xy * du/dx)` combine to give `2*sigma_xy * d^2u/dxdy` for constant sigma).

For the bidomain, you apply this stencil TWICE with different conductivities:
- **Ai**: uses `(sigma_i_xx, sigma_i_xy, sigma_i_yy)`
- **Asum**: uses `(sigma_i_xx + sigma_e_xx, sigma_i_xy + sigma_e_xy, sigma_i_yy + sigma_e_yy)`

**M-matrix / positivity condition** (prevents spurious oscillations):
```
|sigma_xy| <= min(sigma_xx * dy / (2*dx), sigma_yy * dx / (2*dy))
```

This is typically satisfied for realistic cardiac conductivity ratios on reasonable grid spacings.

### 2.4 Conservative Form with Spatially Varying Conductivity

For spatially varying sigma(x,y), the conservative form is essential:

```
div(sigma * grad(u)) = d/dx(sigma_xx * du/dx + sigma_xy * du/dy)
                     + d/dy(sigma_xy * du/dx + sigma_yy * du/dy)
```

**Flux at x-interface (i+1/2, j)**:
```
Fx[i+1/2,j] = sigma_xx[i+1/2,j] * (u[i+1,j] - u[i,j]) / dx
            + sigma_xy[i+1/2,j] * (u[i,j+1] + u[i+1,j+1] - u[i,j-1] - u[i+1,j-1]) / (4*dy)
```

**Interface conductivity via harmonic mean** (critical for sharp tissue boundaries):
```
sigma_xx[i+1/2,j] = 2 * sigma_xx[i,j] * sigma_xx[i+1,j] / (sigma_xx[i,j] + sigma_xx[i+1,j])
```

For the bidomain, harmonic means are computed independently for `sigma_i` and for `sigma_i + sigma_e`:
```
sigma_i_xx_face[i+1/2,j]   = harmonic_mean(sigma_i_xx[i,j], sigma_i_xx[i+1,j])
sigma_sum_xx_face[i+1/2,j]  = harmonic_mean(sigma_i_xx[i,j]+sigma_e_xx[i,j],
                                              sigma_i_xx[i+1,j]+sigma_e_xx[i+1,j])
```

For the cross-term sigma_xy, arithmetic mean is standard:
```
sigma_xy_face = 0.5 * (sigma_xy[i,j] + sigma_xy[i+1,j])
```

### 2.5 Matrix Assembly Pseudocode (Bidomain FDM)

```python
def assemble_bidomain_fdm_operators(Nx, Ny, dx, dy, sigma_i, sigma_e, fiber_angle):
    """
    Assemble the two bidomain FDM operators Ai and Asum.

    Returns:
        Ai: sparse (N x N) for div(sigma_i * grad(.))
        Asum: sparse (N x N) for div((sigma_i + sigma_e) * grad(.))
    """
    N = Nx * Ny
    c, s = np.cos(fiber_angle), np.sin(fiber_angle)

    # Intracellular tensor at each node
    si_xx = sigma_i['l'] * c**2 + sigma_i['t'] * s**2
    si_yy = sigma_i['l'] * s**2 + sigma_i['t'] * c**2
    si_xy = (sigma_i['l'] - sigma_i['t']) * c * s

    # Sum tensor at each node
    ss_xx = (sigma_i['l']+sigma_e['l'])*c**2 + (sigma_i['t']+sigma_e['t'])*s**2
    ss_yy = (sigma_i['l']+sigma_e['l'])*s**2 + (sigma_i['t']+sigma_e['t'])*c**2
    ss_xy = (sigma_i['l']+sigma_e['l'] - sigma_i['t']-sigma_e['t']) * c * s

    def build_operator(Dxx, Dxy, Dyy):
        rows, cols, vals = [], [], []
        for j in range(Ny):
            for i in range(Nx):
                idx = j * Nx + i
                center = 0.0

                # Axial neighbors with harmonic mean face conductivities
                if i < Nx-1:
                    w = 2*Dxx[i,j]*Dxx[i+1,j]/(Dxx[i,j]+Dxx[i+1,j]+1e-30) / dx**2
                    rows.append(idx); cols.append(idx+1); vals.append(w); center -= w
                if i > 0:
                    w = 2*Dxx[i,j]*Dxx[i-1,j]/(Dxx[i,j]+Dxx[i-1,j]+1e-30) / dx**2
                    rows.append(idx); cols.append(idx-1); vals.append(w); center -= w
                if j < Ny-1:
                    w = 2*Dyy[i,j]*Dyy[i,j+1]/(Dyy[i,j]+Dyy[i,j+1]+1e-30) / dy**2
                    rows.append(idx); cols.append(idx+Nx); vals.append(w); center -= w
                if j > 0:
                    w = 2*Dyy[i,j]*Dyy[i,j-1]/(Dyy[i,j]+Dyy[i,j-1]+1e-30) / dy**2
                    rows.append(idx); cols.append(idx-Nx); vals.append(w); center -= w

                # Diagonal neighbors (cross-derivative)
                dxy4 = 4*dx*dy
                if i<Nx-1 and j<Ny-1:
                    w = Dxy[i,j]/dxy4; rows.append(idx); cols.append(idx+Nx+1); vals.append(w); center -= w
                if i>0 and j<Ny-1:
                    w = -Dxy[i,j]/dxy4; rows.append(idx); cols.append(idx+Nx-1); vals.append(w); center -= w
                if i<Nx-1 and j>0:
                    w = -Dxy[i,j]/dxy4; rows.append(idx); cols.append(idx-Nx+1); vals.append(w); center -= w
                if i>0 and j>0:
                    w = Dxy[i,j]/dxy4; rows.append(idx); cols.append(idx-Nx-1); vals.append(w); center -= w

                rows.append(idx); cols.append(idx); vals.append(center)

        return sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))

    Ai = build_operator(si_xx, si_xy, si_yy)
    Asum = build_operator(ss_xx, ss_xy, ss_yy)
    return Ai, Asum
```

### 2.6 Operator Splitting Algorithm

The standard three-step algorithm (Vigmond et al. 2002):

```
Step 1 (Parabolic with Crank-Nicolson):
  (chi*Cm*I/dt - 0.5*Ai) * Vm^{n+1/2} = (chi*Cm*I/dt + 0.5*Ai) * Vm^n + Ai * phi_e^n

Step 2 (Reaction):
  Vm^{n+1} = Vm^{n+1/2} - dt * Iion(Vm^{n+1/2}, q^n) / (chi * Cm)
  q^{n+1} = advance_ionic(Vm^{n+1/2}, q^n, dt)

Step 3 (Elliptic):
  Asum * phi_e^{n+1} = -Ai * Vm^{n+1}
```

Alternative Godunov splitting (simpler, first-order):
```
Step 1: Reaction ODE solve
Step 2: (chi*Cm*I - dt*Ai) * Vm^{n+1} = chi*Cm*V* + dt*Ai*phi_e^n
  OR explicit: Vm^{n+1} = V* + dt/(chi*Cm)*(Ai*V* + Ai*phi_e^n)
Step 3: Asum * phi_e^{n+1} = -Ai * Vm^{n+1}
```

**CFL condition** for explicit parabolic (from the PMC2881536 review):
```
dt <= chi*Cm*dx^2 / (2*(sigma_l + sigma_t))
```
where `sigma_* = sigma_i_* * sigma_e_* / (sigma_i_* + sigma_e_*)`. Semi-implicit methods allow dt over 100x larger.

---

## 3. FVM for Bidomain

### 3.1 Conservative Formulation

Integrating the parabolic equation over control volume Omega_ij:

```
chi*Cm*|Omega_ij|*dVm_ij/dt = integral_{boundary} sigma_i*grad(Vm + phi_e) . n dS
                              - chi*|Omega_ij|*Iion + Istim
```

For the elliptic equation:
```
integral_{boundary} (sigma_i + sigma_e)*grad(phi_e) . n dS = -integral_{boundary} sigma_i*grad(Vm) . n dS
```

### 3.2 Two-Point Flux Approximation (TPFA) for Bidomain

On a structured Cartesian grid, the flux across each face involves the conductivity tensor.

**East face flux** (face normal = (1,0)) for any field u:
```
F_east = sigma_xx_face * (u[i+1,j] - u[i,j]) / dx * dy
       + sigma_xy_face * (du/dy)_face * dy
```

where `(du/dy)_face = 0.25*(u[i,j+1]+u[i+1,j+1]-u[i,j-1]-u[i+1,j-1])/dy` (4-point average).

**North face flux** (face normal = (0,1)):
```
F_north = sigma_xy_face * (du/dx)_face * dx
        + sigma_yy_face * (u[i,j+1] - u[i,j]) / dy * dx
```

For the bidomain, each flux is computed twice:
- Intracellular flux uses `sigma_i` components
- Sum flux uses `sigma_i + sigma_e` components

Face conductivities use harmonic mean for diagonal terms and arithmetic mean for cross terms.

### 3.3 Divergence (Flux Balance)

```
div(sigma*grad(u))_ij = (F_east - F_west + F_north - F_south) / |Omega_ij|
```

For a uniform grid with `|Omega| = dx*dy`, this simplifies to the same stencil structure as FDM but with properly averaged face conductivities.

### 3.4 TPFA Limitations

Standard TPFA is only consistent for K-orthogonal grids. The cross-gradient reconstruction above already extends it to a multi-point scheme. For severe anisotropy, options include MPFA (Multi-Point Flux Approximation), Diamond schemes, or nonlinear two-point methods. For structured grids with moderate cardiac anisotropy ratios (4:1 to 10:1), the extended TPFA with cross-gradient averaging is adequate.

### 3.5 FVM Implementation Pseudocode

```python
def bidomain_fvm_operators(Vm, phi_e, sigma_i_field, sigma_e_field, dx, dy):
    """Compute Ai*u and Asum*u via FVM flux computation."""

    def flux_divergence(u, sigma):
        sxx, sxy, syy = sigma['xx'], sigma['xy'], sigma['yy']

        # Face conductivities
        sxx_e = 2*sxx[:-1,:]*sxx[1:,:]/(sxx[:-1,:]+sxx[1:,:]+1e-30)
        syy_n = 2*syy[:,:-1]*syy[:,1:]/(syy[:,:-1]+syy[:,1:]+1e-30)
        sxy_e = 0.5*(sxy[:-1,:] + sxy[1:,:])
        sxy_n = 0.5*(sxy[:,:-1] + sxy[:,1:])

        # Normal gradients at faces
        du_dx_e = (u[1:,:] - u[:-1,:]) / dx
        du_dy_n = (u[:,1:] - u[:,:-1]) / dy

        # Cross gradients at faces (4-point average)
        du_dy_e = np.zeros_like(du_dx_e)
        du_dy_e[:,1:-1] = 0.25*(u[:-1,2:]+u[1:,2:]-u[:-1,:-2]-u[1:,:-2])/dy
        du_dx_n = np.zeros_like(du_dy_n)
        du_dx_n[1:-1,:] = 0.25*(u[2:,:-1]+u[2:,1:]-u[:-2,:-1]-u[:-2,1:])/dx

        # Fluxes
        Fx = sxx_e*du_dx_e + sxy_e*du_dy_e
        Fy = sxy_n*du_dx_n + syy_n*du_dy_n

        # Divergence with zero-flux BC
        div_u = np.zeros_like(u)
        div_u[1:-1,:] = (Fx[1:,:]-Fx[:-1,:])/dx
        div_u[0,:]  =  Fx[0,:]/dx;   div_u[-1,:] = -Fx[-1,:]/dx
        div_u[:,1:-1]+= (Fy[:,1:]-Fy[:,:-1])/dy
        div_u[:,0]  += Fy[:,0]/dy;    div_u[:,-1]+= -Fy[:,-1]/dy
        return div_u

    sigma_sum = {k: sigma_i_field[k]+sigma_e_field[k] for k in ('xx','yy','xy')}
    Ai_Vm     = flux_divergence(Vm, sigma_i_field)
    Ai_phi_e  = flux_divergence(phi_e, sigma_i_field)
    Asum_phi_e= flux_divergence(phi_e, sigma_sum)
    return Ai_Vm, Ai_phi_e, Asum_phi_e
```

---

## 4. FEM for Bidomain

### 4.1 Weak Form

Multiply by test functions, integrate, apply Green's formula. Boundary integrals vanish under homogeneous Neumann (this is the "natural" BC in FEM -- a key advantage).

**Parabolic** (test function v):
```
integral chi*Cm*(dVm/dt)*v dOmega + integral (sigma_i*grad(Vm)).grad(v) dOmega
  + integral (sigma_i*grad(phi_e)).grad(v) dOmega
  = -integral chi*Iion*v dOmega + integral Istim*v dOmega
```

**Elliptic** (test function w):
```
integral ((sigma_i+sigma_e)*grad(phi_e)).grad(w) dOmega
  = -integral (sigma_i*grad(Vm)).grad(w) dOmega
```

### 4.2 Matrix System

Using finite element basis functions `{phi_k}`:

**Mass matrix**: `M_kl = integral chi*Cm*phi_k*phi_l dOmega`

**Stiffness matrices**:
```
Ki_kl  = integral (sigma_i*grad(phi_k)).grad(phi_l) dOmega
Ke_kl  = integral (sigma_e*grad(phi_k)).grad(phi_l) dOmega
Kie_kl = Ki_kl + Ke_kl
```

**Discrete parabolic**: `M*dVm/dt + Ki*Vm + Ki*phi_e = -M_chi*Iion + Istim_vec`

**Discrete elliptic**: `Kie*phi_e = -Ki*Vm`

### 4.3 Time-Discretized System

Using theta-scheme (theta=1 for backward Euler, theta=0.5 for Crank-Nicolson):

**Parabolic**:
```
(M/dt + theta*Ki)*Vm^{n+1} = (M/dt - (1-theta)*Ki)*Vm^n - Ki*phi_e^n - M_chi*Iion^n + Istim
```

**Elliptic**:
```
Kie*phi_e^{n+1} = -Ki*Vm^{n+1}
```

**Coupled 2x2 block system** (from Marsh, Bhatt et al. 2020, PMC7767930):
```
[mu*M + Ki,    -mu*M  ] [phi_i^{n+1}]   [mu*M*Vm^n + ...]
[-mu*M,     mu*M + Ke ] [phi_e^{n+1}] = [...]
```
where `mu = Cm/dt`.

### 4.4 Assembly with Fiber Rotation

For each triangular element with nodes {a, b, c}:

```python
def element_stiffness(nodes, sigma_tensor):
    x, y = nodes[:,0], nodes[:,1]
    area = 0.5*abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    B = np.array([[y[1]-y[2], y[2]-y[0], y[0]-y[1]],
                  [x[2]-x[1], x[0]-x[2], x[1]-x[0]]]) / (2*area)
    return area * B.T @ sigma_tensor @ B   # 3x3 element matrix

# For each element:
sigma_i = R(theta) @ diag(sigma_i_l, sigma_i_t) @ R(theta).T
sigma_e = R(theta) @ diag(sigma_e_l, sigma_e_t) @ R(theta).T
Ki_elem = element_stiffness(nodes, sigma_i)
Ke_elem = element_stiffness(nodes, sigma_e)
# Assemble into global Ki, Ke
```

### 4.5 Mass Matrix

**Consistent** (accurate): For linear triangles: `M_elem = chi*Cm*area/12 * [[2,1,1],[1,2,1],[1,1,2]]`

**Lumped** (efficient): `M_kk = chi*Cm * (sum of adjacent element areas)/3`, off-diagonal = 0. Diagonal M makes `M^{-1}` trivial, which simplifies explicit parabolic steps.

---

## 5. Matrix Properties

### 5.1 Properties of Ki and Kie

- **Ki, Ke**: Symmetric, positive semi-definite, rank N-1. Nullspace = constant vectors (`Ki*1 = 0`).
- **Kie = Ki + Ke**: Symmetric positive semi-definite, rank N-1, nullspace = constants.
- **M**: Symmetric positive definite (SPD).
- **M/dt + Ki**: SPD (mass matrix provides the definite part). Well-conditioned.

### 5.2 The Elliptic System is Singular

`Kie * phi_e = -Ki * Vm` is singular because `Kie` has a nullspace of constant vectors. Physically, phi_e is defined only up to an additive constant.

**Compatibility condition**: `1^T * (-Ki * Vm) = 0`. This is automatically satisfied because `Ki*1 = 0` implies `1^T*Ki = 0`.

**Three approaches to handle singularity**:

1. **Pin one node** (simplest): Set `phi_e[0] = 0`. Replace row 0 of Kie with identity row, RHS[0] = 0. Breaks symmetry but easy to implement.

2. **Deflation / nullspace projection** (recommended for matrix-free): After each matrix-vector product in CG, subtract the mean: `r = r - mean(r)*ones`. Maintains symmetry.

3. **Regularization**: Add rank-1 term: `Kie_reg = Kie + epsilon*(ones*ones^T)/N`. Makes the system SPD.

### 5.3 Condition Number Considerations

- **Parabolic system `(M/dt + Ki)`**: kappa ~ O(1) to O(10^2). Well-conditioned due to mass matrix dominance.
- **Elliptic system `Kie`**: kappa ~ O(N) ~ O(1/h^2). Ill-conditioned. Requires preconditioning.

**Solver recommendations**:
- **Parabolic**: PCG with Jacobi or SSOR preconditioner. Converges in <10 iterations.
- **Elliptic**: PCG with AMG or geometric multigrid. AMG achieves near-O(N) solution time. Without preconditioning, CG converges in O(sqrt(N)) iterations -- too slow for large grids.
- **Block preconditioner**: The monodomain operator `Amono = Ki*Ke/(Ki+Ke)` (harmonic mean) serves as an effective Schur complement preconditioner for the coupled system.

---

## 6. Structured Grid / Matrix-Free Operators

### 6.1 Extending the Monodomain Stencil to Bidomain

If the monodomain engine already has a stencil operator, the bidomain extension requires:

1. **Precompute two sets of face conductivities** instead of one: `sigma_i` faces for Ai, and `sigma_i + sigma_e` faces for Asum.
2. **The stencil kernel itself is identical** -- only the conductivity coefficients differ.
3. **Storage**: Double the conductivity face arrays.

For uniform (constant) conductivities, two fixed 3x3 convolution kernels:

```python
# Kernel for Ai (intracellular, isotropic case)
kernel_i = torch.tensor([
    [0, sigma_i_t/dy**2, 0],
    [sigma_i_l/dx**2, -2*(sigma_i_l/dx**2 + sigma_i_t/dy**2), sigma_i_l/dx**2],
    [0, sigma_i_t/dy**2, 0]
]).reshape(1,1,3,3)

# Kernel for Asum
sl, st = sigma_i_l+sigma_e_l, sigma_i_t+sigma_e_t
kernel_sum = torch.tensor([
    [0, st/dy**2, 0],
    [sl/dx**2, -2*(sl/dx**2+st/dy**2), sl/dx**2],
    [0, st/dy**2, 0]
]).reshape(1,1,3,3)

# Apply:
Ai_u   = F.conv2d(F.pad(u, (1,1,1,1), 'replicate'), kernel_i)[0,0]
Asum_u = F.conv2d(F.pad(u, (1,1,1,1), 'replicate'), kernel_sum)[0,0]
```

### 6.2 Matrix-Free BidomainStencilOperator

```python
class BidomainStencilOperator:
    def __init__(self, Nx, Ny, dx, dy, sigma_i_field, sigma_e_field):
        # Precompute face conductivities for BOTH operators
        # Intracellular faces (for Ai)
        self.si_xx_e = harmonic_mean_faces_x(sigma_i_field['xx'])
        self.si_yy_n = harmonic_mean_faces_y(sigma_i_field['yy'])
        self.si_xy_e = arithmetic_mean_faces_x(sigma_i_field['xy'])
        self.si_xy_n = arithmetic_mean_faces_y(sigma_i_field['xy'])
        # Sum faces (for Asum)
        ss_xx = sigma_i_field['xx'] + sigma_e_field['xx']
        ss_yy = sigma_i_field['yy'] + sigma_e_field['yy']
        ss_xy = sigma_i_field['xy'] + sigma_e_field['xy']
        self.ss_xx_e = harmonic_mean_faces_x(ss_xx)
        # ... etc

    def apply_Ai(self, u):
        return self._apply_stencil(u, self.si_xx_e, self.si_yy_n, ...)

    def apply_Asum(self, u):
        return self._apply_stencil(u, self.ss_xx_e, self.ss_yy_n, ...)

    def _apply_stencil(self, u, sxx_e, syy_n, sxy_e, sxy_n):
        # FVM-style flux computation (zero-flux BC built in)
        ...
```

### 6.3 Memory Layout for Vm and phi_e

**Recommended: Separate arrays** (for operator splitting):
```
Vm:    shape (Nx, Ny), contiguous
phi_e: shape (Nx, Ny), contiguous
```

Each stencil operates on contiguous memory; cache-friendly. No benefit to interleaving when the parabolic and elliptic solves operate on one variable at a time. PyTorch conv2d expects `(batch, channel, H, W)` format.

### 6.4 Matrix-Free CG Solver for the Elliptic Equation

```python
def cg_solve_elliptic(apply_Asum, rhs, x0, tol=1e-6, max_iter=500):
    """CG with nullspace deflation for singular elliptic system."""
    x = x0.copy()
    r = rhs - apply_Asum(x)
    r -= r.mean()  # deflate nullspace
    p = r.copy()
    rsold = (r*r).sum()

    for it in range(max_iter):
        Ap = apply_Asum(p)
        Ap -= Ap.mean()  # deflate from operator output
        alpha = rsold / (p*Ap).sum()
        x += alpha * p
        r -= alpha * Ap
        rsnew = (r*r).sum()
        if rsnew**0.5 < tol: break
        p = r + (rsnew/rsold) * p
        rsold = rsnew

    x -= x.mean()  # enforce zero-mean solution
    return x
```

---

## 7. Boundary Conditions

### 7.1 Natural Neumann (Zero-Flux) -- Isolated Tissue

Standard for isolated cardiac tissue:
```
n . sigma_i . grad(phi_i) = 0   on dOmega  (intracellular insulated)
n . sigma_e . grad(phi_e) = 0   on dOmega  (extracellular insulated)
```

These together imply: `n . sigma_i . grad(Vm + phi_e) = 0` and `n . (sigma_i + sigma_e) . grad(phi_e) = 0`.

**FDM ghost-node elimination (isotropic)**:
```
Left boundary (i=0): u[-1,j] = u[1,j]   (mirror reflection)
Corner (i=0,j=0): u[-1,0] = u[1,0] AND u[0,-1] = u[0,1]
```

**FDM ghost-node (anisotropic)** -- the full zero-flux condition `sigma_xx*du/dx + sigma_xy*du/dy = 0`:
```
Left boundary: u[-1,j] = u[1,j] + (sigma_xy/sigma_xx)*(dx/dy)*(u[0,j+1]-u[0,j-1])
```

**FVM**: Simply set flux = 0 at boundary faces. No ghost nodes needed.

**FEM**: Homogeneous Neumann is the natural BC -- requires NO explicit treatment.

### 7.2 Compatibility of Boundary Conditions

The BCs on the parabolic and elliptic equations must be compatible. When both operators use zero-flux independently, the integral compatibility condition `integral_{dOmega} (sigma_i * grad(Vm^{n+1})) . n dS = 0` is automatically satisfied.

### 7.3 Bath-Loading Boundary Conditions

When tissue contacts conductive bath (experimental setup):

```
n . sigma_i . grad(phi_i) = 0              (intracellular insulated from bath)
phi_e = phi_bath                             (extracellular continuous with bath)
n . sigma_e . grad(phi_e) = n . sigma_b . grad(phi_bath)   (current continuity)
```

In the bath region: `div(sigma_b * grad(phi_bath)) = 0` (Laplace equation).

**Implementation**:
- Extend domain to include bath nodes
- Bath nodes have only phi_e (no Vm, no ionic model)
- Elliptic system grows: tissue nodes have Kie, bath nodes have sigma_b Laplacian
- Coupling at interface: potential continuity + flux continuity
- Biophysical effect: bath loading causes surface-leading-bulk wavefront curvature because the bath provides a low-resistance extracellular pathway near the surface (Bishop et al. 2011)

---

## Sources

### Key Papers
- Vigmond, Weber dos Santos et al. (2008) - Solvers for the Cardiac Bidomain Equations (PMC2881536)
- Vigmond, Aguel, Trayanova (2002) - Computational techniques for solving the bidomain equations in three dimensions
- Saleheen & Ng (1997) - Finite difference formulations for anisotropic bioelectric problems
- Saleheen & Ng (1998) - 3D finite-difference bidomain for inhomogeneous anisotropic cardiac tissues
- Pathmanathan et al. (2012) - Fully implicit FEM for bidomain electromechanics (PMC3501134)
- Marsh, Bhatt et al. (2020) - Composite BDF for the Bidomain Equations (PMC7767930)
- Pierre et al. (2012) - Accelerating Cardiac Bidomain Simulations Using GPUs (PMC3696513)
- Munteanu & Pavarino (2006) - Algebraic Multigrid Preconditioner for Cardiac Bidomain (PMC5428748)
- Coudiere et al. (2012) - Preconditioning the bidomain with almost linear complexity
- Bishop et al. (2011) - Bath-Loading Effects via Augmented Monodomain (PMC3075562)
- Bendahmane et al. (2010) - FV scheme for cardiac propagation
- Trew et al. (2005) - FVM for discontinuous electrical activation
- Schrader et al. (2023) - Anisotropic Diffusion Stencils

### Software
- openCARP Simulation Environment (opencarp.org)
- cbcbeat (FEniCS-based bidomain solver)
- MonoAlg3D (GPU FVM solver)
- TorchCor (PyTorch FEM cardiac solver)

### Project Internal
- `Research/openCARP_FDM_FVM/01_FDM_Stencils_and_Implementation.md` -- Monodomain FDM stencils (directly extensible to bidomain)
- `Research/openCARP_FDM_FVM/02_openCARP_FDM_FVM_Architecture.md` -- openCARP architecture, FVM flux computation, operator splitting
