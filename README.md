# Chapter 144: Physics-Informed Neural Networks for the Heston Stochastic Volatility Model

## Overview

The Black-Scholes model assumes constant volatility — a simplification that collapses under the weight of real market data. Implied volatility surfaces exhibit smiles, skews, and term structures that scream: volatility itself is stochastic. The **Heston model** (1993) captures this by letting variance follow its own mean-reverting stochastic process. However, solving the Heston PDE across the full (S, v, t) domain is computationally expensive, especially for real-time calibration.

**Physics-Informed Neural Networks (PINNs)** offer an elegant solution: train a neural network to approximate V(S, v, t) while enforcing the Heston PDE as a soft constraint in the loss function. The result is a mesh-free, differentiable solver that produces option prices, Greeks, and entire volatility surfaces in milliseconds.

**Key Insight:** By embedding the Heston PDE directly into the neural network's loss function, we get a solver that respects no-arbitrage constraints, handles complex boundary conditions, and generalizes across the entire (S, v, t) domain — all without a finite-difference grid.

## Trading Strategy

**Core Strategy:** Use a PINN-based Heston solver for real-time option pricing, volatility surface construction, and Greeks computation. Identify mispriced options by comparing PINN-calibrated theoretical prices against market quotes.

**Edge Factors:**
1. Sub-millisecond pricing enables real-time arbitrage detection across strikes and expiries
2. Smooth, differentiable Greeks (including cross-Greeks Vanna, Volga) improve hedging
3. The Heston model captures volatility clustering and mean-reversion observed in crypto markets
4. Calibration to the entire volatility surface simultaneously (not strike-by-strike)

**Target Assets:** Equity index options (SPX) and cryptocurrency options on Bybit (BTC, ETH)

## The Heston Stochastic Volatility Model

### Why Constant Volatility Fails

The Black-Scholes model assumes:

```
dS = μS dt + σS dW₁
```

where σ is constant. This produces flat implied volatility across strikes — contradicted by every options market ever observed. Real markets show:

- **Volatility smile:** OTM puts and calls are more expensive than ATM
- **Volatility skew:** OTM puts are more expensive than OTM calls (especially equities)
- **Term structure:** Short-dated options show steeper smiles than long-dated
- **Volatility clustering:** Periods of high vol beget high vol (GARCH effects)

### The Heston Dynamics

Heston (1993) models the asset price and its variance as a coupled system:

```
dS(t) = μ S(t) dt + √v(t) S(t) dW₁(t)

dv(t) = κ(θ - v(t)) dt + σ √v(t) dW₂(t)
```

where:
- **S(t)**: Spot price
- **v(t)**: Instantaneous variance (v = σ²_local)
- **κ**: Mean-reversion speed of variance
- **θ**: Long-run variance level
- **σ**: Volatility of variance (vol-of-vol)
- **ρ**: Correlation between dW₁ and dW₂ (typically ρ < 0 for equities)
- **μ**: Drift (under risk-neutral measure, μ = r - q)

**Correlation structure:**
```
E[dW₁ dW₂] = ρ dt
```

Negative correlation (ρ < 0) creates the **leverage effect**: when the stock drops, volatility rises — producing the characteristic volatility skew.

### The Feller Condition

For the variance process to remain strictly positive:

```
2κθ > σ²    (Feller condition)
```

When this holds, v(t) never touches zero. When violated, the variance can reach zero but is immediately reflected — the CIR process remains well-defined but requires careful numerical treatment.

**Typical parameter ranges:**

| Parameter | Equities | Crypto (BTC) |
|-----------|----------|--------------|
| κ         | 1-5      | 2-10         |
| θ         | 0.02-0.10| 0.30-1.50    |
| σ (vol-of-vol) | 0.2-0.8 | 0.5-3.0  |
| ρ         | -0.8 to -0.3 | -0.5 to 0.2 |
| v₀        | 0.01-0.10| 0.20-1.00    |

Note: Crypto markets exhibit much higher vol-of-vol and weaker (sometimes positive) correlation.

## The Heston PDE

Under the risk-neutral measure, the European option price V(S, v, t) satisfies:

```
∂V     1        ∂²V            ∂²V     1        ∂²V
── + ─ v S² ──── + ρσvS ──── + ─ σ²v ────
∂t     2       ∂S²           ∂S∂v     2       ∂v²

        ∂V                    ∂V
+ rS ── + κ(θ - v) ── - rV = 0
        ∂S                    ∂v
```

This is a **2D parabolic PDE** in the variables (S, v) with time t. Let us label each term:

| Term | Expression | Meaning |
|------|-----------|---------|
| Time decay | ∂V/∂t | Option value changes with time |
| Spot diffusion | ½vS²∂²V/∂S² | Gamma effect from spot movements |
| Mixed diffusion | ρσvS∂²V/∂S∂v | Cross-effect: spot-variance correlation |
| Variance diffusion | ½σ²v∂²V/∂v² | Volga effect from variance movements |
| Spot drift | rS∂V/∂S | Risk-neutral drift of spot |
| Variance drift | κ(θ-v)∂V/∂v | Mean-reversion pull on variance |
| Discounting | -rV | Risk-free discounting |

### Boundary Conditions

The PDE requires boundary conditions on the computational domain [0, S_max] x [0, v_max] x [0, T]:

**Terminal condition (at t = T):**
```
V(S, v, T) = max(S - K, 0)    (call)
V(S, v, T) = max(K - S, 0)    (put)
```

**At S = 0:**
```
V(0, v, t) = K e^{-r(T-t)}    (put: discounted strike)
V(0, v, t) = 0                 (call)
```

**As S → ∞:**
```
V(S, v, t) ≈ S e^{-q(T-t)}    (call: approaches forward)
V(S, v, t) → 0                 (put)
```

**At v = 0 (Feller boundary):**
The PDE degenerates. The boundary condition becomes:
```
∂V     ∂V              ∂V
── + rS── + κθ── - rV = 0
∂t     ∂S              ∂v
```
This is the reduced PDE with diffusion terms dropped.

**As v → ∞:**
```
V(S, v, t) → S e^{-q(T-t)}    (call: dominated by spot)
∂V/∂v → 0                      (Neumann: value insensitive to further vol increase)
```

## PINN Architecture for the Heston PDE

### Network Design

The PINN takes a 3D input (S, v, t) and outputs the option price V:

```
Input Layer:     (S, v, t) ∈ ℝ³
                      │
              ┌───────┴───────┐
              │  Normalization │    ← Scale inputs to [0, 1] or [-1, 1]
              │  S̃ = S/S_max   │
              │  ṽ = v/v_max   │
              │  t̃ = t/T       │
              └───────┬───────┘
                      │
              ┌───────┴───────┐
              │  Dense(3, 128) │
              │  Tanh          │
              └───────┬───────┘
                      │
              ┌───────┴───────┐
              │  Dense(128,128)│
              │  Tanh          │    × 4-6 hidden layers
              │  + Residual    │
              └───────┬───────┘
                      │
              ┌───────┴───────┐
              │  Dense(128, 1) │
              │  Softplus      │    ← Ensures V > 0
              └───────┬───────┘
                      │
              Output: V̂(S, v, t)
```

**Architecture choices:**
- **Activation:** Tanh (smooth, infinitely differentiable — needed for second-order derivatives)
- **Output activation:** Softplus ensures non-negative prices
- **Residual connections:** Stabilize training for deeper networks
- **Input normalization:** Critical for balancing gradients across S, v, t scales

### Input Feature Engineering

For better conditioning, we use log-spot and normalized inputs:

```python
x = log(S / K)       # Log-moneyness (centered at ATM)
v_norm = v / v_max    # Normalized variance
tau = (T - t) / T     # Normalized time-to-expiry
```

This naturally handles the wide range of spot prices and focuses the network around the important at-the-money region.

## Loss Function Construction

The PINN loss has four components:

```
L_total = λ_pde · L_pde + λ_bc · L_bc + λ_ic · L_ic + λ_data · L_data
```

### 1. PDE Residual Loss

Sample N_pde collocation points (S_i, v_i, t_i) in the interior domain and compute the PDE residual using automatic differentiation:

```python
def heston_pde_residual(model, S, v, t, params):
    """Compute Heston PDE residual at collocation points."""
    S.requires_grad_(True)
    v.requires_grad_(True)
    t.requires_grad_(True)

    V = model(S, v, t)

    # First-order derivatives
    V_t = grad(V, t)
    V_S = grad(V, S)
    V_v = grad(V, v)

    # Second-order derivatives
    V_SS = grad(V_S, S)
    V_vv = grad(V_v, v)
    V_Sv = grad(V_S, v)    # Mixed derivative

    kappa, theta, sigma, rho, r = params

    residual = (V_t
                + 0.5 * v * S**2 * V_SS
                + rho * sigma * v * S * V_Sv
                + 0.5 * sigma**2 * v * V_vv
                + r * S * V_S
                + kappa * (theta - v) * V_v
                - r * V)

    return torch.mean(residual**2)
```

### 2. Boundary Condition Losses

```python
def boundary_loss(model, K, r, T, params):
    """Enforce boundary conditions."""
    loss = 0.0

    # Terminal condition: V(S, v, T) = max(S - K, 0)
    S_term = torch.linspace(0, S_max, N_bc)
    v_term = torch.linspace(0, v_max, N_bc)
    S_grid, v_grid = torch.meshgrid(S_term, v_term)
    t_T = torch.full_like(S_grid, T)
    V_pred = model(S_grid, v_grid, t_T)
    V_exact = torch.maximum(S_grid - K, torch.zeros_like(S_grid))
    loss += torch.mean((V_pred - V_exact)**2)

    # S = 0 boundary (call)
    v_bc = torch.linspace(0, v_max, N_bc)
    t_bc = torch.linspace(0, T, N_bc)
    v_g, t_g = torch.meshgrid(v_bc, t_bc)
    S_zero = torch.zeros_like(v_g)
    V_pred_S0 = model(S_zero, v_g, t_g)
    loss += torch.mean(V_pred_S0**2)  # Call = 0 at S=0

    # v = 0 boundary (degenerate PDE)
    S_bc = torch.linspace(0, S_max, N_bc)
    t_bc = torch.linspace(0, T, N_bc)
    S_g, t_g = torch.meshgrid(S_bc, t_bc)
    v_zero = torch.zeros_like(S_g)
    loss += feller_boundary_loss(model, S_g, v_zero, t_g, params)

    return loss
```

### 3. Data Loss (Calibration)

When market option prices are available:

```python
def data_loss(model, S_market, v_market, t_market, V_market):
    """Fit to observed option prices."""
    V_pred = model(S_market, v_market, t_market)
    return torch.mean((V_pred - V_market)**2)
```

### 4. Handling the Mixed Derivative

The mixed derivative ∂²V/∂S∂v requires careful computation with autograd:

```python
def mixed_derivative(V, S, v):
    """Compute ∂²V/∂S∂v using two sequential grad calls."""
    # First: ∂V/∂S
    V_S = torch.autograd.grad(
        V, S, grad_outputs=torch.ones_like(V),
        create_graph=True, retain_graph=True
    )[0]

    # Second: ∂(∂V/∂S)/∂v
    V_Sv = torch.autograd.grad(
        V_S, v, grad_outputs=torch.ones_like(V_S),
        create_graph=True, retain_graph=True
    )[0]

    return V_Sv
```

The key is `create_graph=True` — this builds a computation graph for the derivative itself, enabling higher-order differentiation.

## Multi-Scale Collocation Strategy

Option prices vary dramatically across the (S, v, t) domain. Near the strike (S ≈ K) and at expiry (t ≈ T), gradients are steep. We use adaptive collocation:

```
Collocation Point Distribution:
┌─────────────────────────────────────────────────┐
│           v_max                                  │
│  ┌─────────────────────────────────────────┐    │
│  │  Sparse points        ·   ·   ·        │    │
│  │     ·   ·   ·   ·   ·   ·   ·         │    │
│  │  ·   ·   ·   ·  ···  ·   ·   ·        │    │
│  │     ·   ·  ·····████·····  ·   ·       │    │
│  │  ·   · ····█████████████···· ·         │    │
│  │     ·····██████ATM██████████···        │    │  Dense near
│  │  ·····███████████████████████····      │    │  strike K
│  │     ···██████████████████████···       │    │  and low v
│  │  ·   ····████████████████····  ·       │    │
│  │     ·   ·····████████·····   ·         │    │
│  │  ·   ·   ·  ·····  ·   ·   ·          │    │
│  └─────────────────────────────────────────┘    │
│  S=0              S=K              S_max         │
└─────────────────────────────────────────────────┘
```

```python
def generate_collocation_points(S_max, v_max, T, K, N_total):
    """Generate multi-scale collocation points."""
    N_uniform = N_total // 2
    N_atm = N_total // 4
    N_low_v = N_total // 4

    # Uniform background
    S_unif = torch.rand(N_uniform) * S_max
    v_unif = torch.rand(N_uniform) * v_max
    t_unif = torch.rand(N_uniform) * T

    # Dense near ATM (S ≈ K)
    S_atm = K + 0.2 * K * torch.randn(N_atm)
    S_atm = torch.clamp(S_atm, 0, S_max)
    v_atm = torch.rand(N_atm) * v_max
    t_atm = torch.rand(N_atm) * T

    # Dense near low variance
    S_low = torch.rand(N_low_v) * S_max
    v_low = torch.rand(N_low_v) * v_max * 0.3  # Focus on low v
    t_low = torch.rand(N_low_v) * T

    S = torch.cat([S_unif, S_atm, S_low])
    v = torch.cat([v_unif, v_atm, v_low])
    t = torch.cat([t_unif, t_atm, t_low])

    return S, v, t
```

## Semi-Analytical Heston Pricing

### Characteristic Function Approach

The Heston model has a semi-closed-form solution via the characteristic function. The call price is:

```
C(S, v, t) = S·P₁ - K·e^{-rτ}·P₂
```

where P₁ and P₂ are computed via inverse Fourier transforms:

```
       1   1  ∞  e^{-iφ ln K} f_j(φ)
P_j = ─ + ─ ∫  Re[ ───────────────── ] dφ,   j = 1, 2
       2   π  0         iφ
```

The characteristic function f_j(φ) is:

```
f_j(φ) = exp(C_j(τ, φ) + D_j(τ, φ)·v₀ + iφ·ln S)
```

with:

```
C_j = r·iφ·τ + (κθ/σ²)·[(b_j - ρσiφ + d_j)·τ - 2·ln((1 - g_j·e^{d_j·τ})/(1 - g_j))]

D_j = ((b_j - ρσiφ + d_j)/σ²)·((1 - e^{d_j·τ})/(1 - g_j·e^{d_j·τ}))
```

where:

```
d_j = √((ρσiφ - b_j)² - σ²(2u_j·iφ - φ²))

g_j = (b_j - ρσiφ + d_j)/(b_j - ρσiφ - d_j)

u₁ = 0.5,  u₂ = -0.5
b₁ = κ + λ - ρσ,  b₂ = κ + λ
```

Here λ is the market price of variance risk (often set to 0 for simplicity).

### Numerical Integration

```python
def heston_call_price(S, K, T, r, v0, kappa, theta, sigma, rho, N=256):
    """Semi-analytical Heston call price via characteristic function."""
    def integrand(phi, j):
        if j == 1:
            u, b = 0.5, kappa - rho * sigma
        else:
            u, b = -0.5, kappa

        d = np.sqrt((rho * sigma * 1j * phi - b)**2
                     - sigma**2 * (2 * u * 1j * phi - phi**2))
        g = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)

        C = r * 1j * phi * T + (kappa * theta / sigma**2) * (
            (b - rho * sigma * 1j * phi + d) * T
            - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))
        )
        D = ((b - rho * sigma * 1j * phi + d) / sigma**2) * (
            (1 - np.exp(d * T)) / (1 - g * np.exp(d * T))
        )

        f = np.exp(C + D * v0 + 1j * phi * np.log(S))
        return np.real(np.exp(-1j * phi * np.log(K)) * f / (1j * phi))

    # Gauss-Laguerre quadrature
    phi_values = np.linspace(1e-8, 100, N)
    dphi = phi_values[1] - phi_values[0]

    P1 = 0.5 + (1/np.pi) * np.sum([integrand(phi, 1) * dphi for phi in phi_values])
    P2 = 0.5 + (1/np.pi) * np.sum([integrand(phi, 2) * dphi for phi in phi_values])

    return S * P1 - K * np.exp(-r * T) * P2
```

## Greeks via Automatic Differentiation

One of the most powerful features of PINNs: Greeks come for free via autograd.

### Standard Greeks

```python
def compute_greeks(model, S, v, t):
    """Compute all Greeks using autograd."""
    S = S.requires_grad_(True)
    v = v.requires_grad_(True)
    t = t.requires_grad_(True)

    V = model(S, v, t)

    # First-order Greeks
    Delta = torch.autograd.grad(V, S, create_graph=True)[0]
    Vega = torch.autograd.grad(V, v, create_graph=True)[0]   # ∂V/∂v
    Theta = torch.autograd.grad(V, t, create_graph=True)[0]   # ∂V/∂t

    # Second-order Greeks
    Gamma = torch.autograd.grad(Delta, S, create_graph=True)[0]  # ∂²V/∂S²
    Vanna = torch.autograd.grad(Delta, v, create_graph=True)[0]  # ∂²V/∂S∂v
    Volga = torch.autograd.grad(Vega, v, create_graph=True)[0]   # ∂²V/∂v²

    return {
        'delta': Delta, 'gamma': Gamma, 'theta': Theta,
        'vega': Vega, 'vanna': Vanna, 'volga': Volga
    }
```

### Why Vanna and Volga Matter

**Vanna (∂²V/∂S∂v):** Measures how Delta changes with volatility. Critical for:
- Hedging in stochastic volatility environments
- Understanding how spot moves affect vol exposure
- Risk management of vol-spot correlation

**Volga (∂²V/∂v²):** Measures convexity of option price with respect to variance. Important for:
- Volatility smile dynamics
- Pricing exotic options with vol-of-vol exposure
- Understanding the "smile premium"

```
Greek Sensitivities in (S, v) Space:

Delta ∂V/∂S          Gamma ∂²V/∂S²         Vanna ∂²V/∂S∂v
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│       ╱─────│      │      ╱╲     │      │    ╱╲       │
│     ╱       │      │    ╱    ╲   │      │  ╱    ╲     │
│   ╱         │      │  ╱      ╲  │      │╱        ╲   │
│ ╱           │      │╱          ╲│      │            ╲ │
│─────────────│      │─────────────│      │──────╳──────│
└─────────────┘      └─────────────┘      └─────────────┘
  0     K   S_max      0    K   S_max       0    K   S_max
```

## Volatility Smile and Skew Generation

### From PINN Prices to Implied Volatility

Given PINN-predicted prices V(S, v, t) for various strikes K, we can invert the Black-Scholes formula to get implied volatility:

```python
def implied_vol_from_pinn(model, S, v0, t, strikes, r, T):
    """Generate implied volatility smile from PINN prices."""
    ivs = []
    for K in strikes:
        V_pinn = model(S, v0, t).item()
        iv = black_scholes_implied_vol(V_pinn, S, K, T - t, r)
        ivs.append(iv)
    return np.array(ivs)
```

### Smile Dynamics

The Heston model produces characteristic smile patterns:

```
Implied Volatility vs Strike (Smile/Skew):

IV(%)
 40 │
    │╲                              ← OTM puts
 35 │  ╲                              (high IV)
    │    ╲
 30 │      ╲
    │        ╲_____
 25 │               ╲___
    │                    ╲___         ← ATM
 20 │                        ╲___
    │                             ╲___
 15 │                                  ╲── OTM calls
    │                                     (lower IV)
 10 │
    └──────────┬──────────┬──────────┬────────────────
              0.8K        K        1.2K         Strike

Impact of Heston Parameters on Smile:
─── ρ = -0.7 (steep skew, equity-like)
--- ρ = -0.3 (moderate skew)
··· ρ = 0.0  (symmetric smile)
─·─ ρ = +0.3 (reverse skew, some crypto)
```

## Calibration to Market Data

### Calibration Objective

Given N market implied volatilities σ_mkt(K_i, T_j), find Heston parameters Θ = (κ, θ, σ, ρ, v₀) that minimize:

```
min_Θ  Σᵢⱼ wᵢⱼ · (σ_model(K_i, T_j; Θ) - σ_mkt(K_i, T_j))²
```

subject to:
```
κ > 0, θ > 0, σ > 0, -1 < ρ < 1, v₀ > 0
2κθ > σ²    (Feller condition)
```

### PINN-Accelerated Calibration

Traditional calibration requires repeatedly solving the Heston PDE (or evaluating the characteristic function) for each parameter guess. With a PINN:

1. **Train once** on a parametric family: V(S, v, t; κ, θ, σ, ρ)
2. **Calibrate instantly** by gradient descent on the parameters through the trained network

```python
def calibrate_heston_pinn(model, market_data, initial_params):
    """Calibrate Heston parameters using PINN as the forward solver."""
    params = torch.tensor(initial_params, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=0.01)

    for epoch in range(1000):
        optimizer.zero_grad()

        kappa, theta, sigma, rho, v0 = params

        # Enforce constraints
        kappa_c = torch.abs(kappa)
        theta_c = torch.abs(theta)
        sigma_c = torch.abs(sigma)
        rho_c = torch.tanh(rho)  # Maps to (-1, 1)
        v0_c = torch.abs(v0)

        total_loss = 0
        for K, T, iv_market in market_data:
            V_model = model(S, v0_c, 0, kappa_c, theta_c, sigma_c, rho_c)
            iv_model = bs_implied_vol(V_model, S, K, T, r)
            total_loss += (iv_model - iv_market)**2

        total_loss.backward()
        optimizer.step()

    return params.detach()
```

## Application to Crypto (Bybit)

### Why Heston for Crypto?

Cryptocurrency options markets exhibit extreme volatility characteristics:

1. **Higher vol-of-vol (σ):** BTC variance can spike 10x in hours
2. **Weaker leverage effect:** ρ is closer to 0 or even positive for some assets
3. **Faster mean-reversion (κ):** Vol spikes revert quickly
4. **Higher baseline variance (θ):** BTC annualized vol ≈ 60-80% vs SPX ≈ 15-20%
5. **Jump components:** Heston alone may not capture flash crashes (Bates extension)

### Bybit Data Integration

```python
import requests

def fetch_bybit_options(symbol="BTC"):
    """Fetch option chain from Bybit."""
    url = "https://api.bybit.com/v5/market/tickers"
    params = {"category": "option", "baseCoin": symbol}
    response = requests.get(url, params=params)
    data = response.json()

    options = []
    for item in data['result']['list']:
        options.append({
            'symbol': item['symbol'],
            'bid': float(item['bid1Price']),
            'ask': float(item['ask1Price']),
            'mark_price': float(item['markPrice']),
            'implied_vol': float(item['markIv']),
            'delta': float(item['delta']),
            'gamma': float(item['gamma']),
            'vega': float(item['vega']),
            'theta': float(item['theta']),
        })
    return options
```

## Comparison: PINN vs Semi-Analytical Heston

```
┌─────────────────────┬───────────────┬──────────────┬──────────────┐
│ Feature             │ Characteristic│ Finite       │ PINN         │
│                     │ Function      │ Difference   │              │
├─────────────────────┼───────────────┼──────────────┼──────────────┤
│ Single price eval   │ ~1 ms         │ ~100 ms      │ ~0.01 ms     │
│ Full surface (100²) │ ~10 s         │ ~10 s        │ ~1 ms        │
│ Greeks              │ Finite diff   │ Grid interp  │ Autograd     │
│ Accuracy            │ Machine eps   │ Grid-dep     │ ~1e-4        │
│ Training cost       │ N/A           │ N/A          │ Minutes-hours│
│ Mesh-free           │ Yes           │ No           │ Yes          │
│ Handles exotics     │ Limited       │ Yes          │ Yes          │
│ Parameter sensitivity│ Recompute   │ Recompute    │ Backprop     │
└─────────────────────┴───────────────┴──────────────┴──────────────┘
```

## Training Protocol

### Full Training Loop

```python
def train_heston_pinn(model, heston_params, domain, epochs=10000):
    """Complete training loop for Heston PINN."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    kappa, theta, sigma, rho, r = heston_params
    S_max, v_max, T = domain
    K = S_max / 2  # ATM strike

    # Loss weights (adaptive)
    lambda_pde = 1.0
    lambda_bc = 10.0
    lambda_ic = 10.0

    history = {'loss': [], 'pde': [], 'bc': [], 'ic': []}

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Generate fresh collocation points each epoch
        S_col, v_col, t_col = generate_collocation_points(
            S_max, v_max, T, K, N_total=4096
        )

        # PDE residual loss
        L_pde = heston_pde_residual(model, S_col, v_col, t_col, heston_params)

        # Boundary condition loss
        L_bc = boundary_loss(model, K, r, T, heston_params)

        # Initial/terminal condition loss
        L_ic = terminal_condition_loss(model, K, S_max, v_max, T)

        # Total loss
        loss = lambda_pde * L_pde + lambda_bc * L_bc + lambda_ic * L_ic

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Adaptive weight adjustment
        if epoch % 100 == 0:
            with torch.no_grad():
                grad_pde = compute_grad_norm(model, L_pde)
                grad_bc = compute_grad_norm(model, L_bc)
                lambda_bc = grad_pde / (grad_bc + 1e-8)

            print(f"Epoch {epoch}: Loss={loss:.6f}, "
                  f"PDE={L_pde:.6f}, BC={L_bc:.6f}")

    return model, history
```

### Training Tips

1. **Start with simple cases:** Train on Black-Scholes first (σ = 0 in Heston), then increase vol-of-vol
2. **Curriculum learning:** Begin with long-dated options, add shorter expiries progressively
3. **Gradient balancing:** Monitor gradient norms of each loss component
4. **Residual-based adaptive refinement:** Add collocation points where residual is large
5. **Transfer learning:** Use a trained PINN from one parameter set as initialization for nearby parameters

## Backtesting Volatility Trading Strategy

### Strategy Logic

```python
def vol_trading_strategy(pinn_model, market_data, heston_params):
    """
    Strategy: Compare PINN-calibrated fair value to market price.
    If PINN price > market ask → BUY (underpriced vol)
    If PINN price < market bid → SELL (overpriced vol)
    Delta-hedge to isolate vol exposure.
    """
    signals = []
    for option in market_data:
        S, K, T, r = option['spot'], option['strike'], option['expiry'], option['rate']
        v0 = heston_params['v0']

        V_model = pinn_model(S, v0, 0).item()

        # Compare to market
        if V_model > option['ask'] * 1.02:  # 2% threshold
            signals.append({'action': 'BUY', 'option': option,
                          'edge': V_model - option['ask']})
        elif V_model < option['bid'] * 0.98:
            signals.append({'action': 'SELL', 'option': option,
                          'edge': option['bid'] - V_model})

    return signals
```

## Project Structure

```
144_pinn_heston_model/
├── README.md                    ← This file
├── README.ru.md                 ← Russian translation
├── readme.simple.md             ← Simple explanation (English)
├── readme.simple.ru.md          ← Simple explanation (Russian)
├── python/
│   ├── __init__.py
│   ├── requirements.txt
│   ├── heston_pinn.py           ← PINN model architecture
│   ├── train.py                 ← Training loop
│   ├── data_loader.py           ← Market data (stocks + Bybit)
│   ├── heston_analytical.py     ← Semi-analytical Heston pricing
│   ├── calibration.py           ← Calibrate Heston to market IV
│   ├── greeks.py                ← Greeks via autograd
│   ├── visualize.py             ← Volatility surfaces and plots
│   └── backtest.py              ← Volatility trading backtest
└── rust_pinn_heston/
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs               ← Core PINN for Heston
    │   └── bin/
    │       ├── train.rs          ← Training binary
    │       ├── price_options.rs  ← Price options via PINN
    │       ├── fetch_data.rs     ← Fetch Bybit data
    │       └── calibrate.rs      ← Calibration binary
    └── examples/
        └── basic_pricing.rs      ← Simple pricing example
```

## Running the Code

### Python

```bash
cd 144_pinn_heston_model/python
pip install -r requirements.txt

# Train the PINN
python train.py --epochs 10000 --lr 1e-3

# Calibrate to market data
python calibration.py --source bybit --symbol BTC

# Compute Greeks
python greeks.py --spot 100 --strike 100 --expiry 0.25

# Visualize volatility surface
python visualize.py --show-smile --show-surface

# Run backtest
python backtest.py --strategy vol_arb --period 30d
```

### Rust

```bash
cd 144_pinn_heston_model/rust_pinn_heston

# Fetch market data
cargo run --bin fetch_data -- --symbol BTCUSDT --exchange bybit

# Train the PINN
cargo run --bin train -- --epochs 5000 --hidden-dim 128

# Price options
cargo run --bin price_options -- --spot 50000 --strike 50000 --expiry 0.25

# Calibrate to market
cargo run --bin calibrate -- --source bybit --symbol BTC
```

## Key Takeaways

1. **The Heston model** captures stochastic volatility with five parameters (κ, θ, σ, ρ, v₀) and produces realistic volatility smiles/skews

2. **PINNs solve the Heston PDE** without a mesh, producing a differentiable approximation V(S, v, t) that respects the physics

3. **The mixed derivative** ∂²V/∂S∂v is the key technical challenge — handled naturally by automatic differentiation

4. **Greeks come for free** via backpropagation, including challenging cross-Greeks (Vanna, Volga)

5. **Crypto markets** demand higher vol-of-vol parameters and faster mean-reversion speeds

6. **Calibration** can be accelerated by differentiating through the PINN with respect to model parameters

7. **Real-time applications** become feasible: once trained, the PINN evaluates in microseconds

## References

1. Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options." *The Review of Financial Studies*, 6(2), 327-343.

2. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations." *Journal of Computational Physics*, 378, 686-707.

3. Bayer, C., & Stemper, B. (2018). "Deep Calibration of Rough Stochastic Volatility Models." arXiv:1810.03399.

4. Salvador, B., Oosterlee, C. W., & van der Meer, R. (2020). "Financial Option Valuation by Unsupervised Learning with Artificial Neural Networks." *Mathematics*, 9(1), 46.

5. Alfonsi, A. (2015). *Affine Diffusions and Related Processes: Simulation, Theory and Applications.* Springer.

6. Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide.* Wiley.
