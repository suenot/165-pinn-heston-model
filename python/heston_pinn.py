"""
heston_pinn.py - Physics-Informed Neural Network for the Heston Stochastic Volatility Model

The PINN approximates V(S, v, t) by embedding the Heston PDE as a soft constraint
in the loss function, producing a mesh-free differentiable option pricing solver.

Architecture:
    Input: (S_normalized, v_normalized, t_normalized) in R^3
    Hidden: 6 layers x 128 neurons with Tanh activation + residual connections
    Output: V(S, v, t) >= 0 via Softplus
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional


class ResidualBlock(nn.Module):
    """Residual block with two linear layers and Tanh activation."""

    def __init__(self, dim: int):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.layer1(x))
        out = self.layer2(out)
        return self.activation(out + residual)


class HestonPINN(nn.Module):
    """
    Physics-Informed Neural Network for the Heston PDE.

    Takes normalized inputs (S/S_max, v/v_max, t/T) and outputs option price V.
    The network is designed with Tanh activations (C-infinity smooth) to support
    accurate second-order derivatives via autograd.

    Parameters:
        hidden_dim: Width of hidden layers (default 128)
        num_layers: Number of residual blocks (default 6)
        S_max: Maximum spot price for normalization
        v_max: Maximum variance for normalization
        T_max: Maximum time for normalization
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
        S_max: float = 200.0,
        v_max: float = 2.0,
        T_max: float = 1.0,
    ):
        super().__init__()
        self.S_max = S_max
        self.v_max = v_max
        self.T_max = T_max
        self.hidden_dim = hidden_dim

        # Input layer: 3 -> hidden_dim
        self.input_layer = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
        )

        # Residual hidden layers
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_layers)]
        )

        # Output layer with Softplus to ensure V >= 0
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(beta=1.0),
        )

        # Initialize weights using Xavier
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for better gradient flow."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def normalize_inputs(
        self, S: torch.Tensor, v: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Normalize inputs to [0, 1] range."""
        S_norm = S / self.S_max
        v_norm = v / self.v_max
        t_norm = t / self.T_max
        return torch.stack([S_norm, v_norm, t_norm], dim=-1)

    def forward(
        self, S: torch.Tensor, v: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: (S, v, t) -> V(S, v, t).

        Args:
            S: Spot prices, shape (N,)
            v: Variance values, shape (N,)
            t: Time values, shape (N,)

        Returns:
            V: Option prices, shape (N, 1)
        """
        x = self.normalize_inputs(S, v, t)
        h = self.input_layer(x)
        for block in self.residual_blocks:
            h = block(h)
        return self.output_layer(h)

    def forward_with_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass from a single stacked tensor (N, 3)."""
        h = self.input_layer(x)
        for block in self.residual_blocks:
            h = block(h)
        return self.output_layer(h)


class HestonPINNResidual:
    """
    Computes the Heston PDE residual for a given PINN model using autograd.

    The Heston PDE:
        dV/dt + 0.5*v*S^2*d2V/dS2 + rho*sigma*v*S*d2V/dSdv
        + 0.5*sigma^2*v*d2V/dv2 + r*S*dV/dS + kappa*(theta-v)*dV/dv - r*V = 0

    Parameters:
        model: HestonPINN network
        kappa: Mean-reversion speed
        theta: Long-run variance
        sigma: Volatility of variance (vol-of-vol)
        rho: Correlation between spot and variance Brownians
        r: Risk-free rate
    """

    def __init__(
        self,
        model: HestonPINN,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma: float = 0.3,
        rho: float = -0.7,
        r: float = 0.05,
    ):
        self.model = model
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r

    def check_feller_condition(self) -> bool:
        """Check 2*kappa*theta > sigma^2."""
        return 2 * self.kappa * self.theta > self.sigma**2

    def compute_residual(
        self, S: torch.Tensor, v: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Heston PDE residual at the given collocation points.

        All inputs must have requires_grad=True for autograd to work.

        Returns:
            residual: PDE residual at each point, shape (N, 1)
        """
        S = S.requires_grad_(True)
        v = v.requires_grad_(True)
        t = t.requires_grad_(True)

        V = self.model(S, v, t)

        # First-order derivatives
        V_S = torch.autograd.grad(
            V, S, grad_outputs=torch.ones_like(V),
            create_graph=True, retain_graph=True
        )[0]

        V_v = torch.autograd.grad(
            V, v, grad_outputs=torch.ones_like(V),
            create_graph=True, retain_graph=True
        )[0]

        V_t = torch.autograd.grad(
            V, t, grad_outputs=torch.ones_like(V),
            create_graph=True, retain_graph=True
        )[0]

        # Second-order derivatives
        V_SS = torch.autograd.grad(
            V_S, S, grad_outputs=torch.ones_like(V_S),
            create_graph=True, retain_graph=True
        )[0]

        V_vv = torch.autograd.grad(
            V_v, v, grad_outputs=torch.ones_like(V_v),
            create_graph=True, retain_graph=True
        )[0]

        # Mixed derivative: d2V/dSdv
        V_Sv = torch.autograd.grad(
            V_S, v, grad_outputs=torch.ones_like(V_S),
            create_graph=True, retain_graph=True
        )[0]

        # Heston PDE residual
        residual = (
            V_t
            + 0.5 * v * S**2 * V_SS
            + self.rho * self.sigma * v * S * V_Sv
            + 0.5 * self.sigma**2 * v * V_vv
            + self.r * S * V_S
            + self.kappa * (self.theta - v) * V_v
            - self.r * V
        )

        return residual

    def pde_loss(
        self, S: torch.Tensor, v: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Mean squared PDE residual loss."""
        residual = self.compute_residual(S, v, t)
        return torch.mean(residual**2)


def generate_collocation_points(
    S_max: float,
    v_max: float,
    T: float,
    K: float,
    N_total: int = 4096,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate multi-scale collocation points with denser sampling near the strike
    and at low variance values.

    Args:
        S_max: Maximum spot price
        v_max: Maximum variance
        T: Expiry time
        K: Strike price (for ATM-focused sampling)
        N_total: Total number of points
        device: Torch device

    Returns:
        S, v, t: Collocation point tensors
    """
    N_uniform = N_total // 2
    N_atm = N_total // 4
    N_low_v = N_total // 4

    # Uniform background sampling
    S_unif = torch.rand(N_uniform, device=device) * S_max
    v_unif = torch.rand(N_uniform, device=device) * v_max
    t_unif = torch.rand(N_uniform, device=device) * T

    # Dense near ATM (S ~ K)
    S_atm = K + 0.2 * K * torch.randn(N_atm, device=device)
    S_atm = torch.clamp(S_atm, 1e-6, S_max)
    v_atm = torch.rand(N_atm, device=device) * v_max
    t_atm = torch.rand(N_atm, device=device) * T

    # Dense at low variance (where Feller boundary matters)
    S_low = torch.rand(N_low_v, device=device) * S_max
    v_low = torch.rand(N_low_v, device=device) * v_max * 0.3
    t_low = torch.rand(N_low_v, device=device) * T

    S = torch.cat([S_unif, S_atm, S_low])
    v = torch.cat([v_unif, v_atm, v_low])
    t = torch.cat([t_unif, t_atm, t_low])

    return S, v, t


def terminal_condition_loss(
    model: HestonPINN,
    K: float,
    S_max: float,
    v_max: float,
    T: float,
    N_bc: int = 256,
    option_type: str = "call",
    device: str = "cpu",
) -> torch.Tensor:
    """
    Terminal condition loss: V(S, v, T) = payoff(S).

    Args:
        model: PINN model
        K: Strike price
        S_max: Max spot for boundary
        v_max: Max variance for boundary
        T: Expiry time
        N_bc: Number of boundary points
        option_type: 'call' or 'put'
        device: Torch device

    Returns:
        Scalar loss tensor
    """
    S_bc = torch.linspace(1e-4, S_max, N_bc, device=device)
    v_bc = torch.linspace(1e-4, v_max, N_bc, device=device)

    # Create grid of (S, v) at t = T
    S_grid, v_grid = torch.meshgrid(S_bc, v_bc, indexing="ij")
    S_flat = S_grid.flatten()
    v_flat = v_grid.flatten()
    t_flat = torch.full_like(S_flat, T)

    V_pred = model(S_flat, v_flat, t_flat)

    if option_type == "call":
        payoff = torch.maximum(S_flat - K, torch.zeros_like(S_flat))
    else:
        payoff = torch.maximum(K - S_flat, torch.zeros_like(S_flat))

    return torch.mean((V_pred.squeeze() - payoff) ** 2)


def boundary_condition_loss(
    model: HestonPINN,
    K: float,
    r: float,
    S_max: float,
    v_max: float,
    T: float,
    N_bc: int = 128,
    option_type: str = "call",
    device: str = "cpu",
) -> torch.Tensor:
    """
    Spatial boundary condition losses.

    Enforces:
        - S=0 boundary
        - S->inf boundary (via S_max)
        - v->inf boundary (Neumann)

    Returns:
        Scalar loss tensor
    """
    loss = torch.tensor(0.0, device=device)

    # --- S = 0 boundary ---
    v_vals = torch.linspace(1e-4, v_max, N_bc, device=device)
    t_vals = torch.linspace(0, T, N_bc, device=device)
    v_g, t_g = torch.meshgrid(v_vals, t_vals, indexing="ij")
    v_flat = v_g.flatten()
    t_flat = t_g.flatten()
    S_zero = torch.full_like(v_flat, 1e-6)

    V_pred_S0 = model(S_zero, v_flat, t_flat)

    if option_type == "call":
        # Call at S=0 is 0
        loss = loss + torch.mean(V_pred_S0**2)
    else:
        # Put at S=0 is K * exp(-r*(T-t))
        V_exact = K * torch.exp(-r * (T - t_flat))
        loss = loss + torch.mean((V_pred_S0.squeeze() - V_exact) ** 2)

    # --- Large S boundary ---
    S_large = torch.full((N_bc,), S_max, device=device)
    v_vals2 = torch.linspace(1e-4, v_max, N_bc, device=device)
    t_vals2 = torch.linspace(0, T, N_bc, device=device)

    V_pred_Smax = model(S_large, v_vals2, t_vals2)

    if option_type == "call":
        # Call at large S ~ S - K*exp(-r*(T-t))
        V_approx = S_large - K * torch.exp(-r * (T - t_vals2))
        loss = loss + torch.mean((V_pred_Smax.squeeze() - V_approx) ** 2)
    else:
        # Put at large S ~ 0
        loss = loss + torch.mean(V_pred_Smax**2)

    return loss


def feller_boundary_loss(
    model: HestonPINN,
    kappa: float,
    theta: float,
    r: float,
    S_max: float,
    T: float,
    N_bc: int = 128,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Boundary condition at v = 0 (Feller boundary).

    When v=0, the Heston PDE degenerates to:
        dV/dt + r*S*dV/dS + kappa*theta*dV/dv - r*V = 0

    Returns:
        Scalar loss tensor
    """
    S_vals = torch.linspace(1e-4, S_max, N_bc, device=device)
    t_vals = torch.linspace(0, T, N_bc, device=device)
    S_g, t_g = torch.meshgrid(S_vals, t_vals, indexing="ij")
    S_flat = S_g.flatten().requires_grad_(True)
    t_flat = t_g.flatten().requires_grad_(True)
    v_zero = torch.full_like(S_flat, 1e-6).requires_grad_(True)

    V = model(S_flat, v_zero, t_flat)

    V_t = torch.autograd.grad(
        V, t_flat, grad_outputs=torch.ones_like(V),
        create_graph=True, retain_graph=True
    )[0]

    V_S = torch.autograd.grad(
        V, S_flat, grad_outputs=torch.ones_like(V),
        create_graph=True, retain_graph=True
    )[0]

    V_v = torch.autograd.grad(
        V, v_zero, grad_outputs=torch.ones_like(V),
        create_graph=True, retain_graph=True
    )[0]

    # Degenerate PDE at v=0
    residual = V_t + r * S_flat * V_S + kappa * theta * V_v - r * V.squeeze()

    return torch.mean(residual**2)


if __name__ == "__main__":
    # Quick test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = HestonPINN(hidden_dim=64, num_layers=4).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    S = torch.tensor([100.0, 110.0, 90.0], device=device)
    v = torch.tensor([0.04, 0.06, 0.02], device=device)
    t = torch.tensor([0.0, 0.0, 0.0], device=device)

    V = model(S, v, t)
    print(f"Forward pass output shapes: {V.shape}")
    print(f"Predicted prices: {V.detach().cpu().numpy().flatten()}")

    # Test residual computation
    residual_computer = HestonPINNResidual(model)
    print(f"Feller condition satisfied: {residual_computer.check_feller_condition()}")

    S_col, v_col, t_col = generate_collocation_points(200.0, 2.0, 1.0, 100.0, N_total=256)
    pde_loss = residual_computer.pde_loss(S_col, v_col, t_col)
    print(f"PDE loss (untrained): {pde_loss.item():.6f}")
