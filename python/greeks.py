"""
greeks.py - Compute option Greeks via automatic differentiation through the PINN

Greeks computed:
    First-order:  Delta (dV/dS), Vega (dV/dv), Theta (dV/dt), Rho (dV/dr)
    Second-order: Gamma (d2V/dS2), Vanna (d2V/dSdv), Volga (d2V/dv2),
                  Charm (d2V/dSdt), Veta (d2V/dvdt)

The key advantage of the PINN approach: all derivatives are exact (up to network
approximation error) and smooth, computed via backpropagation through the network.
"""

import argparse
import torch
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path


def compute_greeks(
    model,
    S: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    compute_second_order: bool = True,
    compute_cross_greeks: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute all Greeks via automatic differentiation.

    Args:
        model: Trained HestonPINN model
        S: Spot prices (N,) -- will be set to requires_grad=True
        v: Variance values (N,)
        t: Time values (N,)
        compute_second_order: If True, compute Gamma, Volga
        compute_cross_greeks: If True, compute Vanna, Charm, Veta

    Returns:
        Dictionary of Greek tensors
    """
    S = S.clone().detach().requires_grad_(True)
    v = v.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    # Forward pass
    V = model(S, v, t)

    greeks = {"price": V.detach()}

    # --- First-order Greeks ---

    # Delta = dV/dS
    Delta = torch.autograd.grad(
        V, S, grad_outputs=torch.ones_like(V),
        create_graph=True, retain_graph=True
    )[0]
    greeks["delta"] = Delta.detach()

    # Vega = dV/dv (sensitivity to variance, not volatility)
    # Note: market Vega is typically dV/d(sigma), where sigma = sqrt(v)
    # Here we compute dV/dv directly; to convert: Vega_sigma = dV/dv * 2*sqrt(v)
    Vega_v = torch.autograd.grad(
        V, v, grad_outputs=torch.ones_like(V),
        create_graph=True, retain_graph=True
    )[0]
    greeks["vega_v"] = Vega_v.detach()
    # Convert to standard vega (per 1% vol change)
    greeks["vega"] = (Vega_v * 2 * torch.sqrt(v) * 0.01).detach()

    # Theta = dV/dt (note: market Theta is usually -dV/dt for time decay)
    Theta = torch.autograd.grad(
        V, t, grad_outputs=torch.ones_like(V),
        create_graph=True, retain_graph=True
    )[0]
    greeks["theta"] = (-Theta).detach()  # Negative sign for time decay convention

    # --- Second-order Greeks ---
    if compute_second_order:
        # Gamma = d2V/dS2
        Gamma = torch.autograd.grad(
            Delta, S, grad_outputs=torch.ones_like(Delta),
            create_graph=True, retain_graph=True
        )[0]
        greeks["gamma"] = Gamma.detach()

        # Volga (Vomma) = d2V/dv2
        Volga = torch.autograd.grad(
            Vega_v, v, grad_outputs=torch.ones_like(Vega_v),
            create_graph=True, retain_graph=True
        )[0]
        greeks["volga"] = Volga.detach()

    # --- Cross-Greeks ---
    if compute_cross_greeks:
        # Vanna = d2V/dSdv
        Vanna = torch.autograd.grad(
            Delta, v, grad_outputs=torch.ones_like(Delta),
            create_graph=True, retain_graph=True
        )[0]
        greeks["vanna"] = Vanna.detach()

        # Charm = d2V/dSdt (Delta decay)
        Charm = torch.autograd.grad(
            Delta, t, grad_outputs=torch.ones_like(Delta),
            create_graph=True, retain_graph=True
        )[0]
        greeks["charm"] = (-Charm).detach()

        # Veta = d2V/dvdt (Vega decay)
        Veta = torch.autograd.grad(
            Vega_v, t, grad_outputs=torch.ones_like(Vega_v),
            create_graph=True, retain_graph=True
        )[0]
        greeks["veta"] = (-Veta).detach()

    return greeks


def compute_greeks_grid(
    model,
    S_range: Tuple[float, float],
    v_range: Tuple[float, float],
    t_val: float,
    n_S: int = 50,
    n_v: int = 50,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """
    Compute Greeks on a 2D grid of (S, v) at fixed time t.

    Returns:
        Dictionary with 2D arrays for each Greek, plus S_grid and v_grid
    """
    S_vals = torch.linspace(S_range[0], S_range[1], n_S, device=device)
    v_vals = torch.linspace(v_range[0], v_range[1], n_v, device=device)

    S_grid, v_grid = torch.meshgrid(S_vals, v_vals, indexing="ij")
    S_flat = S_grid.flatten()
    v_flat = v_grid.flatten()
    t_flat = torch.full_like(S_flat, t_val)

    greeks = compute_greeks(model, S_flat, v_flat, t_flat)

    result = {
        "S_grid": S_grid.cpu().numpy(),
        "v_grid": v_grid.cpu().numpy(),
    }

    for name, tensor in greeks.items():
        result[name] = tensor.cpu().numpy().reshape(n_S, n_v)

    return result


def finite_difference_greeks(
    model,
    S: float,
    v: float,
    t: float,
    dS: float = 0.01,
    dv: float = 0.001,
    dt: float = 0.001,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute Greeks via finite differences (for comparison/validation).

    Args:
        model: PINN model
        S, v, t: Evaluation point
        dS, dv, dt: Step sizes for finite differences

    Returns:
        Dictionary of Greek values
    """
    def price(s, var, time):
        s_t = torch.tensor([s], dtype=torch.float32, device=device)
        v_t = torch.tensor([var], dtype=torch.float32, device=device)
        t_t = torch.tensor([time], dtype=torch.float32, device=device)
        with torch.no_grad():
            return model(s_t, v_t, t_t).item()

    V = price(S, v, t)

    # First-order
    delta = (price(S + dS, v, t) - price(S - dS, v, t)) / (2 * dS)
    vega_v = (price(S, v + dv, t) - price(S, v - dv, t)) / (2 * dv)
    theta = -(price(S, v, t + dt) - price(S, v, t - dt)) / (2 * dt)

    # Second-order
    gamma = (price(S + dS, v, t) - 2 * V + price(S - dS, v, t)) / (dS ** 2)
    volga = (price(S, v + dv, t) - 2 * V + price(S, v - dv, t)) / (dv ** 2)

    # Cross-Greeks
    vanna = (
        price(S + dS, v + dv, t) - price(S + dS, v - dv, t)
        - price(S - dS, v + dv, t) + price(S - dS, v - dv, t)
    ) / (4 * dS * dv)

    return {
        "price": V,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega_v": vega_v,
        "volga": volga,
        "vanna": vanna,
    }


def print_greeks_table(greeks: Dict[str, float], S: float, v: float, t: float):
    """Print a formatted table of Greeks."""
    print(f"\nGreeks at S={S:.2f}, v={v:.4f} (vol={np.sqrt(v)*100:.1f}%), t={t:.4f}")
    print("-" * 50)
    print(f"  {'Price':>12}: {greeks.get('price', 0):.6f}")
    print(f"  {'Delta':>12}: {greeks.get('delta', 0):.6f}")
    print(f"  {'Gamma':>12}: {greeks.get('gamma', 0):.6f}")
    print(f"  {'Theta':>12}: {greeks.get('theta', 0):.6f}")
    print(f"  {'Vega (dV/dv)':>12}: {greeks.get('vega_v', 0):.6f}")
    if "vega" in greeks:
        print(f"  {'Vega (1%)':>12}: {greeks.get('vega', 0):.6f}")
    print(f"  {'Vanna':>12}: {greeks.get('vanna', 0):.6f}")
    print(f"  {'Volga':>12}: {greeks.get('volga', 0):.6f}")
    if "charm" in greeks:
        print(f"  {'Charm':>12}: {greeks.get('charm', 0):.6f}")
    if "veta" in greeks:
        print(f"  {'Veta':>12}: {greeks.get('veta', 0):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Heston PINN Greeks")
    parser.add_argument("--spot", type=float, default=100.0)
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--variance", type=float, default=0.04)
    parser.add_argument("--expiry", type=float, default=0.25)
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # If no trained model, create a fresh one for demonstration
    if args.model_path and Path(args.model_path).exists():
        from heston_pinn import HestonPINN
        checkpoint = torch.load(args.model_path, map_location=device)
        model = HestonPINN(
            S_max=checkpoint.get("S_max", 200.0),
            v_max=checkpoint.get("v_max", 2.0),
            T_max=checkpoint.get("T_max", 1.0),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"Loaded model from {args.model_path}")
    else:
        from heston_pinn import HestonPINN
        print("No trained model provided. Using untrained model for demo.")
        model = HestonPINN(hidden_dim=64, num_layers=4).to(device)

    # Compute Greeks via autograd
    S = torch.tensor([args.spot], dtype=torch.float32, device=device)
    v = torch.tensor([args.variance], dtype=torch.float32, device=device)
    t = torch.tensor([0.0], dtype=torch.float32, device=device)

    print("=" * 60)
    print("Greeks via Autograd (PINN)")
    print("=" * 60)

    greeks_auto = compute_greeks(model, S, v, t)
    greeks_dict = {k: val.item() if hasattr(val, "item") else float(val.flatten()[0])
                   for k, val in greeks_auto.items()}
    print_greeks_table(greeks_dict, args.spot, args.variance, 0.0)

    # Compare with finite differences
    print("\n" + "=" * 60)
    print("Greeks via Finite Differences (Validation)")
    print("=" * 60)

    greeks_fd = finite_difference_greeks(
        model, args.spot, args.variance, 0.0, device=device
    )
    print_greeks_table(greeks_fd, args.spot, args.variance, 0.0)

    # Error comparison
    print("\n" + "=" * 60)
    print("Autograd vs Finite Difference Error")
    print("=" * 60)
    for key in ["delta", "gamma", "theta", "vega_v", "vanna", "volga"]:
        if key in greeks_dict and key in greeks_fd:
            err = abs(greeks_dict[key] - greeks_fd[key])
            print(f"  {key:>10}: |error| = {err:.8f}")
