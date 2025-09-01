"""
visualize.py - Visualization tools for the Heston PINN

Generates:
    1. Implied volatility surface (strike x expiry)
    2. Volatility smile at fixed expiry
    3. Option price surface in (S, v) space
    4. Greeks heatmaps in (S, v) space
    5. Training loss curves
    6. PINN vs analytical comparison
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path
from typing import Dict, Optional

from heston_analytical import (
    heston_call_price,
    implied_volatility,
    heston_implied_vol_surface,
)


def plot_implied_vol_smile(
    S: float = 100.0,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma: float = 0.3,
    rho_values: list = None,
    r: float = 0.05,
    T: float = 0.25,
    save_path: Optional[str] = None,
):
    """
    Plot implied volatility smile for different rho values.

    Shows how correlation affects the skew.
    """
    if rho_values is None:
        rho_values = [-0.7, -0.3, 0.0, 0.3]

    strikes = np.linspace(0.7 * S, 1.3 * S, 50)
    moneyness = strikes / S

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for rho in rho_values:
        ivs = []
        for K in strikes:
            price = heston_call_price(S, K, T, r, v0, kappa, theta, sigma, rho)
            iv = implied_volatility(price, S, K, T, r)
            ivs.append(iv * 100 if not np.isnan(iv) else np.nan)

        label = f"rho = {rho:.1f}"
        if rho < -0.5:
            label += " (equity-like)"
        elif abs(rho) < 0.2:
            label += " (symmetric)"
        elif rho > 0.2:
            label += " (reverse skew)"

        ax.plot(moneyness, ivs, linewidth=2, label=label)

    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="ATM")
    ax.set_xlabel("Moneyness (K/S)", fontsize=12)
    ax.set_ylabel("Implied Volatility (%)", fontsize=12)
    ax.set_title(f"Heston Implied Volatility Smile (T={T}y, "
                 f"sigma={sigma}, kappa={kappa})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def plot_implied_vol_surface(
    S: float = 100.0,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma: float = 0.3,
    rho: float = -0.7,
    r: float = 0.05,
    save_path: Optional[str] = None,
):
    """Plot 3D implied volatility surface (strike x expiry)."""
    strikes = np.linspace(0.75 * S, 1.25 * S, 30)
    expiries = np.linspace(0.05, 1.0, 20)

    iv_surface = heston_implied_vol_surface(
        S, v0, kappa, theta, sigma, rho, r, strikes, expiries
    )

    K_grid, T_grid = np.meshgrid(strikes / S, expiries, indexing="ij")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        K_grid, T_grid, iv_surface * 100,
        cmap=cm.viridis, alpha=0.8, edgecolor="none"
    )

    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Time to Expiry (years)")
    ax.set_zlabel("Implied Volatility (%)")
    ax.set_title("Heston Implied Volatility Surface")
    fig.colorbar(surf, shrink=0.5, aspect=10, label="IV (%)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def plot_option_price_surface(
    model=None,
    K: float = 100.0,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma: float = 0.3,
    rho: float = -0.7,
    r: float = 0.05,
    T: float = 0.5,
    use_analytical: bool = True,
    save_path: Optional[str] = None,
):
    """
    Plot option price as a function of (S, v) at fixed time.

    Can use either the PINN model or analytical Heston.
    """
    S_vals = np.linspace(60, 140, 40)
    v_vals = np.linspace(0.01, 0.20, 40)

    S_grid, v_grid = np.meshgrid(S_vals, v_vals, indexing="ij")
    V_grid = np.zeros_like(S_grid)

    if use_analytical or model is None:
        for i in range(len(S_vals)):
            for j in range(len(v_vals)):
                V_grid[i, j] = heston_call_price(
                    S_vals[i], K, T, r, v_vals[j],
                    kappa, theta, sigma, rho
                )
    else:
        import torch
        device = next(model.parameters()).device
        S_flat = torch.tensor(S_grid.flatten(), dtype=torch.float32, device=device)
        v_flat = torch.tensor(v_grid.flatten(), dtype=torch.float32, device=device)
        t_flat = torch.zeros_like(S_flat)

        with torch.no_grad():
            V_flat = model(S_flat, v_flat, t_flat).cpu().numpy().flatten()
        V_grid = V_flat.reshape(S_grid.shape)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        S_grid, v_grid * 100, V_grid,
        cmap=cm.plasma, alpha=0.8, edgecolor="none"
    )

    ax.set_xlabel("Spot Price S")
    ax.set_ylabel("Variance v (%)")
    ax.set_zlabel("Call Price V")
    ax.set_title(f"Heston Call Price Surface (K={K}, T={T}y)")
    fig.colorbar(surf, shrink=0.5, aspect=10, label="Price")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def plot_greeks_heatmaps(
    K: float = 100.0,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma: float = 0.3,
    rho: float = -0.7,
    r: float = 0.05,
    T: float = 0.25,
    save_path: Optional[str] = None,
):
    """
    Plot heatmaps of Greeks in (S, v) space using finite differences on analytical Heston.
    """
    S_vals = np.linspace(70, 130, 30)
    v_vals = np.linspace(0.01, 0.15, 30)
    dS = 0.5
    dv = 0.001

    delta_grid = np.zeros((len(S_vals), len(v_vals)))
    gamma_grid = np.zeros_like(delta_grid)
    vanna_grid = np.zeros_like(delta_grid)
    volga_grid = np.zeros_like(delta_grid)

    for i, S in enumerate(S_vals):
        for j, v in enumerate(v_vals):
            p = lambda s, var: heston_call_price(s, K, T, r, var, kappa, theta, sigma, rho)

            V0 = p(S, v)
            delta_grid[i, j] = (p(S + dS, v) - p(S - dS, v)) / (2 * dS)
            gamma_grid[i, j] = (p(S + dS, v) - 2 * V0 + p(S - dS, v)) / (dS**2)
            vanna_grid[i, j] = (
                p(S + dS, v + dv) - p(S + dS, v - dv)
                - p(S - dS, v + dv) + p(S - dS, v - dv)
            ) / (4 * dS * dv)
            volga_grid[i, j] = (p(S, v + dv) - 2 * V0 + p(S, v - dv)) / (dv**2)

    S_grid, v_grid = np.meshgrid(S_vals, v_vals, indexing="ij")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    greeks_data = [
        ("Delta", delta_grid, "RdYlBu_r"),
        ("Gamma", gamma_grid, "hot"),
        ("Vanna", vanna_grid, "coolwarm"),
        ("Volga", volga_grid, "viridis"),
    ]

    for ax, (name, data, cmap) in zip(axes.flatten(), greeks_data):
        im = ax.contourf(S_grid, v_grid * 100, data, levels=20, cmap=cmap)
        ax.set_xlabel("Spot Price S")
        ax.set_ylabel("Variance v (%)")
        ax.set_title(f"{name} (K={K}, T={T}y)")
        ax.axvline(x=K, color="white", linestyle="--", alpha=0.7, linewidth=1)
        fig.colorbar(im, ax=ax)

    plt.suptitle("Heston Greeks Heatmaps", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def plot_training_history(
    history_path: str = "checkpoints/training_history.json",
    save_path: Optional[str] = None,
):
    """Plot training loss curves from saved history."""
    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total loss
    axes[0, 0].semilogy(history["epoch"], history["loss"], "b-", linewidth=1.5)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # Component losses
    axes[0, 1].semilogy(history["epoch"], history["pde_loss"], label="PDE", linewidth=1.5)
    axes[0, 1].semilogy(history["epoch"], history["bc_loss"], label="BC", linewidth=1.5)
    axes[0, 1].semilogy(history["epoch"], history["ic_loss"], label="IC", linewidth=1.5)
    if "feller_loss" in history:
        axes[0, 1].semilogy(history["epoch"], history["feller_loss"],
                            label="Feller", linewidth=1.5)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss Component")
    axes[0, 1].set_title("Loss Components")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Validation error
    if "val_mae" in history:
        axes[1, 0].semilogy(history["epoch"], history["val_mae"], "r-", linewidth=1.5)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Mean Absolute Error")
        axes[1, 0].set_title("Validation MAE (vs Analytical Heston)")
        axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    if "lr" in history:
        axes[1, 1].semilogy(history["epoch"], history["lr"], "g-", linewidth=1.5)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Heston PINN Training Progress", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def plot_pinn_vs_analytical(
    model=None,
    K: float = 100.0,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma: float = 0.3,
    rho: float = -0.7,
    r: float = 0.05,
    T: float = 0.5,
    save_path: Optional[str] = None,
):
    """Compare PINN prices against analytical Heston prices."""
    S_vals = np.linspace(70, 130, 50)
    v_vals = [0.02, 0.04, 0.08]

    fig, axes = plt.subplots(1, len(v_vals), figsize=(5 * len(v_vals), 5))

    for idx, v0 in enumerate(v_vals):
        analytical_prices = []
        pinn_prices = []

        for S in S_vals:
            analytical = heston_call_price(S, K, T, r, v0, kappa, theta, sigma, rho)
            analytical_prices.append(analytical)

            if model is not None:
                import torch
                device = next(model.parameters()).device
                S_t = torch.tensor([S], dtype=torch.float32, device=device)
                v_t = torch.tensor([v0], dtype=torch.float32, device=device)
                t_t = torch.tensor([0.0], dtype=torch.float32, device=device)
                with torch.no_grad():
                    pinn_price = model(S_t, v_t, t_t).item()
                pinn_prices.append(pinn_price)

        axes[idx].plot(S_vals, analytical_prices, "b-", linewidth=2, label="Analytical")
        if pinn_prices:
            axes[idx].plot(S_vals, pinn_prices, "r--", linewidth=2, label="PINN")

        axes[idx].axvline(x=K, color="gray", linestyle=":", alpha=0.5)
        axes[idx].set_xlabel("Spot Price S")
        axes[idx].set_ylabel("Call Price V")
        axes[idx].set_title(f"v0 = {v0:.2f} (vol={np.sqrt(v0)*100:.0f}%)")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle(f"Heston Call Prices (K={K}, T={T}y)", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_smile_term_structure(
    S: float = 100.0,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma: float = 0.3,
    rho: float = -0.7,
    r: float = 0.05,
    expiries: list = None,
    save_path: Optional[str] = None,
):
    """Plot how the volatility smile changes with expiry (term structure)."""
    if expiries is None:
        expiries = [0.05, 0.10, 0.25, 0.50, 1.0]

    strikes = np.linspace(0.75 * S, 1.25 * S, 40)
    moneyness = strikes / S

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(expiries)))

    for T_val, color in zip(expiries, colors):
        ivs = []
        for K in strikes:
            price = heston_call_price(S, K, T_val, r, v0, kappa, theta, sigma, rho)
            iv = implied_volatility(price, S, K, T_val, r)
            ivs.append(iv * 100 if not np.isnan(iv) else np.nan)

        days = int(T_val * 365)
        ax.plot(moneyness, ivs, color=color, linewidth=2, label=f"T={days}d")

    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Moneyness (K/S)", fontsize=12)
    ax.set_ylabel("Implied Volatility (%)", fontsize=12)
    ax.set_title("Volatility Smile Term Structure", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heston PINN Visualization")
    parser.add_argument("--show-smile", action="store_true")
    parser.add_argument("--show-surface", action="store_true")
    parser.add_argument("--show-greeks", action="store_true")
    parser.add_argument("--show-term", action="store_true")
    parser.add_argument("--show-price-surface", action="store_true")
    parser.add_argument("--show-all", action="store_true")
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    show_all = args.show_all or not any([
        args.show_smile, args.show_surface, args.show_greeks,
        args.show_term, args.show_price_surface
    ])

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Default Heston parameters (equity)
    params = dict(S=100, v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, r=0.05)

    if show_all or args.show_smile:
        plot_implied_vol_smile(
            **params,
            save_path=str(save_dir / "vol_smile.png") if save_dir else None
        )

    if show_all or args.show_surface:
        plot_implied_vol_surface(
            **params,
            save_path=str(save_dir / "vol_surface.png") if save_dir else None
        )

    if show_all or args.show_greeks:
        plot_greeks_heatmaps(
            K=100, **{k: v for k, v in params.items() if k != "S"},
            save_path=str(save_dir / "greeks_heatmaps.png") if save_dir else None
        )

    if show_all or args.show_term:
        plot_smile_term_structure(
            **params,
            save_path=str(save_dir / "smile_term.png") if save_dir else None
        )

    if show_all or args.show_price_surface:
        plot_option_price_surface(
            K=100, **{k: v for k, v in params.items() if k != "S"},
            T=0.5,
            save_path=str(save_dir / "price_surface.png") if save_dir else None
        )
