"""
train.py - Training loop for the Heston PINN

Trains the PINN by minimizing a composite loss:
    L = lambda_pde * L_pde + lambda_bc * L_bc + lambda_ic * L_ic + lambda_data * L_data

Features:
    - Multi-scale collocation point generation
    - Adaptive loss weighting (gradient-based balancing)
    - Cosine annealing learning rate schedule
    - Curriculum learning (optional)
    - Validation against semi-analytical Heston prices
"""

import argparse
import time
import json
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from heston_pinn import (
    HestonPINN,
    HestonPINNResidual,
    generate_collocation_points,
    terminal_condition_loss,
    boundary_condition_loss,
    feller_boundary_loss,
)
from heston_analytical import heston_call_price


def compute_grad_norm(model: HestonPINN, loss: torch.Tensor) -> float:
    """Compute the norm of gradients of loss w.r.t. model parameters."""
    grads = torch.autograd.grad(
        loss, model.parameters(), retain_graph=True, allow_unused=True
    )
    total_norm = 0.0
    for g in grads:
        if g is not None:
            total_norm += g.norm().item() ** 2
    return np.sqrt(total_norm)


def validate_against_analytical(
    model: HestonPINN,
    heston_params: Dict[str, float],
    K: float,
    device: str = "cpu",
    n_points: int = 20,
) -> Tuple[float, float]:
    """
    Compare PINN prices against semi-analytical Heston prices.

    Returns:
        (mean_abs_error, max_abs_error)
    """
    S_vals = np.linspace(70, 130, n_points)
    v_vals = [0.02, 0.04, 0.08]
    t_val = 0.0  # Price at t=0

    errors = []
    for v0 in v_vals:
        for S in S_vals:
            # Analytical price
            analytical = heston_call_price(
                S, K, heston_params["T"], heston_params["r"],
                v0, heston_params["kappa"], heston_params["theta"],
                heston_params["sigma"], heston_params["rho"],
            )

            # PINN price
            S_t = torch.tensor([S], dtype=torch.float32, device=device)
            v_t = torch.tensor([v0], dtype=torch.float32, device=device)
            t_t = torch.tensor([t_val], dtype=torch.float32, device=device)

            with torch.no_grad():
                pinn_price = model(S_t, v_t, t_t).item()

            errors.append(abs(pinn_price - analytical))

    return np.mean(errors), np.max(errors)


def train(args):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Heston parameters
    heston_params = {
        "kappa": args.kappa,
        "theta": args.theta,
        "sigma": args.sigma_vol,
        "rho": args.rho,
        "r": args.rate,
        "T": args.expiry,
    }

    # Check Feller condition
    feller = 2 * args.kappa * args.theta > args.sigma_vol ** 2
    print(f"Feller condition: {feller} "
          f"(2kT={2*args.kappa*args.theta:.4f} vs s^2={args.sigma_vol**2:.4f})")

    # Domain
    S_max = args.spot * 2.0
    v_max = max(args.theta * 5, 1.0)
    T = args.expiry
    K = args.strike

    # Create model
    model = HestonPINN(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        S_max=S_max,
        v_max=v_max,
        T_max=T,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # PDE residual computer
    residual = HestonPINNResidual(
        model,
        kappa=args.kappa,
        theta=args.theta,
        sigma=args.sigma_vol,
        rho=args.rho,
        r=args.rate,
    )

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss weights (will be adapted)
    lambda_pde = 1.0
    lambda_bc = 10.0
    lambda_ic = 10.0
    lambda_feller = 5.0

    # Training history
    history = {
        "epoch": [],
        "loss": [],
        "pde_loss": [],
        "bc_loss": [],
        "ic_loss": [],
        "feller_loss": [],
        "val_mae": [],
        "val_max_err": [],
        "lr": [],
    }

    print("\n" + "=" * 80)
    print(f"{'Epoch':>7} | {'Loss':>10} | {'PDE':>10} | {'BC':>10} | "
          f"{'IC':>10} | {'Feller':>10} | {'Val MAE':>10} | {'LR':>10}")
    print("=" * 80)

    best_val_mae = float("inf")
    start_time = time.time()

    for epoch in range(args.epochs):
        optimizer.zero_grad()

        # Generate fresh collocation points each epoch
        S_col, v_col, t_col = generate_collocation_points(
            S_max, v_max, T, K, N_total=args.n_collocation, device=device
        )

        # 1. PDE residual loss
        L_pde = residual.pde_loss(S_col, v_col, t_col)

        # 2. Terminal condition loss
        L_ic = terminal_condition_loss(
            model, K, S_max, v_max, T,
            N_bc=args.n_boundary, option_type="call", device=device
        )

        # 3. Spatial boundary loss
        L_bc = boundary_condition_loss(
            model, K, args.rate, S_max, v_max, T,
            N_bc=args.n_boundary, option_type="call", device=device
        )

        # 4. Feller boundary loss (v=0)
        L_feller = feller_boundary_loss(
            model, args.kappa, args.theta, args.rate,
            S_max, T, N_bc=args.n_boundary, device=device
        )

        # Composite loss
        loss = (
            lambda_pde * L_pde
            + lambda_bc * L_bc
            + lambda_ic * L_ic
            + lambda_feller * L_feller
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Adaptive weight adjustment every 200 epochs
        if epoch > 0 and epoch % 200 == 0:
            try:
                grad_pde = compute_grad_norm(model, lambda_pde * L_pde)
                grad_bc = compute_grad_norm(model, lambda_bc * L_bc)
                grad_ic = compute_grad_norm(model, lambda_ic * L_ic)

                mean_grad = (grad_pde + grad_bc + grad_ic) / 3.0
                if grad_bc > 1e-10:
                    lambda_bc = mean_grad / grad_bc
                if grad_ic > 1e-10:
                    lambda_ic = mean_grad / grad_ic
            except RuntimeError:
                pass  # Skip if gradient computation fails

        # Logging
        if epoch % args.log_every == 0:
            val_mae, val_max = validate_against_analytical(
                model, heston_params, K, device=device, n_points=10
            )

            current_lr = scheduler.get_last_lr()[0]

            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["pde_loss"].append(L_pde.item())
            history["bc_loss"].append(L_bc.item())
            history["ic_loss"].append(L_ic.item())
            history["feller_loss"].append(L_feller.item())
            history["val_mae"].append(val_mae)
            history["val_max_err"].append(val_max)
            history["lr"].append(current_lr)

            print(
                f"{epoch:7d} | {loss.item():10.6f} | {L_pde.item():10.6f} | "
                f"{L_bc.item():10.6f} | {L_ic.item():10.6f} | "
                f"{L_feller.item():10.6f} | {val_mae:10.6f} | {current_lr:10.2e}"
            )

            # Save best model
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                save_path = Path(args.save_dir) / "best_model.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "heston_params": heston_params,
                    "val_mae": val_mae,
                    "S_max": S_max,
                    "v_max": v_max,
                    "T_max": T,
                }, str(save_path))

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best validation MAE: {best_val_mae:.6f}")

    # Save final model
    final_path = Path(args.save_dir) / "final_model.pt"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "heston_params": heston_params,
        "history": history,
        "S_max": S_max,
        "v_max": v_max,
        "T_max": T,
    }, str(final_path))

    # Save history
    history_path = Path(args.save_dir) / "training_history.json"
    # Convert numpy types to native Python for JSON serialization
    serializable_history = {}
    for k, v in history.items():
        serializable_history[k] = [float(x) if isinstance(x, (np.floating, float)) else int(x) for x in v]
    with open(history_path, "w") as f:
        json.dump(serializable_history, f, indent=2)

    print(f"Model saved to {final_path}")
    print(f"History saved to {history_path}")

    return model, history


def parse_args():
    parser = argparse.ArgumentParser(description="Train Heston PINN")

    # Heston parameters
    parser.add_argument("--kappa", type=float, default=2.0, help="Mean-reversion speed")
    parser.add_argument("--theta", type=float, default=0.04, help="Long-run variance")
    parser.add_argument("--sigma-vol", type=float, default=0.3, help="Vol of vol")
    parser.add_argument("--rho", type=float, default=-0.7, help="Correlation")
    parser.add_argument("--rate", type=float, default=0.05, help="Risk-free rate")

    # Option parameters
    parser.add_argument("--spot", type=float, default=100.0, help="Initial spot")
    parser.add_argument("--strike", type=float, default=100.0, help="Strike price")
    parser.add_argument("--expiry", type=float, default=1.0, help="Time to expiry")

    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer width")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of residual blocks")

    # Training
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--n-collocation", type=int, default=4096, help="Collocation points")
    parser.add_argument("--n-boundary", type=int, default=128, help="Boundary points")

    # Logging and saving
    parser.add_argument("--log-every", type=int, default=100, help="Log every N epochs")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Save directory")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, history = train(args)
