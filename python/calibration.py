"""
calibration.py - Calibrate Heston model parameters to market implied volatilities

Two calibration approaches:
    1. Classical: Minimize IV error using scipy.optimize (with analytical Heston as forward solver)
    2. PINN-accelerated: Use trained PINN as the forward solver for faster calibration

Constraints:
    - kappa > 0, theta > 0, sigma > 0
    - -1 < rho < 1
    - v0 > 0
    - 2*kappa*theta > sigma^2 (Feller condition, soft constraint)
"""

import argparse
import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import Dict, List, Tuple, Optional
import time

from heston_analytical import (
    heston_call_price,
    heston_put_price,
    implied_volatility,
)


class HestonCalibrator:
    """
    Calibrate Heston parameters to observed implied volatilities.

    Minimizes:
        sum_i w_i * (sigma_model(K_i, T_i; Theta) - sigma_mkt(K_i, T_i))^2

    subject to parameter constraints and the Feller condition.
    """

    def __init__(
        self,
        spot: float,
        rate: float = 0.05,
        use_vega_weights: bool = True,
        feller_penalty: float = 100.0,
    ):
        """
        Args:
            spot: Current spot price
            rate: Risk-free rate
            use_vega_weights: Weight by vega (more liquid ATM options get higher weight)
            feller_penalty: Penalty multiplier for Feller condition violation
        """
        self.spot = spot
        self.rate = rate
        self.use_vega_weights = use_vega_weights
        self.feller_penalty = feller_penalty

    def objective(
        self,
        params: np.ndarray,
        strikes: np.ndarray,
        expiries: np.ndarray,
        market_ivs: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calibration objective: weighted sum of squared IV errors.

        Args:
            params: [kappa, theta, sigma, rho, v0]
            strikes: Strike prices
            expiries: Time to expiry (years)
            market_ivs: Market implied volatilities
            weights: Optional weights per option

        Returns:
            Objective function value
        """
        kappa, theta, sigma, rho, v0 = params

        # Parameter feasibility
        if kappa <= 0 or theta <= 0 or sigma <= 0 or v0 <= 0:
            return 1e10
        if abs(rho) >= 1:
            return 1e10

        if weights is None:
            weights = np.ones(len(strikes))

        total_error = 0.0
        n_valid = 0

        for i in range(len(strikes)):
            try:
                price = heston_call_price(
                    self.spot, strikes[i], expiries[i], self.rate,
                    v0, kappa, theta, sigma, rho,
                )
                model_iv = implied_volatility(
                    price, self.spot, strikes[i], expiries[i], self.rate
                )

                if not np.isnan(model_iv) and model_iv > 0:
                    total_error += weights[i] * (model_iv - market_ivs[i]) ** 2
                    n_valid += 1
            except (ValueError, RuntimeError):
                total_error += weights[i] * 1.0  # Penalty for failed pricing

        if n_valid == 0:
            return 1e10

        # Feller condition penalty
        feller_violation = max(0, sigma**2 - 2 * kappa * theta)
        total_error += self.feller_penalty * feller_violation

        return total_error / n_valid

    def compute_vega_weights(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
    ) -> np.ndarray:
        """
        Compute vega-based weights: ATM options get higher weight.

        Approximation: weight proportional to sqrt(T) * exp(-0.5 * moneyness^2)
        """
        moneyness = np.log(strikes / self.spot)
        weights = np.sqrt(expiries) * np.exp(-0.5 * (moneyness / 0.3) ** 2)
        weights /= weights.sum()
        return weights * len(weights)

    def calibrate(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
        market_ivs: np.ndarray,
        method: str = "differential_evolution",
        initial_guess: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the calibration.

        Args:
            strikes: Strike prices
            expiries: Expiry times
            market_ivs: Market implied volatilities
            method: 'differential_evolution' or 'nelder_mead'
            initial_guess: Initial parameter guess [kappa, theta, sigma, rho, v0]
            verbose: Print progress

        Returns:
            Dict with calibrated parameters and diagnostics
        """
        if verbose:
            print(f"Calibrating Heston to {len(strikes)} options...")
            print(f"Spot: {self.spot:.2f}, Rate: {self.rate:.4f}")

        # Compute weights
        weights = None
        if self.use_vega_weights:
            weights = self.compute_vega_weights(strikes, expiries)

        start_time = time.time()

        # Parameter bounds: [kappa, theta, sigma, rho, v0]
        bounds = [
            (0.1, 20.0),     # kappa
            (0.001, 2.0),    # theta
            (0.05, 5.0),     # sigma
            (-0.99, 0.99),   # rho
            (0.001, 2.0),    # v0
        ]

        if method == "differential_evolution":
            result = differential_evolution(
                self.objective,
                bounds,
                args=(strikes, expiries, market_ivs, weights),
                seed=42,
                maxiter=200,
                tol=1e-8,
                polish=True,
                disp=verbose,
            )
        elif method == "nelder_mead":
            if initial_guess is None:
                # Use median IV for initial v0 and theta
                median_iv = np.median(market_ivs)
                initial_guess = np.array([
                    2.0,           # kappa
                    median_iv**2,  # theta
                    0.5,           # sigma
                    -0.5,          # rho
                    median_iv**2,  # v0
                ])

            result = minimize(
                self.objective,
                initial_guess,
                args=(strikes, expiries, market_ivs, weights),
                method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed = time.time() - start_time

        kappa, theta, sigma, rho, v0 = result.x
        feller = 2 * kappa * theta > sigma**2

        calibrated = {
            "kappa": float(kappa),
            "theta": float(theta),
            "sigma": float(sigma),
            "rho": float(rho),
            "v0": float(v0),
            "objective": float(result.fun),
            "feller_satisfied": bool(feller),
            "feller_ratio": float(2 * kappa * theta / sigma**2),
            "elapsed_seconds": elapsed,
            "n_options": len(strikes),
            "success": bool(result.success) if hasattr(result, "success") else True,
        }

        if verbose:
            print(f"\nCalibration complete in {elapsed:.2f}s")
            print(f"  kappa = {kappa:.4f}")
            print(f"  theta = {theta:.6f} (vol = {np.sqrt(theta)*100:.1f}%)")
            print(f"  sigma = {sigma:.4f}")
            print(f"  rho   = {rho:.4f}")
            print(f"  v0    = {v0:.6f} (vol = {np.sqrt(v0)*100:.1f}%)")
            print(f"  Feller: {feller} (ratio={2*kappa*theta/sigma**2:.2f})")
            print(f"  Objective: {result.fun:.8f}")

        return calibrated

    def compute_calibration_errors(
        self,
        params: Dict,
        strikes: np.ndarray,
        expiries: np.ndarray,
        market_ivs: np.ndarray,
    ) -> Dict:
        """
        Compute detailed calibration errors.

        Returns:
            Dict with MAE, RMSE, max error, and per-option errors
        """
        kappa = params["kappa"]
        theta = params["theta"]
        sigma = params["sigma"]
        rho = params["rho"]
        v0 = params["v0"]

        model_ivs = []
        errors = []

        for i in range(len(strikes)):
            try:
                price = heston_call_price(
                    self.spot, strikes[i], expiries[i], self.rate,
                    v0, kappa, theta, sigma, rho,
                )
                model_iv = implied_volatility(
                    price, self.spot, strikes[i], expiries[i], self.rate
                )
                model_ivs.append(model_iv)
                errors.append(model_iv - market_ivs[i])
            except Exception:
                model_ivs.append(np.nan)
                errors.append(np.nan)

        errors = np.array(errors)
        valid = ~np.isnan(errors)

        return {
            "model_ivs": np.array(model_ivs),
            "errors": errors,
            "mae": float(np.mean(np.abs(errors[valid]))),
            "rmse": float(np.sqrt(np.mean(errors[valid] ** 2))),
            "max_error": float(np.max(np.abs(errors[valid]))),
            "mean_bias": float(np.mean(errors[valid])),
            "n_valid": int(np.sum(valid)),
        }


def calibrate_to_bybit(symbol: str = "BTC", verbose: bool = True) -> Dict:
    """
    End-to-end calibration to Bybit crypto options.

    Args:
        symbol: 'BTC' or 'ETH'
        verbose: Print results

    Returns:
        Calibrated Heston parameters
    """
    from data_loader import load_data

    data = load_data(source="bybit", symbol=symbol)

    if len(data["strikes"]) == 0:
        print("No market data available. Using synthetic data.")
        data = load_data(source="synthetic")

    calibrator = HestonCalibrator(
        spot=data["spot"],
        rate=data["params"]["r"],
    )

    result = calibrator.calibrate(
        strikes=data["strikes"],
        expiries=data["expiries"],
        market_ivs=data["implied_vols"],
        verbose=verbose,
    )

    return result


def calibrate_to_synthetic(verbose: bool = True) -> Dict:
    """
    Calibrate to synthetic data (for testing -- should recover known parameters).
    """
    from data_loader import SyntheticStockData

    # True parameters
    true_params = {
        "kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7, "v0": 0.04
    }
    synth = SyntheticStockData(**true_params)
    options_df = synth.generate_option_chain(noise_level=0.02)

    calibrator = HestonCalibrator(spot=synth.S, rate=synth.r)

    result = calibrator.calibrate(
        strikes=options_df["strike"].values,
        expiries=options_df["expiry"].values,
        market_ivs=options_df["implied_vol"].values,
        verbose=verbose,
    )

    if verbose:
        print("\nTrue vs Calibrated:")
        for key in ["kappa", "theta", "sigma", "rho", "v0"]:
            print(f"  {key:>6}: true={true_params[key]:.4f}, "
                  f"calibrated={result[key]:.4f}, "
                  f"error={abs(result[key] - true_params[key]):.4f}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heston Calibration")
    parser.add_argument("--source", type=str, default="synthetic",
                        choices=["bybit", "synthetic"])
    parser.add_argument("--symbol", type=str, default="BTC")
    args = parser.parse_args()

    if args.source == "bybit":
        result = calibrate_to_bybit(symbol=args.symbol)
    else:
        result = calibrate_to_synthetic()
