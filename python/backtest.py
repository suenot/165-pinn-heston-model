"""
backtest.py - Backtest volatility trading strategy using PINN Heston model

Strategy:
    1. Calibrate Heston to current market implied volatilities
    2. Price options using PINN (or analytical Heston)
    3. Identify mispriced options: PINN price vs market price
    4. Trade: buy underpriced vol, sell overpriced vol
    5. Delta-hedge to isolate volatility exposure
    6. Track P&L over time

Supports both synthetic and Bybit crypto data.
"""

import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from heston_analytical import heston_call_price, heston_put_price, implied_volatility


@dataclass
class OptionPosition:
    """Represents a single option position."""
    strike: float
    expiry: float  # time to expiry in years
    option_type: str  # 'call' or 'put'
    quantity: int  # positive = long, negative = short
    entry_price: float
    entry_iv: float
    model_price: float
    model_iv: float
    delta: float = 0.0
    hedge_shares: float = 0.0  # shares held for delta hedging
    entry_time: int = 0  # step when position was opened


@dataclass
class BacktestResult:
    """Container for backtest results."""
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0
    avg_edge: float = 0.0
    pnl_series: List[float] = field(default_factory=list)
    positions_history: List[Dict] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)


class HestonVolatilityBacktester:
    """
    Backtest a volatility trading strategy based on Heston model mispricing.

    The strategy:
        - Calibrate Heston to the current volatility surface
        - Compare model IVs to market IVs
        - Trade the largest mispricings
        - Delta-hedge all positions
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_positions: int = 10,
        edge_threshold: float = 0.02,  # 2% IV edge required
        position_size: float = 0.05,  # 5% of capital per trade
        delta_hedge: bool = True,
        rebalance_freq: int = 1,  # rebalance delta every N steps
    ):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.edge_threshold = edge_threshold
        self.position_size = position_size
        self.delta_hedge = delta_hedge
        self.rebalance_freq = rebalance_freq

    def generate_synthetic_market(
        self,
        n_steps: int = 252,
        dt: float = 1 / 252,
        S0: float = 100.0,
        true_params: Optional[Dict] = None,
        market_noise: float = 0.01,
    ) -> pd.DataFrame:
        """
        Generate synthetic market data with Heston dynamics.

        Simulates:
            - Spot price path (Euler-Maruyama)
            - Option chain at each step with noisy IVs

        Args:
            n_steps: Number of time steps
            dt: Time step size
            S0: Initial spot price
            true_params: True Heston parameters
            market_noise: Noise added to IVs (simulates market microstructure)

        Returns:
            DataFrame with simulated market data
        """
        if true_params is None:
            true_params = {
                "kappa": 2.0, "theta": 0.04, "sigma": 0.3,
                "rho": -0.7, "v0": 0.04, "r": 0.05,
            }

        kappa = true_params["kappa"]
        theta = true_params["theta"]
        sigma = true_params["sigma"]
        rho = true_params["rho"]
        r = true_params["r"]
        v0 = true_params["v0"]

        # Simulate Heston paths
        S = np.zeros(n_steps + 1)
        v = np.zeros(n_steps + 1)
        S[0] = S0
        v[0] = v0

        np.random.seed(42)

        for i in range(n_steps):
            Z1 = np.random.normal()
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal()

            v_pos = max(v[i], 0)  # Absorbing at 0
            S[i + 1] = S[i] * np.exp(
                (r - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * Z1
            )
            v[i + 1] = max(
                v[i] + kappa * (theta - v[i]) * dt
                + sigma * np.sqrt(v_pos * dt) * Z2,
                0,
            )

        # Generate option chain at each step
        market_data = []
        strikes_frac = [0.90, 0.95, 1.00, 1.05, 1.10]
        expiry_T = 0.25  # 3-month options

        for step in range(n_steps):
            spot = S[step]
            var = v[step]

            for sf in strikes_frac:
                K = spot * sf

                # True Heston price
                true_price = heston_call_price(
                    spot, K, expiry_T, r, var, kappa, theta, sigma, rho
                )
                true_iv = implied_volatility(true_price, spot, K, expiry_T, r)

                if np.isnan(true_iv) or true_iv <= 0:
                    continue

                # Market IV with noise
                market_iv = true_iv + np.random.normal(0, market_noise)
                market_iv = max(market_iv, 0.01)

                # Market price from noisy IV
                from heston_analytical import black_scholes_call
                market_price = black_scholes_call(spot, K, expiry_T, r, market_iv)

                market_data.append({
                    "step": step,
                    "spot": spot,
                    "variance": var,
                    "strike": K,
                    "expiry": expiry_T,
                    "moneyness": K / spot,
                    "true_price": true_price,
                    "true_iv": true_iv,
                    "market_price": market_price,
                    "market_iv": market_iv,
                    "iv_error": market_iv - true_iv,
                })

        return pd.DataFrame(market_data)

    def run_backtest(
        self,
        market_data: pd.DataFrame,
        heston_params: Optional[Dict] = None,
    ) -> BacktestResult:
        """
        Run the volatility trading backtest.

        Args:
            market_data: DataFrame with columns: step, spot, strike, expiry,
                         market_price, market_iv, (optionally: true_iv)
            heston_params: Heston parameters for model pricing

        Returns:
            BacktestResult with P&L and diagnostics
        """
        if heston_params is None:
            heston_params = {
                "kappa": 2.0, "theta": 0.04, "sigma": 0.3,
                "rho": -0.7, "v0": 0.04, "r": 0.05,
            }

        result = BacktestResult()
        capital = self.initial_capital
        positions: List[OptionPosition] = []
        pnl_series = [0.0]
        peak_capital = capital

        steps = sorted(market_data["step"].unique())

        for step_idx, step in enumerate(steps):
            step_data = market_data[market_data["step"] == step]

            if step_data.empty:
                continue

            spot = step_data.iloc[0]["spot"]
            var_est = step_data.iloc[0].get("variance", heston_params["v0"])

            # --- Close expired or stale positions ---
            closed_pnl = 0.0
            new_positions = []
            for pos in positions:
                age = step - pos.entry_time
                remaining_T = pos.expiry - age / 252.0

                if remaining_T <= 0.01:
                    # Option expired: compute final payoff
                    if pos.option_type == "call":
                        payoff = max(spot - pos.strike, 0)
                    else:
                        payoff = max(pos.strike - spot, 0)

                    trade_pnl = pos.quantity * (payoff - pos.entry_price)
                    # Add hedging P&L
                    trade_pnl += pos.hedge_shares * (spot - step_data.iloc[0]["spot"])
                    closed_pnl += trade_pnl

                    result.positions_history.append({
                        "entry_step": pos.entry_time,
                        "exit_step": step,
                        "strike": pos.strike,
                        "type": pos.option_type,
                        "quantity": pos.quantity,
                        "pnl": trade_pnl,
                        "edge": abs(pos.model_iv - pos.entry_iv),
                    })
                else:
                    new_positions.append(pos)

            positions = new_positions

            # --- Generate signals ---
            if len(positions) < self.max_positions:
                for _, row in step_data.iterrows():
                    K = row["strike"]
                    T_opt = row["expiry"]
                    market_iv = row["market_iv"]
                    market_price = row["market_price"]

                    # Model price using Heston
                    model_price = heston_call_price(
                        spot, K, T_opt, heston_params["r"], var_est,
                        heston_params["kappa"], heston_params["theta"],
                        heston_params["sigma"], heston_params["rho"],
                    )
                    model_iv = implied_volatility(model_price, spot, K, T_opt, heston_params["r"])

                    if np.isnan(model_iv):
                        continue

                    edge = model_iv - market_iv

                    # Check edge threshold
                    if abs(edge) > self.edge_threshold:
                        # Determine position size
                        notional = capital * self.position_size
                        n_contracts = max(1, int(notional / market_price)) if market_price > 0 else 0

                        if n_contracts == 0:
                            continue

                        if edge > 0:
                            # Model says IV should be higher -> buy vol
                            quantity = n_contracts
                        else:
                            # Model says IV should be lower -> sell vol
                            quantity = -n_contracts

                        # Compute delta for hedging
                        dS = 0.01
                        price_up = heston_call_price(
                            spot + dS, K, T_opt, heston_params["r"], var_est,
                            heston_params["kappa"], heston_params["theta"],
                            heston_params["sigma"], heston_params["rho"],
                        )
                        delta = (price_up - model_price) / dS

                        pos = OptionPosition(
                            strike=K,
                            expiry=T_opt,
                            option_type="call",
                            quantity=quantity,
                            entry_price=market_price,
                            entry_iv=market_iv,
                            model_price=model_price,
                            model_iv=model_iv,
                            delta=delta,
                            hedge_shares=-quantity * delta if self.delta_hedge else 0,
                            entry_time=step,
                        )
                        positions.append(pos)
                        result.n_trades += 1

                        if len(positions) >= self.max_positions:
                            break

            # --- Mark-to-market ---
            portfolio_value = capital + closed_pnl
            for pos in positions:
                remaining_T = pos.expiry - (step - pos.entry_time) / 252.0
                if remaining_T > 0:
                    current_price = heston_call_price(
                        spot, pos.strike, remaining_T, heston_params["r"],
                        var_est, heston_params["kappa"], heston_params["theta"],
                        heston_params["sigma"], heston_params["rho"],
                    )
                    unrealized = pos.quantity * (current_price - pos.entry_price)
                    portfolio_value += unrealized

            step_pnl = portfolio_value - self.initial_capital
            pnl_series.append(step_pnl)
            capital += closed_pnl

            # Track drawdown
            peak_capital = max(peak_capital, portfolio_value)
            drawdown = (peak_capital - portfolio_value) / peak_capital

        # Compute results
        result.pnl_series = pnl_series
        result.total_pnl = pnl_series[-1]

        daily_returns = np.diff(pnl_series) / self.initial_capital
        result.daily_returns = daily_returns.tolist()

        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            result.sharpe_ratio = float(
                np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            )

        # Drawdown
        cummax = np.maximum.accumulate(np.array(pnl_series) + self.initial_capital)
        drawdowns = 1 - (np.array(pnl_series) + self.initial_capital) / cummax
        result.max_drawdown = float(np.max(drawdowns))

        # Win rate
        if result.positions_history:
            wins = sum(1 for p in result.positions_history if p["pnl"] > 0)
            result.win_rate = wins / len(result.positions_history)
            result.avg_edge = np.mean([p["edge"] for p in result.positions_history])

        return result


def print_backtest_report(result: BacktestResult, title: str = "Backtest Results"):
    """Print formatted backtest report."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(f"  Total P&L:      ${result.total_pnl:>12,.2f}")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:>12.2f}")
    print(f"  Max Drawdown:    {result.max_drawdown:>12.2%}")
    print(f"  Win Rate:        {result.win_rate:>12.2%}")
    print(f"  # Trades:        {result.n_trades:>12d}")
    print(f"  Avg IV Edge:     {result.avg_edge * 100:>12.2f}%")
    print("=" * 60)


def plot_backtest_results(result: BacktestResult, save_path: Optional[str] = None):
    """Plot backtest P&L curve and statistics."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # P&L curve
    axes[0, 0].plot(result.pnl_series, "b-", linewidth=1.5)
    axes[0, 0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("P&L ($)")
    axes[0, 0].set_title("Cumulative P&L")
    axes[0, 0].grid(True, alpha=0.3)

    # Daily returns distribution
    if result.daily_returns:
        axes[0, 1].hist(result.daily_returns, bins=50, color="steelblue", alpha=0.7)
        axes[0, 1].axvline(x=0, color="red", linestyle="--")
        axes[0, 1].set_xlabel("Daily Return")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Return Distribution")
        axes[0, 1].grid(True, alpha=0.3)

    # Drawdown
    cummax = np.maximum.accumulate(np.array(result.pnl_series) + 100000)
    dd = 1 - (np.array(result.pnl_series) + 100000) / cummax
    axes[1, 0].fill_between(range(len(dd)), dd, alpha=0.5, color="red")
    axes[1, 0].set_xlabel("Time Step")
    axes[1, 0].set_ylabel("Drawdown (%)")
    axes[1, 0].set_title("Drawdown")
    axes[1, 0].grid(True, alpha=0.3)

    # Trade P&L distribution
    if result.positions_history:
        trade_pnls = [p["pnl"] for p in result.positions_history]
        colors = ["green" if p > 0 else "red" for p in trade_pnls]
        axes[1, 1].bar(range(len(trade_pnls)), trade_pnls, color=colors, alpha=0.7)
        axes[1, 1].axhline(y=0, color="gray", linestyle="--")
        axes[1, 1].set_xlabel("Trade #")
        axes[1, 1].set_ylabel("P&L ($)")
        axes[1, 1].set_title("Per-Trade P&L")
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Heston Volatility Trading Backtest", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest Heston Vol Trading")
    parser.add_argument("--strategy", type=str, default="vol_arb",
                        choices=["vol_arb"])
    parser.add_argument("--period", type=str, default="252d",
                        help="Backtest period (e.g. 252d for 1 year)")
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--edge-threshold", type=float, default=0.02)
    parser.add_argument("--noise", type=float, default=0.015)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    # Parse period
    n_steps = int(args.period.replace("d", ""))

    print("Generating synthetic market data...")
    backtester = HestonVolatilityBacktester(
        initial_capital=args.capital,
        edge_threshold=args.edge_threshold,
    )

    # Equity-like parameters
    equity_params = {
        "kappa": 2.0, "theta": 0.04, "sigma": 0.3,
        "rho": -0.7, "v0": 0.04, "r": 0.05,
    }

    market_data = backtester.generate_synthetic_market(
        n_steps=n_steps,
        S0=100.0,
        true_params=equity_params,
        market_noise=args.noise,
    )

    print(f"Generated {len(market_data)} option observations over {n_steps} steps")

    # Run with correct model (oracle backtest)
    result_oracle = backtester.run_backtest(market_data, heston_params=equity_params)
    print_backtest_report(result_oracle, "Oracle Backtest (True Params)")

    # Run with slightly wrong model parameters
    wrong_params = {
        "kappa": 1.5, "theta": 0.05, "sigma": 0.4,
        "rho": -0.5, "v0": 0.04, "r": 0.05,
    }
    result_wrong = backtester.run_backtest(market_data, heston_params=wrong_params)
    print_backtest_report(result_wrong, "Misspecified Model Backtest")

    if args.plot:
        plot_backtest_results(result_oracle, save_path=None)
