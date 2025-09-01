"""
data_loader.py - Market data loader for stocks and Bybit crypto options

Fetches option chain data from:
    1. Bybit API (crypto: BTC, ETH options)
    2. Synthetic/sample stock data (for offline testing)

Provides normalized data for PINN training and calibration.
"""

import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import time


class BybitOptionsLoader:
    """
    Fetch cryptocurrency option data from Bybit exchange.

    Bybit provides European-style options on BTC and ETH
    with USDC settlement.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self, symbol: str = "BTC"):
        self.symbol = symbol.upper()

    def fetch_tickers(self) -> List[Dict]:
        """
        Fetch all option tickers for the given base coin.

        Returns:
            List of option ticker dictionaries with fields:
                symbol, bid, ask, mark_price, implied_vol,
                delta, gamma, vega, theta, strike, expiry, option_type
        """
        url = f"{self.BASE_URL}/v5/market/tickers"
        params = {"category": "option", "baseCoin": self.symbol}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data["retCode"] != 0:
                print(f"Bybit API error: {data['retMsg']}")
                return []

            options = []
            for item in data["result"]["list"]:
                # Parse symbol: e.g., BTC-28JUN24-70000-C
                symbol_parts = item["symbol"].split("-")
                if len(symbol_parts) >= 4:
                    expiry_str = symbol_parts[1]
                    strike = float(symbol_parts[2])
                    opt_type = "call" if symbol_parts[3] == "C" else "put"
                else:
                    continue

                # Parse expiry date
                try:
                    expiry_date = datetime.strptime(expiry_str, "%d%b%y")
                    days_to_expiry = (expiry_date - datetime.now()).days
                    T = max(days_to_expiry / 365.0, 1 / 365.0)
                except ValueError:
                    continue

                bid = float(item.get("bid1Price", 0) or 0)
                ask = float(item.get("ask1Price", 0) or 0)
                mark_price = float(item.get("markPrice", 0) or 0)
                implied_vol = float(item.get("markIv", 0) or 0)

                options.append({
                    "symbol": item["symbol"],
                    "strike": strike,
                    "expiry": T,
                    "expiry_date": expiry_date.strftime("%Y-%m-%d"),
                    "option_type": opt_type,
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2 if bid > 0 and ask > 0 else mark_price,
                    "mark_price": mark_price,
                    "implied_vol": implied_vol,
                    "delta": float(item.get("delta", 0) or 0),
                    "gamma": float(item.get("gamma", 0) or 0),
                    "vega": float(item.get("vega", 0) or 0),
                    "theta": float(item.get("theta", 0) or 0),
                    "volume": float(item.get("volume24h", 0) or 0),
                    "open_interest": float(item.get("openInterest", 0) or 0),
                })

            return options

        except requests.RequestException as e:
            print(f"Failed to fetch Bybit data: {e}")
            return []

    def fetch_spot_price(self) -> float:
        """Fetch current spot price from Bybit."""
        url = f"{self.BASE_URL}/v5/market/tickers"
        params = {"category": "spot", "symbol": f"{self.symbol}USDT"}

        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            if data["retCode"] == 0 and data["result"]["list"]:
                return float(data["result"]["list"][0]["lastPrice"])
        except (requests.RequestException, KeyError, IndexError):
            pass

        # Fallback
        return 50000.0 if self.symbol == "BTC" else 3000.0

    def fetch_klines(
        self, interval: str = "D", limit: int = 200
    ) -> pd.DataFrame:
        """
        Fetch OHLCV kline data for the spot market.

        Args:
            interval: Candle interval ('1', '5', '15', '60', 'D', 'W')
            limit: Number of candles

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        url = f"{self.BASE_URL}/v5/market/kline"
        params = {
            "category": "spot",
            "symbol": f"{self.symbol}USDT",
            "interval": interval,
            "limit": limit,
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if data["retCode"] != 0:
                return pd.DataFrame()

            rows = []
            for item in data["result"]["list"]:
                rows.append({
                    "timestamp": pd.to_datetime(int(item[0]), unit="ms"),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                })

            df = pd.DataFrame(rows)
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df

        except requests.RequestException as e:
            print(f"Failed to fetch klines: {e}")
            return pd.DataFrame()

    def get_option_chain(self) -> pd.DataFrame:
        """Get full option chain as a DataFrame."""
        options = self.fetch_tickers()
        if not options:
            return pd.DataFrame()
        return pd.DataFrame(options)

    def get_calibration_data(
        self,
        min_volume: float = 0.0,
        min_oi: float = 0.0,
        option_type: str = "call",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data formatted for Heston calibration.

        Returns:
            (strikes, expiries, implied_vols) as numpy arrays
        """
        df = self.get_option_chain()
        if df.empty:
            return np.array([]), np.array([]), np.array([])

        # Filter
        mask = (
            (df["option_type"] == option_type)
            & (df["implied_vol"] > 0)
            & (df["volume"] >= min_volume)
            & (df["open_interest"] >= min_oi)
        )
        df = df[mask]

        if df.empty:
            return np.array([]), np.array([]), np.array([])

        return (
            df["strike"].values,
            df["expiry"].values,
            df["implied_vol"].values,
        )


class SyntheticStockData:
    """
    Generate synthetic stock option data for testing.

    Produces option chains with known Heston parameters,
    useful for validating the PINN and calibration routines.
    """

    def __init__(
        self,
        S: float = 100.0,
        r: float = 0.05,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma: float = 0.3,
        rho: float = -0.7,
        v0: float = 0.04,
    ):
        self.S = S
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0

    def generate_option_chain(
        self,
        strikes: Optional[np.ndarray] = None,
        expiries: Optional[np.ndarray] = None,
        noise_level: float = 0.0,
    ) -> pd.DataFrame:
        """
        Generate synthetic option chain with Heston prices.

        Args:
            strikes: Strike prices (default: 80-120% of spot)
            expiries: Expiry times in years (default: 1m to 1y)
            noise_level: Std of Gaussian noise added to prices (for realism)

        Returns:
            DataFrame with option data
        """
        from heston_analytical import heston_call_price, implied_volatility

        if strikes is None:
            strikes = self.S * np.array([0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20])
        if expiries is None:
            expiries = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0])

        rows = []
        for T in expiries:
            for K in strikes:
                price = heston_call_price(
                    self.S, K, T, self.r, self.v0,
                    self.kappa, self.theta, self.sigma, self.rho
                )

                # Add noise if requested
                if noise_level > 0:
                    price = max(price + np.random.normal(0, noise_level * price), 0.01)

                iv = implied_volatility(price, self.S, K, T, self.r, "call")

                rows.append({
                    "strike": K,
                    "expiry": T,
                    "expiry_days": int(T * 365),
                    "option_type": "call",
                    "price": price,
                    "implied_vol": iv if not np.isnan(iv) else 0.0,
                    "moneyness": K / self.S,
                    "spot": self.S,
                })

        return pd.DataFrame(rows)

    def generate_training_data(
        self, n_points: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate random (S, v, t, V) training pairs using analytical Heston.

        Returns:
            (S_arr, v_arr, t_arr, V_arr) as numpy arrays
        """
        from heston_analytical import heston_call_price

        S_max = self.S * 2
        v_max = self.theta * 5

        S_arr = np.random.uniform(self.S * 0.5, S_max, n_points)
        v_arr = np.random.uniform(0.005, v_max, n_points)
        t_arr = np.random.uniform(0, 1.0, n_points)
        V_arr = np.zeros(n_points)

        K = self.S  # ATM strike

        for i in range(n_points):
            T_remaining = 1.0 - t_arr[i]
            if T_remaining < 0.001:
                V_arr[i] = max(S_arr[i] - K, 0)
            else:
                V_arr[i] = heston_call_price(
                    S_arr[i], K, T_remaining, self.r, v_arr[i],
                    self.kappa, self.theta, self.sigma, self.rho
                )

        return S_arr, v_arr, t_arr, V_arr


class RealizedVolatilityEstimator:
    """
    Estimate realized variance from historical price data.

    Uses close-to-close and Garman-Klass estimators.
    """

    @staticmethod
    def close_to_close(prices: np.ndarray, annualize: bool = True) -> float:
        """
        Close-to-close realized variance estimator.

        Args:
            prices: Array of closing prices
            annualize: If True, annualize (assuming 252 trading days)

        Returns:
            Realized variance estimate
        """
        log_returns = np.diff(np.log(prices))
        var = np.var(log_returns, ddof=1)
        if annualize:
            var *= 252
        return var

    @staticmethod
    def garman_klass(
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        annualize: bool = True,
    ) -> float:
        """
        Garman-Klass realized variance estimator (more efficient than close-to-close).

        Args:
            opens, highs, lows, closes: OHLC price arrays
            annualize: If True, annualize

        Returns:
            Realized variance estimate
        """
        n = len(closes)
        log_hl = np.log(highs / lows)
        log_co = np.log(closes / opens)

        var = (1 / n) * np.sum(
            0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        )

        if annualize:
            var *= 252
        return var


def load_data(
    source: str = "synthetic",
    symbol: str = "BTC",
    **kwargs,
) -> Dict:
    """
    Unified data loading function.

    Args:
        source: 'bybit' for live Bybit data, 'synthetic' for generated data
        symbol: Asset symbol (BTC, ETH for Bybit)

    Returns:
        Dictionary with keys:
            'spot': current spot price
            'options': DataFrame of option chain
            'strikes': array of strikes
            'expiries': array of expiries
            'implied_vols': array of implied vols
            'params': dict of true/estimated Heston parameters
    """
    if source == "bybit":
        loader = BybitOptionsLoader(symbol=symbol)
        spot = loader.fetch_spot_price()
        options_df = loader.get_option_chain()
        strikes, expiries, ivs = loader.get_calibration_data()

        # Estimate v0 from ATM options
        if len(ivs) > 0:
            v0_est = np.median(ivs) ** 2
        else:
            v0_est = 0.64 if symbol == "BTC" else 0.36

        return {
            "source": "bybit",
            "symbol": symbol,
            "spot": spot,
            "options": options_df,
            "strikes": strikes,
            "expiries": expiries,
            "implied_vols": ivs,
            "params": {
                "v0": v0_est,
                "kappa": 5.0,
                "theta": v0_est,
                "sigma": 1.5,
                "rho": -0.2,
                "r": 0.05,
            },
        }

    else:
        # Synthetic stock data
        synth = SyntheticStockData(**kwargs)
        options_df = synth.generate_option_chain()

        return {
            "source": "synthetic",
            "symbol": "STOCK",
            "spot": synth.S,
            "options": options_df,
            "strikes": options_df["strike"].unique(),
            "expiries": options_df["expiry"].unique(),
            "implied_vols": options_df["implied_vol"].values,
            "params": {
                "v0": synth.v0,
                "kappa": synth.kappa,
                "theta": synth.theta,
                "sigma": synth.sigma,
                "rho": synth.rho,
                "r": synth.r,
            },
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Data Loader Demo")
    print("=" * 60)

    # 1. Synthetic stock data
    print("\n--- Synthetic Stock Options ---")
    data = load_data(source="synthetic")
    print(f"Spot: {data['spot']}")
    print(f"Options: {len(data['options'])} contracts")
    print(f"Strikes: {data['strikes']}")
    print(f"Expiries: {data['expiries']}")
    print(f"IV range: {data['implied_vols'].min():.4f} - {data['implied_vols'].max():.4f}")
    print(f"\nSample options:\n{data['options'].head(10).to_string()}")

    # 2. Bybit data (if available)
    print("\n--- Bybit BTC Options ---")
    btc_data = load_data(source="bybit", symbol="BTC")
    print(f"Spot: ${btc_data['spot']:,.2f}")
    print(f"Options fetched: {len(btc_data['options'])}")
    if not btc_data["options"].empty:
        print(f"Strikes: {len(btc_data['strikes'])} unique")
        print(f"Expiries: {len(btc_data['expiries'])} unique")
        print(f"\nSample:\n{btc_data['options'].head(5).to_string()}")
    else:
        print("(No live data available -- using synthetic fallback)")

    # 3. Realized volatility estimation
    print("\n--- Realized Volatility Estimator ---")
    np.random.seed(42)
    # Simulate some prices
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.02, 100)))
    rv = RealizedVolatilityEstimator.close_to_close(prices)
    print(f"Close-to-close realized variance: {rv:.4f}")
    print(f"Realized vol: {np.sqrt(rv)*100:.1f}%")
