"""
heston_analytical.py - Semi-Analytical Heston Pricing via Characteristic Function

Implements the Heston (1993) / Lewis (2000) approach for European option pricing.
The call price is computed as:
    C = S * P1 - K * exp(-r*T) * P2

where P1 and P2 are obtained via numerical integration of the characteristic function.

This serves as a benchmark to validate the PINN solver.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from typing import Tuple, Optional


def heston_characteristic_function(
    phi: complex,
    S: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    r: float,
    T: float,
    j: int,
) -> complex:
    """
    Heston characteristic function f_j(phi) for j=1,2.

    Uses the formulation from Albrecher et al. (2007) with the
    "little Heston trap" fix for numerical stability.

    Args:
        phi: Fourier variable
        S: Spot price
        v0: Initial variance
        kappa: Mean-reversion speed
        theta: Long-run variance
        sigma: Vol-of-vol
        rho: Correlation
        r: Risk-free rate
        T: Time to expiry
        j: Index (1 or 2)

    Returns:
        Complex value of the characteristic function
    """
    if j == 1:
        u = 0.5
        b = kappa - rho * sigma
    else:
        u = -0.5
        b = kappa

    a = kappa * theta
    x = np.log(S)

    d = np.sqrt(
        (rho * sigma * 1j * phi - b) ** 2
        - sigma**2 * (2 * u * 1j * phi - phi**2)
    )

    # Use the "little Heston trap" formulation for stability
    g = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)

    # Ensure we pick the right branch of the square root
    # by using the formulation that avoids division by near-zero
    exp_dT = np.exp(d * T)

    C = r * 1j * phi * T + (a / sigma**2) * (
        (b - rho * sigma * 1j * phi + d) * T
        - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
    )

    D = ((b - rho * sigma * 1j * phi + d) / sigma**2) * (
        (1.0 - exp_dT) / (1.0 - g * exp_dT)
    )

    return np.exp(C + D * v0 + 1j * phi * x)


def heston_integrand(
    phi: float,
    S: float,
    K: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    r: float,
    T: float,
    j: int,
) -> float:
    """
    Integrand for the Heston pricing formula.

    P_j = 0.5 + (1/pi) * integral_0^inf Re[e^{-i*phi*ln(K)} * f_j(phi) / (i*phi)] dphi
    """
    f = heston_characteristic_function(phi, S, v0, kappa, theta, sigma, rho, r, T, j)
    integrand = np.real(np.exp(-1j * phi * np.log(K)) * f / (1j * phi))
    return integrand


def heston_P(
    S: float,
    K: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    r: float,
    T: float,
    j: int,
    N_quad: int = 256,
) -> float:
    """
    Compute P_j (j=1 or 2) via numerical integration.

    Uses scipy.integrate.quad for adaptive quadrature.
    """
    integral, _ = quad(
        heston_integrand,
        1e-8,
        200.0,  # Upper limit (effectively infinity)
        args=(S, K, v0, kappa, theta, sigma, rho, r, T, j),
        limit=N_quad,
    )
    return 0.5 + (1.0 / np.pi) * integral


def heston_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
) -> float:
    """
    Semi-analytical European call option price under the Heston model.

    C = S * P1 - K * exp(-r*T) * P2

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        v0: Initial variance
        kappa: Mean-reversion speed
        theta: Long-run variance
        sigma: Vol-of-vol
        rho: Correlation

    Returns:
        European call price
    """
    if T <= 0:
        return max(S - K, 0.0)

    P1 = heston_P(S, K, v0, kappa, theta, sigma, rho, r, T, j=1)
    P2 = heston_P(S, K, v0, kappa, theta, sigma, rho, r, T, j=2)

    call_price = S * P1 - K * np.exp(-r * T) * P2
    return max(call_price, 0.0)


def heston_put_price(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
) -> float:
    """
    European put price via put-call parity: P = C - S + K*exp(-r*T).
    """
    call = heston_call_price(S, K, T, r, v0, kappa, theta, sigma, rho)
    return call - S + K * np.exp(-r * T)


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price for comparison."""
    from scipy.stats import norm

    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-8,
) -> float:
    """
    Compute Black-Scholes implied volatility from an option price.

    Uses Brent's method for root finding.

    Args:
        market_price: Observed option price
        S: Spot price
        K: Strike price
        T: Time to expiry
        r: Risk-free rate
        option_type: 'call' or 'put'
        tol: Convergence tolerance

    Returns:
        Implied volatility (annualized)
    """
    from scipy.stats import norm

    if T <= 0:
        return 0.0

    def bs_price(sigma_val):
        if sigma_val <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma_val**2) * T) / (sigma_val * np.sqrt(T))
        d2 = d1 - sigma_val * np.sqrt(T)
        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def objective(sigma_val):
        return bs_price(sigma_val) - market_price

    # Intrinsic value check
    if option_type == "call":
        intrinsic = max(S - K * np.exp(-r * T), 0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0)

    if market_price <= intrinsic + tol:
        return 1e-6  # Near-zero vol

    try:
        iv = brentq(objective, 1e-6, 10.0, xtol=tol)
        return iv
    except ValueError:
        return np.nan


def heston_implied_vol_surface(
    S: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    r: float,
    strikes: np.ndarray,
    expiries: np.ndarray,
) -> np.ndarray:
    """
    Generate the implied volatility surface from Heston model prices.

    Args:
        S: Spot price
        v0: Initial variance
        kappa, theta, sigma, rho: Heston parameters
        r: Risk-free rate
        strikes: Array of strike prices
        expiries: Array of expiry times

    Returns:
        2D array of implied volatilities, shape (len(strikes), len(expiries))
    """
    iv_surface = np.zeros((len(strikes), len(expiries)))

    for i, K in enumerate(strikes):
        for j, T in enumerate(expiries):
            price = heston_call_price(S, K, T, r, v0, kappa, theta, sigma, rho)
            iv = implied_volatility(price, S, K, T, r, option_type="call")
            iv_surface[i, j] = iv

    return iv_surface


if __name__ == "__main__":
    # Heston parameters (typical equity)
    S = 100.0
    K = 100.0
    T = 0.5
    r = 0.05
    v0 = 0.04
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7

    print("=" * 60)
    print("Semi-Analytical Heston Pricing")
    print("=" * 60)

    # Check Feller condition
    feller = 2 * kappa * theta > sigma**2
    print(f"Feller condition (2*kappa*theta > sigma^2): {feller}")
    print(f"  2*kappa*theta = {2*kappa*theta:.4f}, sigma^2 = {sigma**2:.4f}")

    # Compute prices
    call_price = heston_call_price(S, K, T, r, v0, kappa, theta, sigma, rho)
    put_price = heston_put_price(S, K, T, r, v0, kappa, theta, sigma, rho)

    print(f"\nATM Call Price: {call_price:.4f}")
    print(f"ATM Put Price:  {put_price:.4f}")

    # Black-Scholes comparison (using sqrt(v0) as constant vol)
    bs_price = black_scholes_call(S, K, T, r, np.sqrt(v0))
    print(f"BS Call Price (sigma={np.sqrt(v0):.2f}): {bs_price:.4f}")

    # Implied volatility smile
    print("\nImplied Volatility Smile:")
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
    for strike in strikes:
        price = heston_call_price(S, strike, T, r, v0, kappa, theta, sigma, rho)
        iv = implied_volatility(price, S, strike, T, r)
        print(f"  K={strike:6.0f}: Price={price:8.4f}, IV={iv*100:6.2f}%")

    # Crypto parameters (typical BTC)
    print("\n" + "=" * 60)
    print("BTC Heston Parameters")
    print("=" * 60)
    S_btc = 50000.0
    K_btc = 50000.0
    v0_btc = 0.64  # ~80% annualized vol
    kappa_btc = 5.0
    theta_btc = 0.50
    sigma_btc = 1.5
    rho_btc = -0.2

    call_btc = heston_call_price(S_btc, K_btc, T, r, v0_btc, kappa_btc, theta_btc, sigma_btc, rho_btc)
    print(f"BTC ATM Call (6m): ${call_btc:,.2f}")

    feller_btc = 2 * kappa_btc * theta_btc > sigma_btc**2
    print(f"Feller condition: {feller_btc} (2kT={2*kappa_btc*theta_btc:.2f}, s2={sigma_btc**2:.2f})")
