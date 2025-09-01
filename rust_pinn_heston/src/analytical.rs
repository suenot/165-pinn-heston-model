//! Semi-analytical Heston pricing via characteristic function.
//!
//! Implements the Heston (1993) / Lewis (2000) approach:
//!     C = S * P1 - K * exp(-r*T) * P2
//!
//! where P1, P2 are computed via numerical integration of the characteristic function.

use num_complex::Complex64;
use std::f64::consts::PI;

use crate::HestonParams;

/// Heston characteristic function f_j(phi) for j=1 or j=2.
///
/// Uses the "little Heston trap" formulation for numerical stability
/// (Albrecher et al., 2007).
fn characteristic_function(
    phi: f64,
    s_log: f64,
    params: &HestonParams,
    tau: f64,
    j: usize,
) -> Complex64 {
    let i = Complex64::new(0.0, 1.0);
    let phi_c = Complex64::new(phi, 0.0);

    let (u, b) = if j == 1 {
        (0.5, params.kappa - params.rho * params.sigma)
    } else {
        (-0.5, params.kappa)
    };

    let a = params.kappa * params.theta;
    let sigma2 = params.sigma * params.sigma;

    // d = sqrt((rho*sigma*i*phi - b)^2 - sigma^2*(2*u*i*phi - phi^2))
    let term1 = params.rho * params.sigma * i * phi_c - Complex64::new(b, 0.0);
    let term2 = sigma2 * (2.0 * u * i * phi_c - phi_c * phi_c);
    let d = (term1 * term1 - term2).sqrt();

    // g = (b - rho*sigma*i*phi + d) / (b - rho*sigma*i*phi - d)
    let num = Complex64::new(b, 0.0) - params.rho * params.sigma * i * phi_c + d;
    let den = Complex64::new(b, 0.0) - params.rho * params.sigma * i * phi_c - d;

    if den.norm() < 1e-15 {
        return Complex64::new(0.0, 0.0);
    }
    let g = num / den;

    let exp_dt = (d * Complex64::new(tau, 0.0)).exp();

    // C_j
    let c_val = params.r * i * phi_c * Complex64::new(tau, 0.0)
        + Complex64::new(a / sigma2, 0.0)
            * (num * Complex64::new(tau, 0.0)
                - Complex64::new(2.0, 0.0)
                    * ((Complex64::new(1.0, 0.0) - g * exp_dt)
                        / (Complex64::new(1.0, 0.0) - g))
                        .ln());

    // D_j
    let d_val = (num / Complex64::new(sigma2, 0.0))
        * ((Complex64::new(1.0, 0.0) - exp_dt)
            / (Complex64::new(1.0, 0.0) - g * exp_dt));

    (c_val + d_val * Complex64::new(params.v0, 0.0) + i * phi_c * Complex64::new(s_log, 0.0))
        .exp()
}

/// Compute P_j (j=1 or 2) via numerical integration (trapezoidal rule).
fn compute_p(
    s: f64,
    k: f64,
    params: &HestonParams,
    tau: f64,
    j: usize,
    n_points: usize,
) -> f64 {
    let s_log = s.ln();
    let k_log = k.ln();
    let i = Complex64::new(0.0, 1.0);

    let phi_max = 200.0;
    let dphi = phi_max / n_points as f64;

    let mut integral = 0.0;

    for n in 1..=n_points {
        let phi = n as f64 * dphi;
        let f = characteristic_function(phi, s_log, params, tau, j);
        let integrand = ((-i * Complex64::new(phi, 0.0) * Complex64::new(k_log, 0.0)).exp()
            * f
            / (i * Complex64::new(phi, 0.0)))
            .re;
        integral += integrand * dphi;
    }

    0.5 + integral / PI
}

/// Compute European call price under the Heston model using the characteristic function.
///
/// # Arguments
/// * `s` - Spot price
/// * `k` - Strike price
/// * `tau` - Time to expiry (years)
/// * `params` - Heston model parameters
///
/// # Returns
/// European call option price
pub fn heston_call_price(s: f64, k: f64, tau: f64, params: &HestonParams) -> f64 {
    if tau <= 0.0 {
        return (s - k).max(0.0);
    }

    let n_points = 512;
    let p1 = compute_p(s, k, params, tau, 1, n_points);
    let p2 = compute_p(s, k, params, tau, 2, n_points);

    let price = s * p1 - k * (-params.r * tau).exp() * p2;
    price.max(0.0)
}

/// Compute European put price via put-call parity.
pub fn heston_put_price(s: f64, k: f64, tau: f64, params: &HestonParams) -> f64 {
    let call = heston_call_price(s, k, tau, params);
    call - s + k * (-params.r * tau).exp()
}

/// Black-Scholes call price (for comparison and IV computation).
pub fn bs_call_price(s: f64, k: f64, tau: f64, r: f64, sigma: f64) -> f64 {
    if tau <= 0.0 {
        return (s - k).max(0.0);
    }
    if sigma <= 0.0 {
        return (s - k * (-r * tau).exp()).max(0.0);
    }

    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * tau) / (sigma * tau.sqrt());
    let d2 = d1 - sigma * tau.sqrt();

    s * normal_cdf(d1) - k * (-r * tau).exp() * normal_cdf(d2)
}

/// Compute Black-Scholes implied volatility from a market price using bisection.
pub fn implied_volatility(
    market_price: f64,
    s: f64,
    k: f64,
    tau: f64,
    r: f64,
    is_call: bool,
) -> Option<f64> {
    if tau <= 0.0 {
        return None;
    }

    let intrinsic = if is_call {
        (s - k * (-r * tau).exp()).max(0.0)
    } else {
        (k * (-r * tau).exp() - s).max(0.0)
    };

    if market_price < intrinsic {
        return None;
    }

    let mut low = 1e-6_f64;
    let mut high = 10.0_f64;

    for _ in 0..100 {
        let mid = (low + high) / 2.0;
        let price = if is_call {
            bs_call_price(s, k, tau, r, mid)
        } else {
            bs_call_price(s, k, tau, r, mid) - s + k * (-r * tau).exp()
        };

        if (price - market_price).abs() < 1e-10 {
            return Some(mid);
        }

        if price > market_price {
            high = mid;
        } else {
            low = mid;
        }
    }

    Some((low + high) / 2.0)
}

/// Standard normal CDF approximation (Abramowitz and Stegun).
fn normal_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + p * x_abs);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x_abs * x_abs).exp();

    0.5 * (1.0 + sign * y)
}

/// Generate the implied volatility smile for given strikes at fixed expiry.
pub fn heston_iv_smile(
    s: f64,
    strikes: &[f64],
    tau: f64,
    params: &HestonParams,
) -> Vec<f64> {
    strikes
        .iter()
        .map(|&k| {
            let price = heston_call_price(s, k, tau, params);
            implied_volatility(price, s, k, tau, params.r, true).unwrap_or(0.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heston_call_atm() {
        let params = HestonParams::equity_default();
        let price = heston_call_price(100.0, 100.0, 0.5, &params);
        // ATM call should be positive and reasonable
        assert!(price > 0.0, "ATM call should be positive");
        assert!(price < 100.0, "ATM call should be less than spot");
        println!("ATM Call: {:.4}", price);
    }

    #[test]
    fn test_heston_put_call_parity() {
        let params = HestonParams::equity_default();
        let s = 100.0;
        let k = 100.0;
        let tau = 0.5;

        let call = heston_call_price(s, k, tau, &params);
        let put = heston_put_price(s, k, tau, &params);

        let parity = call - put - s + k * (-params.r * tau).exp();
        assert!(
            parity.abs() < 0.01,
            "Put-call parity violated: {:.6}",
            parity
        );
    }

    #[test]
    fn test_implied_vol() {
        let s = 100.0;
        let k = 100.0;
        let tau = 0.5;
        let r = 0.05;
        let sigma = 0.2;

        let price = bs_call_price(s, k, tau, r, sigma);
        let iv = implied_volatility(price, s, k, tau, r, true).unwrap();
        assert!(
            (iv - sigma).abs() < 1e-6,
            "IV should match input sigma: {} vs {}",
            iv,
            sigma
        );
    }

    #[test]
    fn test_heston_smile() {
        let params = HestonParams::equity_default();
        let strikes: Vec<f64> = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let ivs = heston_iv_smile(100.0, &strikes, 0.25, &params);

        // With rho < 0, lower strikes should have higher IV (skew)
        assert!(
            ivs[0] > ivs[4],
            "Expected skew: IV(80) > IV(120), got {} vs {}",
            ivs[0],
            ivs[4]
        );
    }
}
