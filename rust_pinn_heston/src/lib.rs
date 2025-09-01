//! # PINN Heston Model
//!
//! Physics-Informed Neural Network for the Heston Stochastic Volatility Model.
//!
//! This crate implements:
//! - A simple feedforward neural network (PINN) for option pricing
//! - Semi-analytical Heston pricing via characteristic function
//! - Heston parameter calibration
//! - Greeks computation via finite differences
//! - Bybit market data fetching
//!
//! ## The Heston PDE
//!
//! ```text
//! dV/dt + 0.5*v*S^2*d2V/dS2 + rho*sigma*v*S*d2V/dSdv
//!       + 0.5*sigma^2*v*d2V/dv2 + r*S*dV/dS + kappa*(theta-v)*dV/dv - rV = 0
//! ```

pub mod analytical;
pub mod calibration;
pub mod data;
pub mod greeks;
pub mod network;

use serde::{Deserialize, Serialize};

/// Heston model parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HestonParams {
    /// Mean-reversion speed of variance
    pub kappa: f64,
    /// Long-run variance level
    pub theta: f64,
    /// Volatility of variance (vol-of-vol)
    pub sigma: f64,
    /// Correlation between spot and variance Brownians
    pub rho: f64,
    /// Initial variance
    pub v0: f64,
    /// Risk-free interest rate
    pub r: f64,
}

impl HestonParams {
    pub fn new(kappa: f64, theta: f64, sigma: f64, rho: f64, v0: f64, r: f64) -> Self {
        Self {
            kappa,
            theta,
            sigma,
            rho,
            v0,
            r,
        }
    }

    /// Check the Feller condition: 2*kappa*theta > sigma^2.
    /// When satisfied, the variance process remains strictly positive.
    pub fn feller_condition(&self) -> bool {
        2.0 * self.kappa * self.theta > self.sigma * self.sigma
    }

    /// Feller ratio: 2*kappa*theta / sigma^2. Values > 1 satisfy the condition.
    pub fn feller_ratio(&self) -> f64 {
        2.0 * self.kappa * self.theta / (self.sigma * self.sigma)
    }

    /// Default equity-like parameters.
    pub fn equity_default() -> Self {
        Self {
            kappa: 2.0,
            theta: 0.04,
            sigma: 0.3,
            rho: -0.7,
            v0: 0.04,
            r: 0.05,
        }
    }

    /// Default crypto (BTC) parameters with higher vol-of-vol.
    pub fn crypto_default() -> Self {
        Self {
            kappa: 5.0,
            theta: 0.50,
            sigma: 1.5,
            rho: -0.2,
            v0: 0.64,
            r: 0.05,
        }
    }
}

/// Option specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionSpec {
    pub spot: f64,
    pub strike: f64,
    pub expiry: f64,  // Time to expiry in years
    pub option_type: OptionType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OptionType {
    Call,
    Put,
}

impl std::fmt::Display for OptionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptionType::Call => write!(f, "Call"),
            OptionType::Put => write!(f, "Put"),
        }
    }
}

/// Market data point for a single option.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketOption {
    pub symbol: String,
    pub strike: f64,
    pub expiry: f64,
    pub option_type: OptionType,
    pub bid: f64,
    pub ask: f64,
    pub mid: f64,
    pub mark_price: f64,
    pub implied_vol: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub volume: f64,
    pub open_interest: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feller_condition_equity() {
        let params = HestonParams::equity_default();
        // 2 * 2.0 * 0.04 = 0.16 > 0.09 = 0.3^2
        assert!(params.feller_condition());
        assert!(params.feller_ratio() > 1.0);
    }

    #[test]
    fn test_feller_condition_crypto() {
        let params = HestonParams::crypto_default();
        // 2 * 5.0 * 0.50 = 5.0 > 2.25 = 1.5^2
        assert!(params.feller_condition());
    }

    #[test]
    fn test_feller_condition_violated() {
        let params = HestonParams::new(0.5, 0.02, 1.0, -0.5, 0.04, 0.05);
        // 2 * 0.5 * 0.02 = 0.02 < 1.0 = 1.0^2
        assert!(!params.feller_condition());
    }
}
