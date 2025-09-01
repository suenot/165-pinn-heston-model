//! Greeks computation via finite differences through the PINN.
//!
//! Computes:
//! - Delta (dV/dS), Gamma (d2V/dS2)
//! - Vega (dV/dv), Volga (d2V/dv2)
//! - Theta (dV/dt)
//! - Vanna (d2V/dSdv)
//! - Charm (d2V/dSdt)
//! - Veta (d2V/dvdt)

use crate::network::HestonPINN;
use serde::{Deserialize, Serialize};

/// Complete set of option Greeks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Greeks {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega_v: f64,   // dV/dv (per unit variance)
    pub vega: f64,      // dV/dsigma * 0.01 (per 1% vol change)
    pub vanna: f64,     // d2V/dSdv
    pub volga: f64,     // d2V/dv2
    pub charm: f64,     // d2V/dSdt
    pub veta: f64,      // d2V/dvdt
}

impl std::fmt::Display for Greeks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  Price:    {:.6}", self.price)?;
        writeln!(f, "  Delta:    {:.6}", self.delta)?;
        writeln!(f, "  Gamma:    {:.6}", self.gamma)?;
        writeln!(f, "  Theta:    {:.6}", self.theta)?;
        writeln!(f, "  Vega(dv): {:.6}", self.vega_v)?;
        writeln!(f, "  Vega(1%): {:.6}", self.vega)?;
        writeln!(f, "  Vanna:    {:.6}", self.vanna)?;
        writeln!(f, "  Volga:    {:.6}", self.volga)?;
        writeln!(f, "  Charm:    {:.6}", self.charm)?;
        writeln!(f, "  Veta:     {:.6}", self.veta)
    }
}

/// Compute all Greeks at a single point using finite differences.
///
/// # Arguments
/// * `model` - Trained PINN model
/// * `s` - Spot price
/// * `v` - Variance
/// * `t` - Current time
/// * `ds` - Step size for spot (default: 0.01 * S)
/// * `dv` - Step size for variance (default: 0.001)
/// * `dt` - Step size for time (default: 0.001)
pub fn compute_greeks(
    model: &HestonPINN,
    s: f64,
    v: f64,
    t: f64,
    ds: Option<f64>,
    dv: Option<f64>,
    dt: Option<f64>,
) -> Greeks {
    let ds = ds.unwrap_or(s * 0.001);
    let dv = dv.unwrap_or(0.0005);
    let dt = dt.unwrap_or(0.001);

    let v_pos = v.max(dv * 2.0);

    // Helper closures
    let p = |s_val: f64, v_val: f64, t_val: f64| -> f64 {
        model.forward(
            s_val.max(1e-6),
            v_val.max(1e-8),
            t_val.clamp(0.0, model.t_max),
        )
    };

    let price = p(s, v_pos, t);

    // --- First-order derivatives ---

    // Delta = dV/dS
    let delta = (p(s + ds, v_pos, t) - p(s - ds, v_pos, t)) / (2.0 * ds);

    // Vega_v = dV/dv
    let vega_v = (p(s, v_pos + dv, t) - p(s, (v_pos - dv).max(1e-8), t)) / (2.0 * dv);

    // Vega in standard terms (per 1% vol move)
    let vega = vega_v * 2.0 * v_pos.sqrt() * 0.01;

    // Theta = -dV/dt (negative sign convention)
    let theta = -(p(s, v_pos, t + dt) - p(s, v_pos, (t - dt).max(0.0))) / (2.0 * dt);

    // --- Second-order derivatives ---

    // Gamma = d2V/dS2
    let gamma = (p(s + ds, v_pos, t) - 2.0 * price + p(s - ds, v_pos, t)) / (ds * ds);

    // Volga = d2V/dv2
    let volga =
        (p(s, v_pos + dv, t) - 2.0 * price + p(s, (v_pos - dv).max(1e-8), t)) / (dv * dv);

    // --- Cross-derivatives ---

    // Vanna = d2V/dSdv
    let vanna = (p(s + ds, v_pos + dv, t) - p(s + ds, (v_pos - dv).max(1e-8), t)
        - p(s - ds, v_pos + dv, t)
        + p(s - ds, (v_pos - dv).max(1e-8), t))
        / (4.0 * ds * dv);

    // Charm = d2V/dSdt
    let charm = -(p(s + ds, v_pos, t + dt) - p(s + ds, v_pos, (t - dt).max(0.0))
        - p(s - ds, v_pos, t + dt)
        + p(s - ds, v_pos, (t - dt).max(0.0)))
        / (4.0 * ds * dt);

    // Veta = d2V/dvdt
    let veta = -(p(s, v_pos + dv, t + dt) - p(s, v_pos + dv, (t - dt).max(0.0))
        - p(s, (v_pos - dv).max(1e-8), t + dt)
        + p(s, (v_pos - dv).max(1e-8), (t - dt).max(0.0)))
        / (4.0 * dv * dt);

    Greeks {
        price,
        delta,
        gamma,
        theta,
        vega_v,
        vega,
        vanna,
        volga,
        charm,
        veta,
    }
}

/// Compute Greeks using the analytical Heston model (for comparison).
pub fn analytical_greeks(
    s: f64,
    k: f64,
    tau: f64,
    params: &crate::HestonParams,
    ds: Option<f64>,
    dv: Option<f64>,
    dt: Option<f64>,
) -> Greeks {
    use crate::analytical::heston_call_price;

    let ds = ds.unwrap_or(s * 0.001);
    let dv = dv.unwrap_or(0.0005);
    let dt = dt.unwrap_or(0.001);

    let p = |s_val: f64, v0: f64, tau_val: f64| -> f64 {
        let mut params_copy = params.clone();
        params_copy.v0 = v0;
        heston_call_price(s_val, k, tau_val.max(1e-6), &params_copy)
    };

    let price = p(s, params.v0, tau);
    let v0 = params.v0;

    let delta = (p(s + ds, v0, tau) - p(s - ds, v0, tau)) / (2.0 * ds);
    let gamma = (p(s + ds, v0, tau) - 2.0 * price + p(s - ds, v0, tau)) / (ds * ds);
    let vega_v = (p(s, v0 + dv, tau) - p(s, (v0 - dv).max(1e-8), tau)) / (2.0 * dv);
    let vega = vega_v * 2.0 * v0.sqrt() * 0.01;
    let theta = -(p(s, v0, tau - dt) - p(s, v0, tau + dt)) / (2.0 * dt);
    let volga = (p(s, v0 + dv, tau) - 2.0 * price + p(s, (v0 - dv).max(1e-8), tau)) / (dv * dv);
    let vanna = (p(s + ds, v0 + dv, tau) - p(s + ds, (v0 - dv).max(1e-8), tau)
        - p(s - ds, v0 + dv, tau)
        + p(s - ds, (v0 - dv).max(1e-8), tau))
        / (4.0 * ds * dv);

    let charm = -(p(s + ds, v0, tau - dt) - p(s + ds, v0, tau + dt)
        - p(s - ds, v0, tau - dt) + p(s - ds, v0, tau + dt))
        / (4.0 * ds * dt);

    let veta = -(p(s, v0 + dv, tau - dt) - p(s, v0 + dv, tau + dt)
        - p(s, (v0 - dv).max(1e-8), tau - dt) + p(s, (v0 - dv).max(1e-8), tau + dt))
        / (4.0 * dv * dt);

    Greeks {
        price,
        delta,
        gamma,
        theta,
        vega_v,
        vega,
        vanna,
        volga,
        charm,
        veta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greeks_analytical() {
        let params = crate::HestonParams::equity_default();
        let greeks = analytical_greeks(100.0, 100.0, 0.5, &params, None, None, None);

        println!("Analytical Greeks (ATM, T=0.5):\n{}", greeks);

        // Basic sanity checks
        assert!(greeks.delta > 0.0 && greeks.delta < 1.0, "Delta out of range");
        assert!(greeks.gamma > 0.0, "Gamma should be positive");
        assert!(greeks.theta < 0.0, "Theta should be negative for long calls");
    }
}
