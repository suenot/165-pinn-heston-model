//! Heston model calibration to market implied volatilities.
//!
//! Minimizes:
//!     sum_i w_i * (sigma_model(K_i, T_i; params) - sigma_mkt(K_i, T_i))^2
//!
//! Uses differential evolution (population-based optimizer) for global search,
//! followed by Nelder-Mead for local refinement.

use crate::analytical::{heston_call_price, implied_volatility};
use crate::HestonParams;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Market observation for calibration.
#[derive(Debug, Clone)]
pub struct CalibrationPoint {
    pub strike: f64,
    pub expiry: f64,
    pub market_iv: f64,
    pub weight: f64,
}

/// Calibration result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    pub params: HestonParams,
    pub objective: f64,
    pub feller_satisfied: bool,
    pub feller_ratio: f64,
    pub n_iterations: usize,
    pub n_evals: usize,
}

impl std::fmt::Display for CalibrationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Calibration Result:")?;
        writeln!(f, "  kappa = {:.4}", self.params.kappa)?;
        writeln!(
            f,
            "  theta = {:.6} (vol = {:.1}%)",
            self.params.theta,
            self.params.theta.sqrt() * 100.0
        )?;
        writeln!(f, "  sigma = {:.4}", self.params.sigma)?;
        writeln!(f, "  rho   = {:.4}", self.params.rho)?;
        writeln!(
            f,
            "  v0    = {:.6} (vol = {:.1}%)",
            self.params.v0,
            self.params.v0.sqrt() * 100.0
        )?;
        writeln!(f, "  Feller: {} (ratio={:.2})", self.feller_satisfied, self.feller_ratio)?;
        writeln!(f, "  Objective: {:.8}", self.objective)?;
        writeln!(f, "  Iterations: {}", self.n_iterations)?;
        writeln!(f, "  Function evals: {}", self.n_evals)
    }
}

/// Objective function for calibration.
fn calibration_objective(
    params_vec: &[f64],
    spot: f64,
    rate: f64,
    market_data: &[CalibrationPoint],
    feller_penalty: f64,
) -> f64 {
    let kappa = params_vec[0];
    let theta = params_vec[1];
    let sigma = params_vec[2];
    let rho = params_vec[3];
    let v0 = params_vec[4];

    // Feasibility check
    if kappa <= 0.0 || theta <= 0.0 || sigma <= 0.0 || v0 <= 0.0 {
        return 1e10;
    }
    if rho.abs() >= 1.0 {
        return 1e10;
    }

    let params = HestonParams::new(kappa, theta, sigma, rho, v0, rate);
    let mut total_error = 0.0;
    let mut total_weight = 0.0;

    for point in market_data {
        let price = heston_call_price(spot, point.strike, point.expiry, &params);
        if let Some(model_iv) = implied_volatility(price, spot, point.strike, point.expiry, rate, true) {
            let err = model_iv - point.market_iv;
            total_error += point.weight * err * err;
            total_weight += point.weight;
        } else {
            total_error += point.weight * 1.0;
            total_weight += point.weight;
        }
    }

    if total_weight <= 0.0 {
        return 1e10;
    }

    let obj = total_error / total_weight;

    // Feller penalty
    let feller_violation = (sigma * sigma - 2.0 * kappa * theta).max(0.0);
    obj + feller_penalty * feller_violation
}

/// Simple differential evolution optimizer for Heston calibration.
///
/// Population-based global optimizer suitable for the 5D parameter space.
pub fn calibrate_differential_evolution(
    spot: f64,
    rate: f64,
    market_data: &[CalibrationPoint],
    max_iterations: usize,
    population_size: usize,
    feller_penalty: f64,
) -> CalibrationResult {
    let mut rng = rand::thread_rng();

    // Bounds: [kappa, theta, sigma, rho, v0]
    let bounds = [
        (0.1, 20.0),   // kappa
        (0.001, 2.0),  // theta
        (0.05, 5.0),   // sigma
        (-0.99, 0.99), // rho
        (0.001, 2.0),  // v0
    ];
    let dim = bounds.len();

    // Initialize population
    let mut population: Vec<Vec<f64>> = (0..population_size)
        .map(|_| {
            bounds
                .iter()
                .map(|&(lo, hi)| rng.gen_range(lo..hi))
                .collect()
        })
        .collect();

    let mut fitness: Vec<f64> = population
        .iter()
        .map(|p| calibration_objective(p, spot, rate, market_data, feller_penalty))
        .collect();

    let mut n_evals = population_size;
    let cr = 0.9; // Crossover rate
    let f_scale = 0.8; // Differential weight

    for iter in 0..max_iterations {
        for i in 0..population_size {
            // Select three distinct random indices (not i)
            let mut indices = Vec::new();
            while indices.len() < 3 {
                let idx = rng.gen_range(0..population_size);
                if idx != i && !indices.contains(&idx) {
                    indices.push(idx);
                }
            }
            let (a, b, c) = (indices[0], indices[1], indices[2]);

            // Create trial vector
            let j_rand = rng.gen_range(0..dim);
            let mut trial = population[i].clone();

            for j in 0..dim {
                if rng.gen::<f64>() < cr || j == j_rand {
                    trial[j] = population[a][j]
                        + f_scale * (population[b][j] - population[c][j]);
                    // Clip to bounds
                    trial[j] = trial[j].clamp(bounds[j].0, bounds[j].1);
                }
            }

            let trial_fitness =
                calibration_objective(&trial, spot, rate, market_data, feller_penalty);
            n_evals += 1;

            if trial_fitness < fitness[i] {
                population[i] = trial;
                fitness[i] = trial_fitness;
            }
        }

        // Check convergence
        let best_fitness = fitness.iter().cloned().fold(f64::INFINITY, f64::min);
        let mean_fitness: f64 = fitness.iter().sum::<f64>() / fitness.len() as f64;

        if iter % 50 == 0 {
            eprintln!(
                "  DE iter {}: best={:.8}, mean={:.6}",
                iter, best_fitness, mean_fitness
            );
        }

        if mean_fitness - best_fitness < 1e-10 {
            break;
        }
    }

    // Find best
    let best_idx = fitness
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let best = &population[best_idx];
    let params = HestonParams::new(best[0], best[1], best[2], best[3], best[4], rate);

    CalibrationResult {
        feller_satisfied: params.feller_condition(),
        feller_ratio: params.feller_ratio(),
        params,
        objective: fitness[best_idx],
        n_iterations: max_iterations,
        n_evals,
    }
}

/// Nelder-Mead simplex optimizer for local refinement.
pub fn calibrate_nelder_mead(
    spot: f64,
    rate: f64,
    market_data: &[CalibrationPoint],
    initial_guess: &[f64; 5],
    max_iterations: usize,
    feller_penalty: f64,
) -> CalibrationResult {
    let dim = 5;
    let alpha = 1.0; // Reflection
    let gamma = 2.0; // Expansion
    let rho_nm = 0.5; // Contraction
    let sigma_nm = 0.5; // Shrink

    // Bounds for clamping
    let bounds = [
        (0.1, 20.0),
        (0.001, 2.0),
        (0.05, 5.0),
        (-0.99, 0.99),
        (0.001, 2.0),
    ];

    let clamp = |x: &mut Vec<f64>| {
        for (i, val) in x.iter_mut().enumerate() {
            *val = val.clamp(bounds[i].0, bounds[i].1);
        }
    };

    // Initialize simplex
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(dim + 1);
    simplex.push(initial_guess.to_vec());

    for i in 0..dim {
        let mut vertex = initial_guess.to_vec();
        vertex[i] *= 1.05;
        clamp(&mut vertex);
        simplex.push(vertex);
    }

    let mut values: Vec<f64> = simplex
        .iter()
        .map(|v| calibration_objective(v, spot, rate, market_data, feller_penalty))
        .collect();

    let mut n_evals = dim + 1;

    for _iter in 0..max_iterations {
        // Sort by function value
        let mut indices: Vec<usize> = (0..=dim).collect();
        indices.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());

        let best_idx = indices[0];
        let worst_idx = indices[dim];
        let second_worst_idx = indices[dim - 1];

        // Check convergence
        let range = values[worst_idx] - values[best_idx];
        if range < 1e-12 {
            break;
        }

        // Centroid (excluding worst)
        let mut centroid = vec![0.0; dim];
        for &idx in &indices[..dim] {
            for j in 0..dim {
                centroid[j] += simplex[idx][j];
            }
        }
        for j in 0..dim {
            centroid[j] /= dim as f64;
        }

        // Reflection
        let mut reflected: Vec<f64> = (0..dim)
            .map(|j| centroid[j] + alpha * (centroid[j] - simplex[worst_idx][j]))
            .collect();
        clamp(&mut reflected);
        let f_reflected = calibration_objective(&reflected, spot, rate, market_data, feller_penalty);
        n_evals += 1;

        if f_reflected < values[best_idx] {
            // Expansion
            let mut expanded: Vec<f64> = (0..dim)
                .map(|j| centroid[j] + gamma * (reflected[j] - centroid[j]))
                .collect();
            clamp(&mut expanded);
            let f_expanded = calibration_objective(&expanded, spot, rate, market_data, feller_penalty);
            n_evals += 1;

            if f_expanded < f_reflected {
                simplex[worst_idx] = expanded;
                values[worst_idx] = f_expanded;
            } else {
                simplex[worst_idx] = reflected;
                values[worst_idx] = f_reflected;
            }
        } else if f_reflected < values[second_worst_idx] {
            simplex[worst_idx] = reflected;
            values[worst_idx] = f_reflected;
        } else {
            // Contraction
            let mut contracted: Vec<f64> = (0..dim)
                .map(|j| centroid[j] + rho_nm * (simplex[worst_idx][j] - centroid[j]))
                .collect();
            clamp(&mut contracted);
            let f_contracted =
                calibration_objective(&contracted, spot, rate, market_data, feller_penalty);
            n_evals += 1;

            if f_contracted < values[worst_idx] {
                simplex[worst_idx] = contracted;
                values[worst_idx] = f_contracted;
            } else {
                // Shrink
                for &idx in &indices[1..] {
                    for j in 0..dim {
                        simplex[idx][j] =
                            simplex[best_idx][j] + sigma_nm * (simplex[idx][j] - simplex[best_idx][j]);
                    }
                    clamp(&mut simplex[idx]);
                    values[idx] =
                        calibration_objective(&simplex[idx], spot, rate, market_data, feller_penalty);
                    n_evals += 1;
                }
            }
        }
    }

    let best_idx = values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let best = &simplex[best_idx];
    let params = HestonParams::new(best[0], best[1], best[2], best[3], best[4], rate);

    CalibrationResult {
        feller_satisfied: params.feller_condition(),
        feller_ratio: params.feller_ratio(),
        params,
        objective: values[best_idx],
        n_iterations: max_iterations,
        n_evals,
    }
}

/// Generate synthetic calibration data from known Heston parameters.
pub fn generate_synthetic_data(
    spot: f64,
    params: &HestonParams,
    noise_level: f64,
) -> Vec<CalibrationPoint> {
    let moneyness = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15];
    let expiries = [1.0 / 12.0, 3.0 / 12.0, 6.0 / 12.0, 1.0];

    let mut rng = rand::thread_rng();
    let mut data = Vec::new();

    for &t in &expiries {
        for &m in &moneyness {
            let k = spot * m;
            let price = heston_call_price(spot, k, t, params);
            if let Some(iv) = implied_volatility(price, spot, k, t, params.r, true) {
                let noisy_iv = if noise_level > 0.0 {
                    (iv + rng.gen_range(-noise_level..noise_level)).max(0.01)
                } else {
                    iv
                };

                // Vega-like weighting: higher near ATM
                let weight = (-0.5 * ((m - 1.0) / 0.3).powi(2)).exp() * t.sqrt();

                data.push(CalibrationPoint {
                    strike: k,
                    expiry: t,
                    market_iv: noisy_iv,
                    weight,
                });
            }
        }
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_synthetic() {
        let true_params = HestonParams::equity_default();
        let spot = 100.0;

        // Generate data with small noise
        let data = generate_synthetic_data(spot, &true_params, 0.005);
        assert!(!data.is_empty(), "Should generate calibration data");

        // Calibrate using DE
        let result = calibrate_differential_evolution(
            spot,
            true_params.r,
            &data,
            100, // Reduced for test speed
            20,
            100.0,
        );

        println!("{}", result);

        // Should roughly recover parameters (with noise, won't be exact)
        assert!(result.objective < 0.01, "Objective should be small");
    }

    #[test]
    fn test_nelder_mead() {
        let true_params = HestonParams::equity_default();
        let spot = 100.0;
        let data = generate_synthetic_data(spot, &true_params, 0.0);

        let initial = [2.5, 0.05, 0.35, -0.6, 0.05];
        let result = calibrate_nelder_mead(spot, true_params.r, &data, &initial, 2000, 100.0);

        println!("{}", result);
        assert!(result.objective < 0.001, "Should calibrate well to clean data");
    }
}
