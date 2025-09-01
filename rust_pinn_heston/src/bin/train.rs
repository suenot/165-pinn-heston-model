//! Training binary for the Heston PINN.
//!
//! Trains the neural network by minimizing the composite loss:
//!     L = lambda_pde * L_pde + lambda_bc * L_bc + lambda_ic * L_ic
//!
//! Uses finite-difference-based gradient estimation (SPSA) since we don't
//! have autograd in pure Rust. For production training, use the Python implementation
//! with PyTorch autograd.

use clap::Parser;
use pinn_heston::network::{generate_collocation_points, HestonPINN, TrainConfig};
use pinn_heston::analytical::heston_call_price;
use pinn_heston::HestonParams;
use rand::Rng;

#[derive(Parser, Debug)]
#[command(name = "train", about = "Train the Heston PINN model")]
struct Args {
    /// Number of training epochs
    #[arg(long, default_value_t = 1000)]
    epochs: usize,

    /// Hidden layer dimension
    #[arg(long, default_value_t = 64)]
    hidden_dim: usize,

    /// Number of hidden layers
    #[arg(long, default_value_t = 4)]
    num_layers: usize,

    /// Learning rate
    #[arg(long, default_value_t = 0.001)]
    lr: f64,

    /// Number of collocation points
    #[arg(long, default_value_t = 512)]
    n_collocation: usize,

    /// Spot price
    #[arg(long, default_value_t = 100.0)]
    spot: f64,

    /// Strike price
    #[arg(long, default_value_t = 100.0)]
    strike: f64,

    /// Time to expiry
    #[arg(long, default_value_t = 1.0)]
    expiry: f64,

    /// Output model path
    #[arg(long, default_value = "model.json")]
    output: String,

    /// Use crypto parameters
    #[arg(long)]
    crypto: bool,
}

/// Compute terminal condition loss: V(S, v, T) should equal payoff.
fn terminal_condition_loss(model: &HestonPINN, strike: f64, n_points: usize) -> f64 {
    let mut rng = rand::thread_rng();
    let mut total = 0.0;

    for _ in 0..n_points {
        let s = rng.gen_range(1.0..model.s_max);
        let v = rng.gen_range(0.001..model.v_max);
        let t = model.t_max; // At expiry

        let predicted = model.forward(s, v, t);
        let payoff = (s - strike).max(0.0); // Call payoff

        total += (predicted - payoff).powi(2);
    }

    total / n_points as f64
}

/// Compute boundary loss at S=0 (call should be 0).
fn boundary_loss_s0(model: &HestonPINN, n_points: usize) -> f64 {
    let mut rng = rand::thread_rng();
    let mut total = 0.0;

    for _ in 0..n_points {
        let v = rng.gen_range(0.001..model.v_max);
        let t = rng.gen_range(0.0..model.t_max);

        let predicted = model.forward(1e-6, v, t);
        total += predicted.powi(2); // Call at S=0 should be 0
    }

    total / n_points as f64
}

/// Compute PDE residual loss at collocation points.
fn pde_residual_loss(
    model: &HestonPINN,
    points: &[(f64, f64, f64)],
    params: &HestonParams,
) -> f64 {
    let mut total = 0.0;

    for &(s, v, t) in points {
        let residual = model.pde_residual(
            s, v, t,
            params.kappa, params.theta, params.sigma, params.rho, params.r,
        );
        total += residual.powi(2);
    }

    total / points.len() as f64
}

/// Compute total loss.
fn total_loss(
    model: &HestonPINN,
    points: &[(f64, f64, f64)],
    params: &HestonParams,
    strike: f64,
    config: &TrainConfig,
) -> f64 {
    let l_pde = pde_residual_loss(model, points, params);
    let l_ic = terminal_condition_loss(model, strike, config.n_boundary);
    let l_bc = boundary_loss_s0(model, config.n_boundary);

    config.lambda_pde * l_pde + config.lambda_ic * l_ic + config.lambda_bc * l_bc
}

/// Validate against analytical Heston prices.
fn validate(model: &HestonPINN, strike: f64, params: &HestonParams) -> (f64, f64) {
    let spots = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
    let v_vals = [0.02, 0.04, 0.08];

    let mut errors = Vec::new();

    for &v0 in &v_vals {
        let mut p = params.clone();
        p.v0 = v0;
        for &s in &spots {
            let analytical = heston_call_price(s, strike, params.r, &p);
            let pinn = model.forward(s, v0, 0.0);
            errors.push((analytical - pinn).abs());
        }
    }

    let mae = errors.iter().sum::<f64>() / errors.len() as f64;
    let max_err = errors.iter().cloned().fold(0.0_f64, f64::max);
    (mae, max_err)
}

fn main() {
    let args = Args::parse();

    let params = if args.crypto {
        HestonParams::crypto_default()
    } else {
        HestonParams::equity_default()
    };

    println!("=== Heston PINN Training ===");
    println!("Parameters: kappa={}, theta={}, sigma={}, rho={}, v0={}",
             params.kappa, params.theta, params.sigma, params.rho, params.v0);
    println!("Feller condition: {} (ratio={:.2})",
             params.feller_condition(), params.feller_ratio());
    println!("Architecture: {} hidden x {} layers", args.hidden_dim, args.num_layers);

    let s_max = args.spot * 2.0;
    let v_max = (params.theta * 5.0).max(1.0);
    let t_max = args.expiry;

    let mut model = HestonPINN::new(args.hidden_dim, args.num_layers, s_max, v_max, t_max);
    println!("Model parameters: {}", model.num_parameters());

    let config = TrainConfig {
        epochs: args.epochs,
        learning_rate: args.lr,
        n_collocation: args.n_collocation,
        n_boundary: 64,
        lambda_pde: 1.0,
        lambda_bc: 10.0,
        lambda_ic: 10.0,
        log_every: args.epochs / 20,
    };

    println!("\nTraining with SPSA gradient estimation...");
    println!("{:>7} | {:>10} | {:>10} | {:>10} | {:>10}",
             "Epoch", "Loss", "PDE", "Val MAE", "Val Max");
    println!("{}", "-".repeat(65));

    let perturbation = 0.01;
    let mut best_val_mae = f64::INFINITY;

    for epoch in 0..config.epochs {
        // Generate fresh collocation points
        let points = generate_collocation_points(
            s_max, v_max, t_max, args.strike, config.n_collocation,
        );

        // SPSA gradient estimation and update
        // Simultaneously Perturbation Stochastic Approximation
        let mut rng = rand::thread_rng();
        let c_k = perturbation / (1.0 + epoch as f64).powf(0.101);
        let a_k = args.lr / (1.0 + epoch as f64).powf(0.602);

        // For each layer, perturb all weights simultaneously
        for layer_idx in 0..model.layers.len() {
            let (rows, cols) = model.layers[layer_idx].weights.dim();

            // Random direction
            let delta_w: Vec<Vec<f64>> = (0..rows)
                .map(|_| {
                    (0..cols)
                        .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
                        .collect()
                })
                .collect();

            let delta_b: Vec<f64> = (0..model.layers[layer_idx].biases.len())
                .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
                .collect();

            // Perturb positive
            for i in 0..rows {
                for j in 0..cols {
                    model.layers[layer_idx].weights[[i, j]] += c_k * delta_w[i][j];
                }
            }
            for (i, d) in delta_b.iter().enumerate() {
                model.layers[layer_idx].biases[i] += c_k * d;
            }
            let loss_plus = total_loss(&model, &points, &params, args.strike, &config);

            // Perturb negative (2*c_k from positive)
            for i in 0..rows {
                for j in 0..cols {
                    model.layers[layer_idx].weights[[i, j]] -= 2.0 * c_k * delta_w[i][j];
                }
            }
            for (i, d) in delta_b.iter().enumerate() {
                model.layers[layer_idx].biases[i] -= 2.0 * c_k * d;
            }
            let loss_minus = total_loss(&model, &points, &params, args.strike, &config);

            // Restore and apply gradient
            let grad_est = (loss_plus - loss_minus) / (2.0 * c_k);

            for i in 0..rows {
                for j in 0..cols {
                    model.layers[layer_idx].weights[[i, j]] +=
                        c_k * delta_w[i][j] - a_k * grad_est * delta_w[i][j];
                }
            }
            for (i, d) in delta_b.iter().enumerate() {
                model.layers[layer_idx].biases[i] += c_k * d - a_k * grad_est * d;
            }
        }

        // Logging
        if config.log_every > 0 && epoch % config.log_every == 0 {
            let points_val = generate_collocation_points(
                s_max, v_max, t_max, args.strike, 128,
            );
            let current_loss = total_loss(&model, &points_val, &params, args.strike, &config);
            let pde_loss = pde_residual_loss(&model, &points_val, &params);
            let (val_mae, val_max) = validate(&model, args.strike, &params);

            println!("{:7} | {:10.6} | {:10.6} | {:10.6} | {:10.6}",
                     epoch, current_loss, pde_loss, val_mae, val_max);

            if val_mae < best_val_mae {
                best_val_mae = val_mae;
                if let Err(e) = model.save(&args.output) {
                    eprintln!("Failed to save model: {}", e);
                }
            }
        }
    }

    println!("\nTraining complete. Best validation MAE: {:.6}", best_val_mae);
    println!("Model saved to: {}", args.output);

    // Final validation
    println!("\nFinal price comparison:");
    let spots = [80.0, 90.0, 100.0, 110.0, 120.0];
    for &s in &spots {
        let pinn = model.forward(s, params.v0, 0.0);
        let analytical = heston_call_price(s, args.strike, args.expiry, &params);
        println!("  S={:.0}: PINN={:.4}, Analytical={:.4}, Error={:.4}",
                 s, pinn, analytical, (pinn - analytical).abs());
    }
}
