//! Calibrate Heston model parameters to market data.
//!
//! Supports:
//! - Calibration to synthetic data (for testing)
//! - Calibration to Bybit crypto option IVs
//!
//! Uses differential evolution for global search followed by
//! Nelder-Mead for local refinement.

use clap::Parser;
use pinn_heston::calibration::{
    calibrate_differential_evolution, calibrate_nelder_mead,
    generate_synthetic_data, CalibrationPoint,
};
use pinn_heston::data::fetch_bybit_options;
use pinn_heston::HestonParams;
use pinn_heston::OptionType;

#[derive(Parser, Debug)]
#[command(name = "calibrate", about = "Calibrate Heston model to market data")]
struct Args {
    /// Data source: 'synthetic' or 'bybit'
    #[arg(long, default_value = "synthetic")]
    source: String,

    /// Symbol for Bybit data
    #[arg(long, default_value = "BTC")]
    symbol: String,

    /// Spot price (for synthetic data)
    #[arg(long, default_value_t = 100.0)]
    spot: f64,

    /// Risk-free rate
    #[arg(long, default_value_t = 0.05)]
    rate: f64,

    /// Maximum DE iterations
    #[arg(long, default_value_t = 200)]
    max_iter: usize,

    /// DE population size
    #[arg(long, default_value_t = 30)]
    population: usize,

    /// Noise level for synthetic data
    #[arg(long, default_value_t = 0.01)]
    noise: f64,

    /// Feller penalty strength
    #[arg(long, default_value_t = 100.0)]
    feller_penalty: f64,

    /// Output results as JSON
    #[arg(long)]
    json: bool,
}

fn main() {
    let args = Args::parse();

    println!("=== Heston Model Calibration ===");
    println!("Source: {}", args.source);

    let (spot, market_data) = match args.source.as_str() {
        "bybit" => {
            println!("Fetching Bybit {} options...", args.symbol);

            let options = match fetch_bybit_options(&args.symbol) {
                Ok(opts) => opts,
                Err(e) => {
                    eprintln!("Failed to fetch Bybit data: {}", e);
                    eprintln!("Falling back to synthetic data.");
                    let true_params = if args.symbol == "BTC" {
                        HestonParams::crypto_default()
                    } else {
                        HestonParams::equity_default()
                    };
                    let spot = if args.symbol == "BTC" { 50000.0 } else { 100.0 };
                    let data = generate_synthetic_data(spot, &true_params, args.noise);
                    println!("Generated {} synthetic points", data.len());
                    return run_calibration(spot, &data, &args);
                }
            };

            // Get spot price
            let spot_price = match pinn_heston::data::fetch_bybit_spot(&args.symbol) {
                Ok(p) => p,
                Err(_) => if args.symbol == "BTC" { 50000.0 } else { 3000.0 },
            };

            // Convert to CalibrationPoints (use calls with positive IV)
            let cal_data: Vec<CalibrationPoint> = options
                .iter()
                .filter(|o| o.option_type == OptionType::Call && o.implied_vol > 0.01)
                .map(|o| {
                    let moneyness = o.strike / spot_price;
                    let weight = (-0.5 * ((moneyness - 1.0) / 0.3).powi(2)).exp()
                        * o.expiry.sqrt();
                    CalibrationPoint {
                        strike: o.strike,
                        expiry: o.expiry,
                        market_iv: o.implied_vol,
                        weight: weight.max(0.01),
                    }
                })
                .collect();

            if cal_data.is_empty() {
                eprintln!("No valid calibration data from Bybit. Using synthetic.");
                let true_params = HestonParams::crypto_default();
                let data = generate_synthetic_data(spot_price, &true_params, args.noise);
                (spot_price, data)
            } else {
                println!("Using {} market options for calibration", cal_data.len());
                (spot_price, cal_data)
            }
        }
        _ => {
            // Synthetic data
            let true_params = HestonParams::equity_default();
            println!("True parameters:");
            println!("  kappa={:.4}, theta={:.4}, sigma={:.4}, rho={:.4}, v0={:.4}",
                     true_params.kappa, true_params.theta, true_params.sigma,
                     true_params.rho, true_params.v0);

            let data = generate_synthetic_data(args.spot, &true_params, args.noise);
            println!("Generated {} calibration points (noise={:.3})", data.len(), args.noise);

            (args.spot, data)
        }
    };

    run_calibration(spot, &market_data, &args);
}

fn run_calibration(spot: f64, market_data: &[CalibrationPoint], args: &Args) {
    println!("\n--- Phase 1: Differential Evolution (Global) ---");
    let de_result = calibrate_differential_evolution(
        spot,
        args.rate,
        market_data,
        args.max_iter,
        args.population,
        args.feller_penalty,
    );
    println!("{}", de_result);

    // Refine with Nelder-Mead
    println!("--- Phase 2: Nelder-Mead (Local Refinement) ---");
    let initial = [
        de_result.params.kappa,
        de_result.params.theta,
        de_result.params.sigma,
        de_result.params.rho,
        de_result.params.v0,
    ];

    let nm_result = calibrate_nelder_mead(
        spot,
        args.rate,
        market_data,
        &initial,
        5000,
        args.feller_penalty,
    );
    println!("{}", nm_result);

    // Compare if synthetic
    if args.source == "synthetic" {
        let true_params = HestonParams::equity_default();
        println!("--- Parameter Recovery ---");
        println!("  {:>8} {:>10} {:>10} {:>10}",
                 "Param", "True", "Calibrated", "Error");
        println!("  {}", "-".repeat(42));

        let pairs = [
            ("kappa", true_params.kappa, nm_result.params.kappa),
            ("theta", true_params.theta, nm_result.params.theta),
            ("sigma", true_params.sigma, nm_result.params.sigma),
            ("rho", true_params.rho, nm_result.params.rho),
            ("v0", true_params.v0, nm_result.params.v0),
        ];

        for (name, true_val, cal_val) in &pairs {
            println!("  {:>8} {:>10.4} {:>10.4} {:>10.4}",
                     name, true_val, cal_val, (cal_val - true_val).abs());
        }
    }

    // JSON output
    if args.json {
        match serde_json::to_string_pretty(&nm_result) {
            Ok(json) => println!("\n{}", json),
            Err(e) => eprintln!("JSON error: {}", e),
        }
    }

    // Suggested usage
    println!("\n--- Calibrated Parameters (copy-paste) ---");
    println!("kappa={:.6}", nm_result.params.kappa);
    println!("theta={:.6}", nm_result.params.theta);
    println!("sigma={:.6}", nm_result.params.sigma);
    println!("rho={:.6}", nm_result.params.rho);
    println!("v0={:.6}", nm_result.params.v0);
}
