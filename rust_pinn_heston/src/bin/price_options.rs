//! Price European options using the Heston model.
//!
//! Supports both semi-analytical (characteristic function) and PINN-based pricing.
//! Outputs prices, implied volatilities, and Greeks.

use clap::Parser;
use pinn_heston::analytical::{heston_call_price, heston_put_price, heston_iv_smile, implied_volatility};
use pinn_heston::greeks::analytical_greeks;
use pinn_heston::network::HestonPINN;
use pinn_heston::HestonParams;

#[derive(Parser, Debug)]
#[command(name = "price_options", about = "Price options using Heston model")]
struct Args {
    /// Spot price
    #[arg(long, default_value_t = 100.0)]
    spot: f64,

    /// Strike price
    #[arg(long, default_value_t = 100.0)]
    strike: f64,

    /// Time to expiry (years)
    #[arg(long, default_value_t = 0.5)]
    expiry: f64,

    /// Use crypto parameters
    #[arg(long)]
    crypto: bool,

    /// PINN model path (if available)
    #[arg(long)]
    model_path: Option<String>,

    /// Show smile
    #[arg(long)]
    smile: bool,

    /// Show Greeks
    #[arg(long)]
    greeks: bool,
}

fn main() {
    let args = Args::parse();

    let params = if args.crypto {
        HestonParams::crypto_default()
    } else {
        HestonParams::equity_default()
    };

    let asset_type = if args.crypto { "Crypto (BTC)" } else { "Equity" };

    println!("===================================================");
    println!("  Heston Option Pricing - {}", asset_type);
    println!("===================================================");
    println!("  Spot:    {:.2}", args.spot);
    println!("  Strike:  {:.2}", args.strike);
    println!("  Expiry:  {:.4} years ({:.0} days)", args.expiry, args.expiry * 365.0);
    println!("  Rate:    {:.2}%", params.r * 100.0);
    println!();
    println!("  Heston Parameters:");
    println!("    kappa = {:.4} (mean-reversion speed)", params.kappa);
    println!("    theta = {:.4} (long-run variance, vol={:.1}%)",
             params.theta, params.theta.sqrt() * 100.0);
    println!("    sigma = {:.4} (vol-of-vol)", params.sigma);
    println!("    rho   = {:.4} (correlation)", params.rho);
    println!("    v0    = {:.4} (current variance, vol={:.1}%)",
             params.v0, params.v0.sqrt() * 100.0);
    println!("    Feller: {} (ratio={:.2})",
             params.feller_condition(), params.feller_ratio());
    println!();

    // Analytical pricing
    let call_price = heston_call_price(args.spot, args.strike, args.expiry, &params);
    let put_price = heston_put_price(args.spot, args.strike, args.expiry, &params);

    let call_iv = implied_volatility(
        call_price, args.spot, args.strike, args.expiry, params.r, true
    ).unwrap_or(0.0);

    println!("--- Analytical Heston Prices ---");
    println!("  Call: {:.4} (IV: {:.2}%)", call_price, call_iv * 100.0);
    println!("  Put:  {:.4}", put_price);

    // Put-call parity check
    let parity = call_price - put_price - args.spot + args.strike * (-params.r * args.expiry).exp();
    println!("  Put-Call Parity Error: {:.2e}", parity.abs());

    // PINN pricing (if model available)
    if let Some(model_path) = &args.model_path {
        match HestonPINN::load(model_path) {
            Ok(model) => {
                let pinn_price = model.forward(args.spot, params.v0, 0.0);
                println!("\n--- PINN Price ---");
                println!("  Call: {:.4}", pinn_price);
                println!("  Error vs Analytical: {:.2e}", (pinn_price - call_price).abs());
            }
            Err(e) => {
                eprintln!("Failed to load PINN model: {}", e);
            }
        }
    }

    // Greeks
    if args.greeks {
        println!("\n--- Greeks (Finite Differences on Analytical) ---");
        let greeks = analytical_greeks(
            args.spot, args.strike, args.expiry, &params, None, None, None,
        );
        print!("{}", greeks);
    }

    // Implied volatility smile
    if args.smile {
        println!("\n--- Implied Volatility Smile ---");
        let moneyness_levels: Vec<f64> = vec![0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20];
        let strikes: Vec<f64> = moneyness_levels.iter().map(|m| args.spot * m).collect();
        let ivs = heston_iv_smile(args.spot, &strikes, args.expiry, &params);

        println!("  {:>10} {:>10} {:>10} {:>10}",
                 "Moneyness", "Strike", "Price", "IV (%)");
        println!("  {}", "-".repeat(45));

        for i in 0..strikes.len() {
            let price = heston_call_price(args.spot, strikes[i], args.expiry, &params);
            println!("  {:>10.2} {:>10.2} {:>10.4} {:>10.2}",
                     moneyness_levels[i], strikes[i], price, ivs[i] * 100.0);
        }

        // Term structure
        println!("\n--- Smile Term Structure ---");
        let expiries = [0.05, 0.10, 0.25, 0.50, 1.0];
        let atm_strike = args.spot;

        println!("  {:>10} {:>10} {:>10} {:>10}",
                 "Expiry(d)", "Price", "IV(%)", "Skew(%)");
        println!("  {}", "-".repeat(45));

        for &t in &expiries {
            let atm_price = heston_call_price(args.spot, atm_strike, t, &params);
            let atm_iv = implied_volatility(atm_price, args.spot, atm_strike, t, params.r, true)
                .unwrap_or(0.0);

            let otm_strike = args.spot * 0.90;
            let otm_price = heston_call_price(args.spot, otm_strike, t, &params);
            let otm_iv = implied_volatility(otm_price, args.spot, otm_strike, t, params.r, true)
                .unwrap_or(0.0);

            let skew = (otm_iv - atm_iv) * 100.0;

            println!("  {:>10.0} {:>10.4} {:>10.2} {:>10.2}",
                     t * 365.0, atm_price, atm_iv * 100.0, skew);
        }
    }
}
