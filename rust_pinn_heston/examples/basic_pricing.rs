//! Basic example: Price a European call option under the Heston model.
//!
//! Demonstrates:
//! - Setting up Heston parameters for equities and crypto
//! - Computing call/put prices
//! - Generating implied volatility smile
//! - Computing Greeks
//!
//! Run with:
//!     cargo run --example basic_pricing

use pinn_heston::analytical::{heston_call_price, heston_put_price, heston_iv_smile};
use pinn_heston::greeks::analytical_greeks;
use pinn_heston::HestonParams;

fn main() {
    println!("====================================================");
    println!("  Heston Model - Basic Pricing Example");
    println!("====================================================\n");

    // --- Equity Example ---
    println!("--- Equity Parameters (SPX-like) ---");
    let equity = HestonParams::equity_default();
    let s = 100.0;
    let k = 100.0;
    let t = 0.5;

    println!("  S={}, K={}, T={:.2}y", s, k, t);
    println!("  kappa={}, theta={}, sigma={}, rho={}, v0={}",
             equity.kappa, equity.theta, equity.sigma, equity.rho, equity.v0);
    println!("  Feller: {} (ratio={:.2})\n",
             equity.feller_condition(), equity.feller_ratio());

    let call = heston_call_price(s, k, t, &equity);
    let put = heston_put_price(s, k, t, &equity);

    println!("  ATM Call Price:  {:.4}", call);
    println!("  ATM Put Price:   {:.4}", put);
    println!("  Put-Call Parity: {:.2e}\n",
             (call - put - s + k * (-equity.r * t).exp()).abs());

    // Implied volatility smile
    let strikes: Vec<f64> = vec![80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];
    let ivs = heston_iv_smile(s, &strikes, t, &equity);

    println!("  Implied Volatility Smile:");
    println!("  {:>8} {:>8} {:>8}", "Strike", "IV(%)", "Skew");
    println!("  {}", "-".repeat(28));
    let atm_iv = ivs[4]; // ATM is at index 4 (strike=100)
    for (strike, iv) in strikes.iter().zip(ivs.iter()) {
        let skew = (iv - atm_iv) * 100.0;
        println!("  {:>8.0} {:>8.2} {:>+8.2}", strike, iv * 100.0, skew);
    }

    // Greeks
    println!("\n  Greeks (ATM):");
    let greeks = analytical_greeks(s, k, t, &equity, None, None, None);
    print!("{}", greeks);

    // --- Crypto Example ---
    println!("\n--- Crypto Parameters (BTC-like) ---");
    let crypto = HestonParams::crypto_default();
    let s_btc = 50000.0;
    let k_btc = 50000.0;
    let t_btc = 0.25; // 3 months

    println!("  S=${}, K=${}, T={:.2}y ({:.0} days)", s_btc, k_btc, t_btc, t_btc * 365.0);
    println!("  kappa={}, theta={} (vol={:.0}%), sigma={}, rho={}, v0={} (vol={:.0}%)",
             crypto.kappa, crypto.theta, crypto.theta.sqrt() * 100.0,
             crypto.sigma, crypto.rho, crypto.v0, crypto.v0.sqrt() * 100.0);
    println!("  Feller: {} (ratio={:.2})\n",
             crypto.feller_condition(), crypto.feller_ratio());

    let call_btc = heston_call_price(s_btc, k_btc, t_btc, &crypto);
    let put_btc = heston_put_price(s_btc, k_btc, t_btc, &crypto);

    println!("  ATM Call: ${:.2}", call_btc);
    println!("  ATM Put:  ${:.2}", put_btc);

    let btc_strikes: Vec<f64> = vec![
        40000.0, 42000.0, 44000.0, 46000.0, 48000.0,
        50000.0, 52000.0, 54000.0, 56000.0, 58000.0, 60000.0,
    ];
    let btc_ivs = heston_iv_smile(s_btc, &btc_strikes, t_btc, &crypto);

    println!("\n  BTC Implied Volatility Smile:");
    println!("  {:>10} {:>8} {:>8}", "Strike", "IV(%)", "Moneyness");
    println!("  {}", "-".repeat(30));
    for (strike, iv) in btc_strikes.iter().zip(btc_ivs.iter()) {
        println!("  {:>10.0} {:>8.2} {:>8.2}", strike, iv * 100.0, strike / s_btc);
    }

    // BTC Greeks
    println!("\n  BTC Greeks (ATM):");
    let btc_greeks = analytical_greeks(s_btc, k_btc, t_btc, &crypto, None, None, None);
    print!("{}", btc_greeks);

    println!("\n====================================================");
    println!("  Note: BTC options show much higher vol levels and");
    println!("  weaker skew compared to equity options.");
    println!("====================================================");
}
