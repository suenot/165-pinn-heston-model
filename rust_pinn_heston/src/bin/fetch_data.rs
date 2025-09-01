//! Fetch market data from Bybit exchange.
//!
//! Retrieves:
//! - Option chain data (tickers, IVs, Greeks)
//! - Spot price
//! - OHLCV kline data
//! - Realized volatility estimates

use clap::Parser;
use pinn_heston::data::{
    fetch_bybit_klines, fetch_bybit_options, fetch_bybit_spot,
    realized_variance_cc, realized_variance_gk,
};
use pinn_heston::OptionType;

#[derive(Parser, Debug)]
#[command(name = "fetch_data", about = "Fetch market data from Bybit")]
struct Args {
    /// Base coin symbol (BTC, ETH)
    #[arg(long, default_value = "BTC")]
    symbol: String,

    /// Exchange (currently only bybit)
    #[arg(long, default_value = "bybit")]
    exchange: String,

    /// Fetch option chain
    #[arg(long)]
    options: bool,

    /// Fetch kline data
    #[arg(long)]
    klines: bool,

    /// Kline interval
    #[arg(long, default_value = "D")]
    interval: String,

    /// Number of klines
    #[arg(long, default_value_t = 100)]
    limit: usize,

    /// Show all data (options + klines + spot)
    #[arg(long)]
    all: bool,

    /// Output format (text, json)
    #[arg(long, default_value = "text")]
    format: String,
}

fn main() {
    let args = Args::parse();
    let show_all = args.all || (!args.options && !args.klines);

    println!("=== Bybit Market Data: {} ===", args.symbol);
    println!();

    // Spot price
    match fetch_bybit_spot(&args.symbol) {
        Ok(price) => {
            println!("Spot Price: ${:.2}", price);
        }
        Err(e) => {
            eprintln!("Failed to fetch spot: {}", e);
            println!("Using fallback spot price.");
        }
    }
    println!();

    // Option chain
    if show_all || args.options {
        println!("--- Option Chain ---");
        match fetch_bybit_options(&args.symbol) {
            Ok(options) => {
                if options.is_empty() {
                    println!("No options available.");
                } else {
                    let calls: Vec<_> = options.iter().filter(|o| o.option_type == OptionType::Call).collect();
                    let puts: Vec<_> = options.iter().filter(|o| o.option_type == OptionType::Put).collect();

                    println!("Total options: {} ({} calls, {} puts)",
                             options.len(), calls.len(), puts.len());

                    // Show top options by volume
                    let mut sorted = options.clone();
                    sorted.sort_by(|a, b| b.volume.partial_cmp(&a.volume).unwrap());

                    println!("\nTop options by volume:");
                    println!("  {:>30} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}",
                             "Symbol", "Strike", "Bid", "Ask", "Mark", "IV", "Volume");
                    println!("  {}", "-".repeat(85));

                    for opt in sorted.iter().take(20) {
                        println!("  {:>30} {:>8.0} {:>8.2} {:>8.2} {:>8.2} {:>8.4} {:>10.0}",
                                 opt.symbol, opt.strike, opt.bid, opt.ask,
                                 opt.mark_price, opt.implied_vol, opt.volume);
                    }

                    // IV statistics
                    let call_ivs: Vec<f64> = calls.iter()
                        .filter(|o| o.implied_vol > 0.0)
                        .map(|o| o.implied_vol)
                        .collect();

                    if !call_ivs.is_empty() {
                        let mean_iv: f64 = call_ivs.iter().sum::<f64>() / call_ivs.len() as f64;
                        let min_iv = call_ivs.iter().cloned().fold(f64::INFINITY, f64::min);
                        let max_iv = call_ivs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                        println!("\nCall IV Statistics:");
                        println!("  Mean: {:.4} ({:.1}%)", mean_iv, mean_iv * 100.0);
                        println!("  Min:  {:.4} ({:.1}%)", min_iv, min_iv * 100.0);
                        println!("  Max:  {:.4} ({:.1}%)", max_iv, max_iv * 100.0);
                    }

                    if args.format == "json" {
                        match serde_json::to_string_pretty(&options) {
                            Ok(json) => println!("\n{}", json),
                            Err(e) => eprintln!("JSON serialization error: {}", e),
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to fetch options: {}", e);
            }
        }
        println!();
    }

    // Kline data
    if show_all || args.klines {
        println!("--- OHLCV Kline Data ({} interval, {} candles) ---",
                 args.interval, args.limit);

        match fetch_bybit_klines(&args.symbol, &args.interval, args.limit) {
            Ok(candles) => {
                if candles.is_empty() {
                    println!("No kline data available.");
                } else {
                    println!("Fetched {} candles", candles.len());

                    // Show last 5 candles
                    println!("\nRecent candles:");
                    println!("  {:>15} {:>10} {:>10} {:>10} {:>10} {:>12}",
                             "Timestamp", "Open", "High", "Low", "Close", "Volume");
                    println!("  {}", "-".repeat(72));

                    for c in candles.iter().rev().take(5).rev() {
                        println!("  {:>15} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>12.2}",
                                 c.timestamp_ms, c.open, c.high, c.low, c.close, c.volume);
                    }

                    // Realized volatility estimates
                    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
                    let rv_cc = realized_variance_cc(&closes, true);
                    let rv_gk = realized_variance_gk(&candles, true);

                    println!("\nRealized Volatility Estimates:");
                    println!("  Close-to-Close: {:.4} (vol={:.1}%)", rv_cc, rv_cc.sqrt() * 100.0);
                    println!("  Garman-Klass:   {:.4} (vol={:.1}%)", rv_gk, rv_gk.sqrt() * 100.0);

                    // Price statistics
                    let min_price = closes.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_price = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let last_price = *closes.last().unwrap();

                    println!("\nPrice Statistics:");
                    println!("  Current: {:.2}", last_price);
                    println!("  Range:   {:.2} - {:.2}", min_price, max_price);
                    println!("  Change:  {:.2}%",
                             (last_price / closes[0] - 1.0) * 100.0);
                }
            }
            Err(e) => {
                eprintln!("Failed to fetch klines: {}", e);
            }
        }
    }
}
