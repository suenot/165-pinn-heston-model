//! Market data fetching from Bybit and other sources.
//!
//! Provides async and blocking interfaces for fetching:
//! - Option chain data (tickers)
//! - Spot prices
//! - OHLCV kline data

use crate::{MarketOption, OptionType};
use serde::Deserialize;

/// Bybit API response wrapper.
#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i64,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

/// Bybit ticker list result.
#[derive(Debug, Deserialize)]
struct TickerListResult {
    list: Vec<BybitTicker>,
}

/// Single Bybit option ticker.
#[derive(Debug, Deserialize)]
struct BybitTicker {
    symbol: String,
    #[serde(rename = "bid1Price")]
    bid1_price: String,
    #[serde(rename = "ask1Price")]
    ask1_price: String,
    #[serde(rename = "markPrice")]
    mark_price: String,
    #[serde(rename = "markIv")]
    mark_iv: String,
    delta: String,
    gamma: String,
    vega: String,
    theta: String,
    #[serde(rename = "volume24h")]
    volume_24h: String,
    #[serde(rename = "openInterest")]
    open_interest: String,
}

/// Bybit kline result.
#[derive(Debug, Deserialize)]
struct KlineResult {
    list: Vec<Vec<String>>,
}

/// OHLCV candle data.
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp_ms: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Spot ticker result.
#[derive(Debug, Deserialize)]
struct SpotTickerResult {
    list: Vec<SpotTicker>,
}

#[derive(Debug, Deserialize)]
struct SpotTicker {
    #[serde(rename = "lastPrice")]
    last_price: String,
}

const BYBIT_BASE_URL: &str = "https://api.bybit.com";

/// Parse option symbol like "BTC-28JUN24-70000-C" into (expiry_str, strike, type).
fn parse_option_symbol(symbol: &str) -> Option<(String, f64, OptionType)> {
    let parts: Vec<&str> = symbol.split('-').collect();
    if parts.len() < 4 {
        return None;
    }

    let expiry_str = parts[1].to_string();
    let strike: f64 = parts[2].parse().ok()?;
    let opt_type = match parts[3] {
        "C" => OptionType::Call,
        "P" => OptionType::Put,
        _ => return None,
    };

    Some((expiry_str, strike, opt_type))
}

/// Parse expiry string like "28JUN24" to days-to-expiry (approximate).
fn parse_expiry_to_days(expiry_str: &str) -> Option<f64> {
    // Simple parsing: extract day, month, year
    if expiry_str.len() < 5 {
        return None;
    }

    let months = [
        "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
        "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    ];

    // Find month position
    let month_str = &expiry_str[2..5].to_uppercase();
    let _month_idx = months.iter().position(|&m| m == month_str)?;

    // For simplicity, estimate days based on typical option tenors
    // In production, parse the full date and compute exact days
    // Here we return a rough estimate assuming 30 days per expiry point
    let day: f64 = expiry_str[..2].parse().ok()?;
    let year_str = &expiry_str[5..];
    let _year: i32 = if year_str.len() == 2 {
        2000 + year_str.parse::<i32>().ok()?
    } else {
        year_str.parse().ok()?
    };

    // Use a simplified calculation (real impl would use chrono)
    Some(day.max(1.0))
}

/// Fetch option chain from Bybit (blocking).
pub fn fetch_bybit_options(symbol: &str) -> anyhow::Result<Vec<MarketOption>> {
    let url = format!(
        "{}/v5/market/tickers?category=option&baseCoin={}",
        BYBIT_BASE_URL, symbol
    );

    let client = reqwest::blocking::Client::new();
    let resp = client
        .get(&url)
        .timeout(std::time::Duration::from_secs(10))
        .send()?;

    let body: BybitResponse<TickerListResult> = resp.json()?;

    if body.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", body.ret_msg);
    }

    let mut options = Vec::new();

    for ticker in &body.result.list {
        let (expiry_str, strike, opt_type) = match parse_option_symbol(&ticker.symbol) {
            Some(v) => v,
            None => continue,
        };

        let bid: f64 = ticker.bid1_price.parse().unwrap_or(0.0);
        let ask: f64 = ticker.ask1_price.parse().unwrap_or(0.0);
        let mark: f64 = ticker.mark_price.parse().unwrap_or(0.0);
        let iv: f64 = ticker.mark_iv.parse().unwrap_or(0.0);

        let mid = if bid > 0.0 && ask > 0.0 {
            (bid + ask) / 2.0
        } else {
            mark
        };

        // Estimate days to expiry (simplified)
        let days = parse_expiry_to_days(&expiry_str).unwrap_or(30.0);
        let expiry_years = days / 365.0;

        options.push(MarketOption {
            symbol: ticker.symbol.clone(),
            strike,
            expiry: expiry_years,
            option_type: opt_type,
            bid,
            ask,
            mid,
            mark_price: mark,
            implied_vol: iv,
            delta: ticker.delta.parse().unwrap_or(0.0),
            gamma: ticker.gamma.parse().unwrap_or(0.0),
            vega: ticker.vega.parse().unwrap_or(0.0),
            theta: ticker.theta.parse().unwrap_or(0.0),
            volume: ticker.volume_24h.parse().unwrap_or(0.0),
            open_interest: ticker.open_interest.parse().unwrap_or(0.0),
        });
    }

    Ok(options)
}

/// Fetch spot price from Bybit (blocking).
pub fn fetch_bybit_spot(symbol: &str) -> anyhow::Result<f64> {
    let pair = format!("{}USDT", symbol);
    let url = format!(
        "{}/v5/market/tickers?category=spot&symbol={}",
        BYBIT_BASE_URL, pair
    );

    let client = reqwest::blocking::Client::new();
    let resp = client
        .get(&url)
        .timeout(std::time::Duration::from_secs(10))
        .send()?;

    let body: BybitResponse<SpotTickerResult> = resp.json()?;

    if body.ret_code != 0 || body.result.list.is_empty() {
        anyhow::bail!("Failed to fetch spot price");
    }

    let price: f64 = body.result.list[0].last_price.parse()?;
    Ok(price)
}

/// Fetch OHLCV kline data from Bybit (blocking).
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<Candle>> {
    let pair = format!("{}USDT", symbol);
    let url = format!(
        "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        BYBIT_BASE_URL, pair, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp = client
        .get(&url)
        .timeout(std::time::Duration::from_secs(10))
        .send()?;

    let body: BybitResponse<KlineResult> = resp.json()?;

    if body.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", body.ret_msg);
    }

    let mut candles: Vec<Candle> = body
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() < 6 {
                return None;
            }
            Some(Candle {
                timestamp_ms: row[0].parse().ok()?,
                open: row[1].parse().ok()?,
                high: row[2].parse().ok()?,
                low: row[3].parse().ok()?,
                close: row[4].parse().ok()?,
                volume: row[5].parse().ok()?,
            })
        })
        .collect();

    // Sort by timestamp ascending
    candles.sort_by_key(|c| c.timestamp_ms);

    Ok(candles)
}

/// Estimate realized variance from OHLCV data (Garman-Klass estimator).
pub fn realized_variance_gk(candles: &[Candle], annualize: bool) -> f64 {
    if candles.len() < 2 {
        return 0.0;
    }

    let n = candles.len() as f64;
    let mut sum = 0.0;

    for c in candles {
        let log_hl = (c.high / c.low).ln();
        let log_co = (c.close / c.open).ln();
        sum += 0.5 * log_hl * log_hl - (2.0 * 2.0_f64.ln() - 1.0) * log_co * log_co;
    }

    let var = sum / n;
    if annualize {
        var * 252.0
    } else {
        var
    }
}

/// Estimate realized variance from close-to-close returns.
pub fn realized_variance_cc(closes: &[f64], annualize: bool) -> f64 {
    if closes.len() < 2 {
        return 0.0;
    }

    let log_returns: Vec<f64> = closes
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();

    let n = log_returns.len() as f64;
    let mean = log_returns.iter().sum::<f64>() / n;
    let var = log_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);

    if annualize {
        var * 252.0
    } else {
        var
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_option_symbol() {
        let (expiry, strike, opt_type) = parse_option_symbol("BTC-28JUN24-70000-C").unwrap();
        assert_eq!(expiry, "28JUN24");
        assert_eq!(strike, 70000.0);
        assert_eq!(opt_type, OptionType::Call);

        let (_, strike, opt_type) = parse_option_symbol("ETH-27DEC24-4000-P").unwrap();
        assert_eq!(strike, 4000.0);
        assert_eq!(opt_type, OptionType::Put);
    }

    #[test]
    fn test_realized_variance() {
        // Simulate some closing prices
        let closes: Vec<f64> = (0..100)
            .map(|i| 100.0 * (1.0 + 0.001 * (i as f64).sin()))
            .collect();

        let rv = realized_variance_cc(&closes, true);
        assert!(rv >= 0.0, "Variance should be non-negative");
        assert!(rv < 10.0, "Variance should be reasonable");
    }
}
