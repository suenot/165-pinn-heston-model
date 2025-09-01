//! Simple feedforward neural network for the Heston PINN.
//!
//! This is a from-scratch implementation using ndarray, without an external
//! deep learning framework. The network uses:
//! - Tanh activations (C-infinity smooth, needed for PDE derivatives)
//! - Softplus output (ensures V >= 0)
//! - Optional residual connections
//!
//! For production use with GPU support, consider tch-rs (PyTorch bindings).

use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// A dense (fully connected) layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseLayer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
}

impl DenseLayer {
    /// Create a new dense layer with Xavier initialization.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();

        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let biases = Array1::zeros(output_dim);

        Self { weights, biases }
    }

    /// Forward pass: output = input @ weights + biases.
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        input.dot(&self.weights) + &self.biases
    }

    /// Forward pass for batch: output = input @ weights + biases (broadcast).
    pub fn forward_batch(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut output = input.dot(&self.weights);
        for mut row in output.rows_mut() {
            row += &self.biases;
        }
        output
    }
}

/// Activation functions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    Tanh,
    Softplus,
    ReLU,
    Identity,
}

impl Activation {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::Tanh => x.tanh(),
            Activation::Softplus => (1.0 + x.exp()).ln().max(0.0),
            Activation::ReLU => x.max(0.0),
            Activation::Identity => x,
        }
    }

    pub fn apply_array(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| self.apply(v))
    }

    /// Derivative (needed for manual backprop).
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::Softplus => 1.0 / (1.0 + (-x).exp()), // sigmoid
            Activation::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Identity => 1.0,
        }
    }
}

/// The PINN neural network for Heston option pricing.
///
/// Architecture: Input(3) -> [Dense + Tanh] x N -> Dense(1) + Softplus
///
/// Input: (S_normalized, v_normalized, t_normalized) in [0, 1]^3
/// Output: V(S, v, t) >= 0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HestonPINN {
    pub layers: Vec<DenseLayer>,
    pub activations: Vec<Activation>,
    pub s_max: f64,
    pub v_max: f64,
    pub t_max: f64,
}

impl HestonPINN {
    /// Create a new PINN with the given architecture.
    ///
    /// # Arguments
    /// * `hidden_dim` - Width of hidden layers
    /// * `num_hidden` - Number of hidden layers
    /// * `s_max` - Max spot price for normalization
    /// * `v_max` - Max variance for normalization
    /// * `t_max` - Max time for normalization
    pub fn new(
        hidden_dim: usize,
        num_hidden: usize,
        s_max: f64,
        v_max: f64,
        t_max: f64,
    ) -> Self {
        let mut layers = Vec::new();
        let mut activations = Vec::new();

        // Input layer: 3 -> hidden_dim
        layers.push(DenseLayer::new(3, hidden_dim));
        activations.push(Activation::Tanh);

        // Hidden layers
        for _ in 0..num_hidden {
            layers.push(DenseLayer::new(hidden_dim, hidden_dim));
            activations.push(Activation::Tanh);
        }

        // Output layer: hidden_dim -> 1
        layers.push(DenseLayer::new(hidden_dim, 1));
        activations.push(Activation::Softplus);

        Self {
            layers,
            activations,
            s_max,
            v_max,
            t_max,
        }
    }

    /// Normalize inputs to [0, 1] range.
    fn normalize(&self, s: f64, v: f64, t: f64) -> Array1<f64> {
        Array1::from(vec![s / self.s_max, v / self.v_max, t / self.t_max])
    }

    /// Forward pass: (S, v, t) -> V(S, v, t).
    pub fn forward(&self, s: f64, v: f64, t: f64) -> f64 {
        let mut x = self.normalize(s, v, t);

        for (layer, activation) in self.layers.iter().zip(self.activations.iter()) {
            x = layer.forward(&x);
            x = activation.apply_array(&x);
        }

        x[0]
    }

    /// Forward pass for a batch of inputs.
    pub fn forward_batch(&self, inputs: &[(f64, f64, f64)]) -> Vec<f64> {
        inputs.iter().map(|&(s, v, t)| self.forward(s, v, t)).collect()
    }

    /// Compute the Heston PDE residual at a point using finite differences.
    ///
    /// This approximates:
    ///     dV/dt + 0.5*v*S^2*d2V/dS2 + rho*sigma*v*S*d2V/dSdv
    ///     + 0.5*sigma^2*v*d2V/dv2 + r*S*dV/dS + kappa*(theta-v)*dV/dv - rV
    pub fn pde_residual(
        &self,
        s: f64,
        v: f64,
        t: f64,
        kappa: f64,
        theta: f64,
        sigma: f64,
        rho: f64,
        r: f64,
    ) -> f64 {
        let ds = s * 0.001;
        let dv = v.max(0.001) * 0.01;
        let dt = 0.001;

        let v_center = self.forward(s, v, t);

        // First derivatives
        let v_s_up = self.forward(s + ds, v, t);
        let v_s_dn = self.forward(s - ds, v, t);
        let dv_ds = (v_s_up - v_s_dn) / (2.0 * ds);

        let v_v_up = self.forward(s, v + dv, t);
        let v_v_dn = self.forward(s, (v - dv).max(1e-8), t);
        let dv_dvar = (v_v_up - v_v_dn) / (2.0 * dv);

        let v_t_up = self.forward(s, v, (t + dt).min(self.t_max));
        let v_t_dn = self.forward(s, v, (t - dt).max(0.0));
        let dv_dt = (v_t_up - v_t_dn) / (2.0 * dt);

        // Second derivatives
        let d2v_ds2 = (v_s_up - 2.0 * v_center + v_s_dn) / (ds * ds);
        let d2v_dv2 = (v_v_up - 2.0 * v_center + v_v_dn) / (dv * dv);

        // Mixed derivative d2V/dSdv
        let v_su_vu = self.forward(s + ds, v + dv, t);
        let v_su_vd = self.forward(s + ds, (v - dv).max(1e-8), t);
        let v_sd_vu = self.forward(s - ds, v + dv, t);
        let v_sd_vd = self.forward(s - ds, (v - dv).max(1e-8), t);
        let d2v_dsdv = (v_su_vu - v_su_vd - v_sd_vu + v_sd_vd) / (4.0 * ds * dv);

        // Heston PDE residual
        dv_dt
            + 0.5 * v * s * s * d2v_ds2
            + rho * sigma * v * s * d2v_dsdv
            + 0.5 * sigma * sigma * v * d2v_dv2
            + r * s * dv_ds
            + kappa * (theta - v) * dv_dvar
            - r * v_center
    }

    /// Total number of parameters in the network.
    pub fn num_parameters(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.len() + l.biases.len())
            .sum()
    }

    /// Simple SGD update for a single parameter perturbation (for training).
    pub fn perturb_and_eval(
        &mut self,
        layer_idx: usize,
        weight_idx: (usize, usize),
        delta: f64,
        eval_fn: &dyn Fn(&HestonPINN) -> f64,
    ) -> f64 {
        let original = self.layers[layer_idx].weights[weight_idx];
        self.layers[layer_idx].weights[weight_idx] = original + delta;
        let loss_plus = eval_fn(self);
        self.layers[layer_idx].weights[weight_idx] = original - delta;
        let loss_minus = eval_fn(self);
        self.layers[layer_idx].weights[weight_idx] = original;

        (loss_plus - loss_minus) / (2.0 * delta)
    }

    /// Save the model to a JSON file.
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a model from a JSON file.
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let model: Self = serde_json::from_str(&json)?;
        Ok(model)
    }
}

/// Simple training configuration.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub n_collocation: usize,
    pub n_boundary: usize,
    pub lambda_pde: f64,
    pub lambda_bc: f64,
    pub lambda_ic: f64,
    pub log_every: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 1000,
            learning_rate: 0.001,
            n_collocation: 512,
            n_boundary: 64,
            lambda_pde: 1.0,
            lambda_bc: 10.0,
            lambda_ic: 10.0,
            log_every: 100,
        }
    }
}

/// Generate collocation points with multi-scale distribution.
pub fn generate_collocation_points(
    s_max: f64,
    v_max: f64,
    t_max: f64,
    strike: f64,
    n_total: usize,
) -> Vec<(f64, f64, f64)> {
    let mut rng = rand::thread_rng();
    let mut points = Vec::with_capacity(n_total);

    let n_uniform = n_total / 2;
    let n_atm = n_total / 4;
    let n_low_v = n_total - n_uniform - n_atm;

    // Uniform background
    for _ in 0..n_uniform {
        let s = rng.gen_range(1.0..s_max);
        let v = rng.gen_range(0.001..v_max);
        let t = rng.gen_range(0.0..t_max);
        points.push((s, v, t));
    }

    // Dense near ATM
    for _ in 0..n_atm {
        let s = strike + 0.2 * strike * rng.gen_range(-1.0..1.0_f64);
        let s = s.clamp(1.0, s_max);
        let v = rng.gen_range(0.001..v_max);
        let t = rng.gen_range(0.0..t_max);
        points.push((s, v, t));
    }

    // Dense at low variance
    for _ in 0..n_low_v {
        let s = rng.gen_range(1.0..s_max);
        let v = rng.gen_range(0.001..v_max * 0.3);
        let t = rng.gen_range(0.0..t_max);
        points.push((s, v, t));
    }

    points
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_forward() {
        let model = HestonPINN::new(32, 3, 200.0, 1.0, 1.0);
        let price = model.forward(100.0, 0.04, 0.0);
        assert!(price >= 0.0, "Price must be non-negative, got {}", price);
        println!("Untrained price at S=100, v=0.04, t=0: {:.6}", price);
    }

    #[test]
    fn test_network_params() {
        let model = HestonPINN::new(64, 4, 200.0, 1.0, 1.0);
        let n_params = model.num_parameters();
        println!("Total parameters: {}", n_params);
        assert!(n_params > 0);
    }

    #[test]
    fn test_collocation_points() {
        let points = generate_collocation_points(200.0, 1.0, 1.0, 100.0, 256);
        assert_eq!(points.len(), 256);
        for (s, v, t) in &points {
            assert!(*s > 0.0 && *s <= 200.0);
            assert!(*v > 0.0 && *v <= 1.0);
            assert!(*t >= 0.0 && *t <= 1.0);
        }
    }

    #[test]
    fn test_pde_residual() {
        let model = HestonPINN::new(32, 3, 200.0, 1.0, 1.0);
        let residual = model.pde_residual(100.0, 0.04, 0.25, 2.0, 0.04, 0.3, -0.7, 0.05);
        // Untrained model will have non-zero residual, just check it's finite
        assert!(residual.is_finite(), "Residual should be finite");
        println!("PDE residual (untrained): {:.6}", residual);
    }
}
