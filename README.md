# 🚀 Fractional-Neural Jump-Diffusion Credit Risk Model

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-purple)](https://qiskit.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-ICQFRPM%202025-yellow)](https://example.com/paper)

> **A cutting-edge quantum-enhanced neural framework for multi-regime credit risk assessment with path-dependent memory effects** 🧠⚡

---

## 🌟 What is This Project?

This repository implements a **revolutionary approach** to credit risk modeling that combines five groundbreaking mathematical contributions into a unified framework. Think of it as the fusion of:

- 🧠 **Neural Networks** (for learning complex patterns)
- 🌊 **Fractional Brownian Motion** (for memory effects)
- 🔄 **Regime Switching** (for market state transitions)
- ⚛️ **Quantum Computing** (for portfolio optimization)
- 📈 **Jump-Diffusion Processes** (for sudden market shocks)

### 🎯 The Big Picture

Traditional credit risk models assume markets have no memory and follow simple patterns. **We know that's not true!** 

This model captures:
- **Memory effects**: Past events influence future risk (via fractional kernels)
- **Regime changes**: Markets switch between normal, stress, and crisis states
- **Jump shocks**: Sudden credit events that traditional models miss
- **Quantum optimization**: Leveraging quantum computers for better portfolio allocation

---

## 📄 Conference Paper Support

This code supports our **ICQFRPM 2025** conference paper:

### 📖 Paper: Fractional-Neural Jump-Diffusion Models for Multi-Regime Credit Risk Assessment: A Quantum-Enhanced Optimization Framework with Path-Dependent Memory Effects

**Key Contributions:**
1. **Novel Fractional Kernel**: Mathematically rigorous implementation of memory effects in credit dynamics
2. **Neural SDE Framework**: Time-dependent drift and volatility learned from data
3. **Regime Learning**: Neural networks that discover hidden market states
4. **Quantum Portfolio Optimization**: QAOA-based solution for combinatorial portfolio problems
5. **Memory-Augmented Jump Intensity**: Jump probabilities that depend on historical path

**How the Code Relates:**
- Each mathematical contribution has its own class implementation
- Full end-to-end pipeline from data generation to portfolio optimization
- Extensive visualization and validation of theoretical results
- Production-ready code with proper error handling and documentation

---

## ✨ Features

### 🧮 Mathematical Innovation
- **Fractional Kernel**: `K_H(t,s) = (t-s)^(H-1/2) / Γ(H+1/2)` - mathematically correct!
- **Neural SDE**: `dX_t = μ_θ(X_t,t,S_t)dt + σ_θ(X_t,t,S_t)dB^H_t + dJ_t`
- **Learned Transitions**: `P(S_{t+1}=j|S_t=i)` discovered via neural networks
- **Memory Integration**: `λ(t) = f_θ(∫ K(t,s)X_s ds)` for path-dependent effects

### 🔬 Technical Excellence
- **Proper fBm Generation**: Cholesky decomposition of exact covariance matrix
- **Regime-Dependent Networks**: Separate neural networks for each market regime
- **Quantum QAOA**: Real quantum optimization with ZZ interactions
- **Publication-Quality Plots**: 11 comprehensive visualizations

### 🎮 User-Friendly
- **One-Click Execution**: Run `main()` and watch the magic happen
- **Configurable Parameters**: Easy to adjust Hurst parameter, regimes, assets
- **Extensive Logging**: See exactly what each component is doing
- **Error Handling**: Graceful fallbacks when quantum hardware isn't available

---

## 🔧 Setup & Installation

### 📋 Prerequisites
```bash
Python 3.8+
CUDA (optional, for GPU acceleration)
```

### 🚀 Quick Install
```bash
# Clone the repository
git clone https://github.com/NiharJani2002/Fractional-Neural-Jump-Diffusion-Models-for-Multi-Regime-Credit-Risk-Assessment-A-Quantum
cd fractional-neural-credit-risk

# Install dependencies
pip install numpy scipy matplotlib torch scikit-learn pandas yfinance
pip install qiskit qiskit-algorithms qiskit-ibm-runtime qiskit-optimization qiskit-aer

# For development
pip install -r requirements-dev.txt
```

### 🐍 Alternative: Conda Environment
```bash
conda create -n credit-risk python=3.9
conda activate credit-risk
pip install -r requirements.txt
```

---

## 🎯 Usage Examples

### 🏃‍♂️ Quick Start (5 minutes)
```python
# Import and run everything!
from fractional_neural_jump_diffusion_models import main

# Execute the full pipeline
main()
```

### 🛠️ Custom Configuration
```python
from fractional_neural_jump_diffusion_models import *

# Configure your model
config = {
    'n_assets': 10,          # Number of credit instruments
    'hurst': 0.7,            # Memory parameter (0.5 = no memory)
    'n_regimes': 3,          # Market states (normal/stress/crisis)
    'T': 2.0,                # Time horizon (years)
    'dt': 0.01,              # Time step
    'hidden_size': 128       # Neural network size
}

# Generate synthetic credit data
spreads, regimes = generate_realistic_credit_data(
    T=config['T'], 
    dt=config['dt'], 
    n_assets=config['n_assets'], 
    hurst=config['hurst']
)

# Initialize the neural SDE model
model = RegimeSwitchingNeuralSDE(
    dim=config['n_assets'],
    n_regimes=config['n_regimes'],
    hurst=config['hurst'],
    hidden_size=config['hidden_size']
)

# Train the model
fbm_gen = FractionalBrownianMotion(config['hurst'], config['T'], int(config['T']/config['dt'])-1)
model = train_model(model, spreads, regimes, fbm_gen, n_epochs=100)

print("🎉 Model trained successfully!")
```

### 🎨 Visualization Only
```python
# Just want to see the pretty plots?
spreads, regimes = generate_realistic_credit_data()
weights = np.ones(5) / 5  # Equal weights
returns = np.diff(spreads, axis=0)

create_final_plots(spreads, regimes, weights, returns[-252:] @ weights, 
                  model, 0.7, np.eye(3)/3)
```

---

## 🧠 How It Works

### 🏗️ Architecture Overview

```
Input Data (Credit Spreads)
           ↓
    ┌─────────────────────┐
    │  Fractional Kernel  │ ← Memory Effects
    └─────────────────────┘
           ↓
    ┌─────────────────────┐
    │   Neural SDE        │ ← Learning Dynamics
    │  ├─ Drift Network   │
    │  ├─ Vol Network     │
    │  └─ Jump Intensity  │
    └─────────────────────┘
           ↓
    ┌─────────────────────┐
    │  Regime Learning    │ ← Hidden States
    └─────────────────────┘
           ↓
    ┌─────────────────────┐
    │ Quantum Portfolio   │ ← Optimization
    │    Optimization     │
    └─────────────────────┘
           ↓
    Optimal Portfolio Weights
```

### 🔍 Component Deep Dive

#### 1. **Fractional Kernel** 🌊
```python
class FractionalKernel:
    """
    Implements K_H(t,s) = (t-s)^(H-1/2) / Γ(H+1/2)
    
    This is THE mathematically correct kernel for fractional Brownian motion.
    H > 0.5: Long memory (persistent)
    H < 0.5: Anti-persistent (mean reverting)
    H = 0.5: Standard Brownian motion (no memory)
    """
```

#### 2. **Neural SDE** 🧠
```python
class RegimeSwitchingNeuralSDE:
    """
    Solves: dX_t = μ_θ(X_t,t,S_t)dt + σ_θ(X_t,t,S_t)dB^H_t + dJ_t
    
    Where:
    - μ_θ: Neural network for drift (mean reversion)
    - σ_θ: Neural network for volatility (risk)
    - S_t: Hidden regime state
    - dJ_t: Compound Poisson jumps (credit events)
    """
```

#### 3. **Quantum Optimizer** ⚛️
```python
class QuantumPortfolioOptimizer:
    """
    Solves: min x^T Q x subject to x ∈ {0,1}^n
    
    Uses QAOA (Quantum Approximate Optimization Algorithm):
    - Encodes portfolio problem as QUBO
    - Converts to Ising Hamiltonian
    - Optimizes using quantum variational circuits
    """
```

---

## 📊 Expected Output

When you run the code, you'll see:

### 🖥️ Console Output
```
================================================================================
FRACTIONAL-NEURAL CREDIT RISK MODEL - ALL 5 CONTRIBUTIONS CORRECT
================================================================================

[1/5] Data Generation...
✓ Generated 200 time steps with CORRECT fractional kernel H=0.7

[2/5] Model Initialization...
✓ Neural SDE initialized with 3 regimes, dimension 5

[3/5] Training...
Epoch   0 | SDE Loss: 0.045231 | Trans Loss: -0.023451
Epoch  10 | SDE Loss: 0.012891 | Trans Loss: -0.019234
...
✓ Training complete

[4/5] Portfolio Optimization...
✓ Classical weights: [0.156 0.234 0.198 0.201 0.211]
  Objective: -0.008234
✓ Quantum weights: [0.200 0.200 0.200 0.200 0.200]
  Objective: -0.007891

[5/5] Performance Evaluation...

================================================================================
RESULTS - ALL 5 CONTRIBUTIONS VERIFIED
================================================================================

✓ CONTRIBUTION 1: Fractional Kernel K_H(t,s) = (t-s)^(H-1/2) / Γ(H+1/2)
✓ CONTRIBUTION 2: Neural SDE dX_t = μ_θ(X_t,t,S_t)dt + σ_θ(X_t,t,S_t)dB^H_t + dJ_t
✓ CONTRIBUTION 3: Regime Transition P(S_{t+1}=j|S_t=i) LEARNED
✓ CONTRIBUTION 4: Quantum QUBO min x^T Q x
✓ CONTRIBUTION 5: Memory-Augmented Intensity λ(t) = f_θ(∫ K(t,s)X_s ds)

Portfolio Performance:
Annual Sharpe Ratio: 3.6436
VaR (95%): -30.73 bps
Max Drawdown: -3.03%
```

### 📈 Visual Output
The code generates a comprehensive 11-panel visualization showing:

1. **Credit Spread Evolution** - Time series of all assets
2. **Regime States** - Market conditions over time  
3. **Fractional Kernel** - Memory weighting function
4. **Transition Matrix** - Learned regime probabilities
5. **Portfolio Weights** - Optimal allocation
6. **Neural Architecture** - Model structure diagram
7. **Return Distribution** - Portfolio performance histogram
8. **Cumulative Returns** - Portfolio growth over time
9. **Correlation Matrix** - Asset relationships
10. **Drawdown Analysis** - Risk assessment
11. **Performance Summary** - Key statistics

---

## 🔬 Behind the Scenes

### 🎭 Fun Facts

- **Mathematical Precision**: The fractional kernel implementation uses the EXACT Mandelbrot-Van Ness representation (not approximations!)
- **Quantum Ready**: Code automatically detects if quantum hardware is available and gracefully falls back to classical optimization
- **Memory Magic**: The H=0.7 Hurst parameter means the model has "long memory" - events from weeks ago still influence today's risk!
- **Regime Detection**: The neural network discovers market regimes WITHOUT being told what they are
- **Production Quality**: Used proper software engineering practices with type hints, error handling, and comprehensive logging

### 🧪 Implementation Choices

**Why Cholesky Decomposition for fBm?**
We use the mathematically rigorous approach rather than fast approximations. This ensures our fractional Brownian motion has the EXACT covariance structure theory predicts.

**Why QAOA over Classical Optimizers?**
Portfolio optimization is fundamentally combinatorial (which assets to include?). Quantum computers have theoretical advantages for these NP-hard problems.

**Why Layer Normalization in Neural Networks?**
Financial time series have varying scales and non-stationary statistics. Layer normalization makes training more stable.

### 🎪 Easter Eggs

- Look for the "FIXED: Was..." comments in the quantum optimizer - we found and fixed several subtle bugs!
- The regime colors in plots (green/yellow/red) represent normal/stress/crisis states
- ASCII art in the neural architecture diagram took way too long to perfect 😅

---
```

---

## 🤝 Contributing

We love contributions! Here's how you can help:

### 🐛 Bug Reports
Found a bug? Please open an issue with:
- Python version
- Operating system
- Error message and stack trace
- Minimal code to reproduce

### 💡 Feature Requests
Have an idea? We'd love to hear it! Open an issue with:
- Clear description of the feature
- Why it would be useful
- Example use case

### 🔧 Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Make your changes with tests
4. Submit a pull request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Fractional Brownian Motion Theory**: Mandelbrot & Van Ness (1968)
- **Regime Switching Models**: Hamilton (1989)
- **Neural SDEs**: Chen et al. (2018)
- **QAOA Algorithm**: Farhi, Goldstone & Gutmann (2014)
- **The Coffee**: Local coffee shop for fueling late-night coding sessions ☕

---

## 📞 Contact & Support

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Join our [Discussions](https://github.com/NiharJani2002/Fractional-Neural-Jump-Diffusion-Models-for-Multi-Regime-Credit-Risk-Assessment-A-Quantum/discussions) for questions
- **Email**: [niharmaheshjani@gmail.com](mailto:niharmaheshjani@gmail.com)
- **Paper Questions**: Reference ICQFRPM 2025 proceedings

---

## 🚀 What's Next?

### 🎋 Future Enhancements
- [ ] Multi-asset correlation learning
- [ ] Real-time data integration
- [ ] GPU acceleration for large portfolios
- [ ] Web interface for interactive exploration
- [ ] Integration with quantum cloud services

### 🌍 Real-World Applications
- **Risk Management**: Banks and hedge funds
- **Regulatory Compliance**: Basel III stress testing
- **Portfolio Construction**: Quantitative asset management
- **Research**: Academic studies in quantitative finance

---

⭐ **If you find this project useful, please star the repository!** ⭐

---

*Made with ❤️ by researchers who believe quantum + AI = the future of finance*
