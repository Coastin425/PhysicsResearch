# Physics Research Toolkit

A comprehensive Python toolkit for deep physics research, including mathematical analysis, modeling, hypothesis testing, and visualization tools.

## Overview

This toolkit provides researchers with powerful tools to:
- Perform deep mathematical analysis (numerical methods, statistics, Fourier transforms)
- Build and validate mathematical models of physical systems
- Conduct rigorous hypothesis testing and statistical analysis
- Visualize research data and results

## Features

### 📊 Mathematical Analysis
- **Numerical Methods**: Differentiation, integration, ODE solvers
- **Statistical Analysis**: Comprehensive statistics, correlation, linear regression
- **Fourier Analysis**: FFT, power spectrum, signal filtering

### 🔬 Mathematical Modeling
- **Physics Models**: Harmonic oscillators, exponential decay, Gaussian distributions
- **Parameter Fitting**: Non-linear least squares fitting, chi-squared analysis
- **Differential Equations**: System solvers for ODEs, pendulum, driven oscillators
- **Symbolic Analysis**: Symbolic differentiation, integration, Taylor series

### 🧪 Hypothesis Testing
- **Statistical Tests**: t-tests, chi-squared, ANOVA, Kolmogorov-Smirnov
- **Correlation Analysis**: Pearson, Spearman, partial correlation
- **Experimental Design**: Power analysis, confidence intervals, bootstrap methods

### 📈 Visualization
- Time series plots
- Scatter plots with regression
- Histograms and distributions
- Residual analysis
- Frequency spectra
- Phase space plots
- Contour plots

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Coastin425/PhysicsResearch.git
cd PhysicsResearch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Example 1: Numerical Analysis
```python
from physics_research.math_analysis import NumericalAnalysis
import numpy as np

# Compute a derivative
def f(x):
    return x**2

derivative = NumericalAnalysis.differentiate(f, x=2.0)
print(f"Derivative at x=2: {derivative}")  # Output: 4.0

# Integrate a function
def g(x):
    return np.sin(x)

integral, error = NumericalAnalysis.integrate(g, 0, np.pi)
print(f"Integral: {integral}")  # Output: 2.0
```

### Example 2: Model Fitting
```python
from physics_research.modeling import PhysicsModels, ParameterFitting
import numpy as np

# Generate noisy data
t = np.linspace(0, 5, 50)
y_data = 100 * np.exp(-0.5 * t) + np.random.normal(0, 3, 50)

# Fit exponential decay model
result = ParameterFitting.fit_model(
    PhysicsModels.exponential_decay,
    t, y_data,
    initial_guess=[90, 0.4]
)

print(f"Fitted parameters: {result['parameters']}")
print(f"R²: {result['r_squared']}")
```

### Example 3: Hypothesis Testing
```python
from physics_research.hypothesis_testing import HypothesisTesting
import numpy as np

# Compare two experimental groups
control = np.random.normal(100, 15, 50)
treatment = np.random.normal(105, 15, 50)

result = HypothesisTesting.t_test(control, treatment)
print(f"p-value: {result['p_value']}")
print(f"Significant: {result['significant_at_0.05']}")
```

## Examples

Run the included example scripts to see the toolkit in action:

```bash
# Mathematical analysis examples
python examples/example_math_analysis.py

# Modeling examples
python examples/example_modeling.py

# Hypothesis testing examples
python examples/example_hypothesis_testing.py
```

## Module Documentation

### physics_research.math_analysis

#### NumericalAnalysis
- `differentiate(f, x, h)` - Numerical differentiation
- `integrate(f, a, b, method)` - Numerical integration
- `solve_ode(f, y0, t_span, t_eval)` - ODE solver

#### StatisticalAnalysis
- `compute_statistics(data)` - Comprehensive statistics
- `correlation(x, y)` - Pearson correlation
- `linear_regression(x, y)` - Linear regression analysis

#### FourierAnalysis
- `fft(signal_data, sample_rate)` - Fast Fourier Transform
- `power_spectrum(signal_data, sample_rate)` - Power spectral density
- `filter_signal(signal_data, cutoff, sample_rate, filter_type)` - Signal filtering

### physics_research.modeling

#### PhysicsModels
- `harmonic_oscillator(t, omega, A, phi)` - Simple harmonic motion
- `damped_oscillator(t, omega0, gamma, A, phi)` - Damped oscillation
- `exponential_decay(t, N0, lambda_decay)` - Exponential decay
- `gaussian(x, mu, sigma, A)` - Gaussian distribution

#### ParameterFitting
- `fit_model(model, x_data, y_data, initial_guess, bounds)` - Model fitting
- `chi_squared(observed, expected, uncertainties)` - Chi-squared statistic

#### DifferentialEquations
- `solve_system(equations, initial_conditions, time_span)` - System solver
- `pendulum(t, state, g, L)` - Pendulum equations
- `damped_driven_oscillator(t, state, omega0, gamma, F0, omega_d)` - Driven oscillator

#### SymbolicAnalysis
- `create_symbols(symbol_string)` - Create symbolic variables
- `differentiate(expression, variable, order)` - Symbolic differentiation
- `integrate(expression, variable, limits)` - Symbolic integration
- `taylor_series(expression, variable, point, order)` - Taylor expansion

### physics_research.hypothesis_testing

#### HypothesisTesting
- `t_test(sample1, sample2, population_mean, alternative)` - t-test
- `chi_squared_test(observed, expected)` - Chi-squared test
- `anova(*samples)` - One-way ANOVA
- `kolmogorov_smirnov(sample, distribution, params)` - K-S test

#### CorrelationAnalysis
- `pearson_correlation(x, y)` - Pearson correlation
- `spearman_correlation(x, y)` - Spearman correlation
- `partial_correlation(x, y, z)` - Partial correlation

#### ExperimentalDesign
- `power_analysis(effect_size, alpha, power, test_type)` - Sample size calculation
- `confidence_interval(data, confidence)` - Confidence interval
- `bootstrap_ci(data, statistic, n_bootstrap, confidence)` - Bootstrap CI

### physics_research.visualization

#### DataVisualization
- `plot_time_series(t, y, labels, title, xlabel, ylabel)` - Time series plot
- `plot_scatter(x, y, title, xlabel, ylabel, fit_line)` - Scatter plot
- `plot_histogram(data, bins, title, density)` - Histogram
- `plot_residuals(y_true, y_pred, x)` - Residual plot
- `plot_spectrum(frequencies, magnitudes, log_scale)` - Frequency spectrum
- `plot_phase_space(x, v)` - Phase space plot
- `plot_contour(x, y, z, levels)` - Contour plot
- `save_figure(fig, filename, dpi)` - Save figure to file

## Dependencies

- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Matplotlib >= 3.7.0
- SymPy >= 1.12
- Pandas >= 2.0.0

## Use Cases

This toolkit is designed for:
- **Experimental Physics**: Analyze experimental data, fit models, test hypotheses
- **Theoretical Physics**: Solve differential equations, perform symbolic analysis
- **Computational Physics**: Numerical simulations, parameter optimization
- **Data Analysis**: Statistical analysis of research data
- **Education**: Teaching physics and data analysis concepts

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## License

This project is open source and available for physics research and education.

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.
