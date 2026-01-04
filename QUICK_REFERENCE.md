# Quick Reference Guide

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Basic Usage

### 1. Mathematical Analysis

```python
from physics_research.math_analysis import NumericalAnalysis

# Differentiate a function
def f(x):
    return x**2

derivative = NumericalAnalysis.differentiate(f, x=2.0)
# Returns: 4.0

# Integrate a function
integral, error = NumericalAnalysis.integrate(lambda x: x**2, 0, 1)
# Returns: (0.333..., error_estimate)
```

### 2. Statistical Analysis

```python
from physics_research.math_analysis import StatisticalAnalysis
import numpy as np

data = np.random.normal(100, 15, 1000)
stats = StatisticalAnalysis.compute_statistics(data)
# Returns: {'mean': ..., 'std': ..., 'var': ..., etc.}
```

### 3. Model Fitting

```python
from physics_research.modeling import PhysicsModels, ParameterFitting

# Fit exponential decay
t_data = np.linspace(0, 5, 50)
y_data = 100 * np.exp(-0.5 * t_data) + noise

result = ParameterFitting.fit_model(
    PhysicsModels.exponential_decay,
    t_data, y_data,
    initial_guess=[90, 0.4]
)
# Returns: parameters, covariance, r_squared, residuals
```

### 4. Hypothesis Testing

```python
from physics_research.hypothesis_testing import HypothesisTesting

# T-test
result = HypothesisTesting.t_test(sample1, sample2)
# Returns: t_statistic, p_value, significance

# Chi-squared test
result = HypothesisTesting.chi_squared_test(observed, expected)
# Returns: chi2_statistic, p_value, degrees_of_freedom
```

### 5. Differential Equations

```python
from physics_research.modeling import DifferentialEquations

# Solve harmonic oscillator
def harmonic(t, y):
    x, v = y
    omega = 2.0
    return [v, -omega**2 * x]

result = DifferentialEquations.solve_system(
    harmonic,
    initial_conditions=[1.0, 0.0],
    time_span=(0, 10)
)
# Returns: {'t': time_array, 'y': solution_array}
```

### 6. Visualization

```python
from physics_research.visualization import DataVisualization

# Time series plot
fig = DataVisualization.plot_time_series(
    t, y,
    title='My Experiment',
    xlabel='Time (s)',
    ylabel='Amplitude'
)

# Save figure
DataVisualization.save_figure(fig, 'output.png', dpi=300)
```

## Common Workflows

### Workflow 1: Experimental Data Analysis

1. Load/generate data
2. Compute statistics: `StatisticalAnalysis.compute_statistics(data)`
3. Fit model: `ParameterFitting.fit_model(model, x, y)`
4. Test hypothesis: `HypothesisTesting.t_test(sample1, sample2)`
5. Visualize: `DataVisualization.plot_scatter(x, y, fit_line=True)`

### Workflow 2: Theoretical Model Analysis

1. Define differential equations
2. Solve system: `DifferentialEquations.solve_system(equations, initial_conditions, time_span)`
3. Analyze solution: `FourierAnalysis.fft(signal, sample_rate)`
4. Visualize: `DataVisualization.plot_phase_space(x, v)`

### Workflow 3: Comparing Experiments

1. Run experiments and collect data
2. Test for significant differences: `HypothesisTesting.anova(group1, group2, group3)`
3. Calculate correlation: `CorrelationAnalysis.pearson_correlation(x, y)`
4. Determine sample size for follow-up: `ExperimentalDesign.power_analysis(effect_size, alpha, power)`

## Tips and Tricks

- Always set `np.random.seed()` for reproducible results
- Use `initial_guess` parameter in model fitting for faster convergence
- Check residuals for model validation
- Use bootstrap methods for non-parametric statistics
- Save figures in high DPI (300) for publications

## Examples

See the `examples/` directory for complete working examples:
- `example_math_analysis.py` - Mathematical tools
- `example_modeling.py` - Physics models and fitting
- `example_hypothesis_testing.py` - Statistical tests
- `demo_complete_workflow.py` - Complete research workflow
