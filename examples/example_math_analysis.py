"""
Example: Mathematical Analysis
Demonstrates numerical differentiation, integration, and Fourier analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from physics_research.math_analysis import NumericalAnalysis, StatisticalAnalysis, FourierAnalysis


def main():
    print("=" * 60)
    print("Physics Research Toolkit - Mathematical Analysis Examples")
    print("=" * 60)
    
    # Example 1: Numerical Differentiation
    print("\n1. Numerical Differentiation")
    print("-" * 40)
    
    def f(x):
        return x**2
    
    x = 2.0
    numerical_derivative = NumericalAnalysis.differentiate(f, x)
    analytical_derivative = 2 * x  # f'(x) = 2x
    
    print(f"Function: f(x) = x²")
    print(f"Numerical derivative at x={x}: {numerical_derivative:.6f}")
    print(f"Analytical derivative at x={x}: {analytical_derivative:.6f}")
    print(f"Error: {abs(numerical_derivative - analytical_derivative):.2e}")
    
    # Example 2: Numerical Integration
    print("\n2. Numerical Integration")
    print("-" * 40)
    
    def g(x):
        return np.sin(x)
    
    a, b = 0, np.pi
    integral, error = NumericalAnalysis.integrate(g, a, b)
    analytical_integral = 2.0  # ∫sin(x)dx from 0 to π = 2
    
    print(f"Function: g(x) = sin(x)")
    print(f"Integral from {a} to {b}: {integral:.6f}")
    print(f"Analytical result: {analytical_integral:.6f}")
    print(f"Error estimate: {error:.2e}")
    
    # Example 3: Statistical Analysis
    print("\n3. Statistical Analysis")
    print("-" * 40)
    
    # Generate sample data
    np.random.seed(42)
    data = np.random.normal(loc=100, scale=15, size=1000)
    
    stats = StatisticalAnalysis.compute_statistics(data)
    
    print("Sample statistics (normal distribution μ=100, σ=15):")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Example 4: Fourier Analysis
    print("\n4. Fourier Analysis")
    print("-" * 40)
    
    # Create a signal with multiple frequencies
    sample_rate = 1000  # Hz
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Signal: 50 Hz + 120 Hz + noise
    signal = (np.sin(2 * np.pi * 50 * t) + 
              0.5 * np.sin(2 * np.pi * 120 * t) + 
              0.1 * np.random.randn(len(t)))
    
    frequencies, magnitudes = FourierAnalysis.fft(signal, sample_rate)
    
    # Find peaks
    peaks_idx = np.where((magnitudes[1:-1] > magnitudes[:-2]) & 
                         (magnitudes[1:-1] > magnitudes[2:]))[0] + 1
    peak_freqs = frequencies[peaks_idx]
    peak_mags = magnitudes[peaks_idx]
    
    # Sort by magnitude
    sorted_idx = np.argsort(peak_mags)[-5:]
    top_frequencies = peak_freqs[sorted_idx]
    
    print(f"Signal contains frequencies: 50 Hz and 120 Hz")
    print(f"Top detected frequencies: {top_frequencies}")
    
    # Example 5: ODE Solver
    print("\n5. Solving Differential Equations")
    print("-" * 40)
    
    # Simple harmonic oscillator: d²x/dt² + ω²x = 0
    def harmonic_oscillator(t, y):
        x, v = y
        omega = 2 * np.pi  # 1 Hz
        return [v, -omega**2 * x]
    
    y0 = [1.0, 0.0]  # Initial position and velocity
    t_span = (0, 2)
    
    solution = NumericalAnalysis.solve_ode(harmonic_oscillator, y0, t_span)
    
    print(f"Solved harmonic oscillator ODE")
    print(f"Time points: {len(solution['t'])}")
    print(f"Initial position: {solution['y'][0][0]:.4f}")
    print(f"Final position: {solution['y'][0][-1]:.4f}")
    print(f"Period ≈ 1 second (as expected)")
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
