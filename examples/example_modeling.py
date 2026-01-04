"""
Example: Mathematical Modeling
Demonstrates physics models, parameter fitting, and differential equations.
"""

import numpy as np
import matplotlib.pyplot as plt
from physics_research.modeling import PhysicsModels, ParameterFitting, DifferentialEquations
from physics_research.visualization import DataVisualization


def main():
    print("=" * 60)
    print("Physics Research Toolkit - Mathematical Modeling Examples")
    print("=" * 60)
    
    # Example 1: Harmonic Oscillator Model
    print("\n1. Simple Harmonic Oscillator")
    print("-" * 40)
    
    t = np.linspace(0, 10, 1000)
    omega = 2 * np.pi * 0.5  # 0.5 Hz
    A = 2.0
    phi = np.pi / 4
    
    x = PhysicsModels.harmonic_oscillator(t, omega, A, phi)
    
    print(f"Frequency: {omega/(2*np.pi):.2f} Hz")
    print(f"Amplitude: {A:.2f}")
    print(f"Phase shift: {phi:.4f} rad")
    print(f"Max displacement: {np.max(x):.4f}")
    print(f"Min displacement: {np.min(x):.4f}")
    
    # Example 2: Parameter Fitting
    print("\n2. Parameter Fitting with Noisy Data")
    print("-" * 40)
    
    # Generate noisy exponential decay data
    np.random.seed(42)
    t_data = np.linspace(0, 5, 50)
    N0_true = 100
    lambda_true = 0.5
    noise = np.random.normal(0, 3, len(t_data))
    y_data = PhysicsModels.exponential_decay(t_data, N0_true, lambda_true) + noise
    
    # Fit the model
    fit_result = ParameterFitting.fit_model(
        PhysicsModels.exponential_decay,
        t_data,
        y_data,
        initial_guess=[90, 0.4]
    )
    
    N0_fit, lambda_fit = fit_result['parameters']
    
    print("Exponential decay: N(t) = N0 * exp(-λt)")
    print(f"True parameters: N0={N0_true}, λ={lambda_true}")
    print(f"Fitted parameters: N0={N0_fit:.2f}, λ={lambda_fit:.4f}")
    print(f"R²: {fit_result['r_squared']:.6f}")
    print(f"Parameter uncertainties: {fit_result['std_errors']}")
    
    # Example 3: Pendulum Simulation
    print("\n3. Nonlinear Pendulum Simulation")
    print("-" * 40)
    
    # Initial conditions: 45 degrees, no initial velocity
    theta0 = np.pi / 4
    omega0 = 0.0
    initial_conditions = [theta0, omega0]
    
    # Solve the pendulum equations
    result = DifferentialEquations.solve_system(
        DifferentialEquations.pendulum,
        initial_conditions,
        (0, 10),
        num_points=1000
    )
    
    theta = result['y'][0]
    omega = result['y'][1]
    
    print(f"Initial angle: {theta0 * 180/np.pi:.1f} degrees")
    print(f"Maximum angle: {np.max(np.abs(theta)) * 180/np.pi:.1f} degrees")
    print(f"Maximum angular velocity: {np.max(np.abs(omega)):.4f} rad/s")
    print(f"Energy approximately conserved (periodic motion)")
    
    # Example 4: Damped Driven Oscillator
    print("\n4. Damped Driven Harmonic Oscillator")
    print("-" * 40)
    
    omega0 = 2.0  # Natural frequency
    gamma = 0.1   # Damping coefficient
    F0 = 1.0      # Driving force amplitude
    omega_d = 1.8 # Driving frequency
    
    initial_conditions = [1.0, 0.0]  # Initial position and velocity
    
    # Create a wrapper function for the driven oscillator
    def driven_osc(t, y):
        return DifferentialEquations.damped_driven_oscillator(
            t, y, omega0, gamma, F0, omega_d
        )
    
    result = DifferentialEquations.solve_system(
        driven_osc,
        initial_conditions,
        (0, 50),
        num_points=5000
    )
    
    x = result['y'][0]
    v = result['y'][1]
    
    print(f"Natural frequency: {omega0:.2f} rad/s")
    print(f"Driving frequency: {omega_d:.2f} rad/s")
    print(f"Damping coefficient: {gamma:.2f}")
    print(f"System reaches steady state after transient")
    print(f"Steady-state amplitude: {np.std(x[-1000:]):.4f}")
    
    # Example 5: Gaussian Distribution Model
    print("\n5. Gaussian Distribution Fitting")
    print("-" * 40)
    
    # Generate data from a Gaussian
    np.random.seed(42)
    x_data = np.random.normal(loc=5, scale=2, size=1000)
    
    # Create histogram bins
    counts, bin_edges = np.histogram(x_data, bins=30)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Fit Gaussian model
    fit_result = ParameterFitting.fit_model(
        PhysicsModels.gaussian,
        bin_centers,
        counts,
        initial_guess=[5, 2, 100]
    )
    
    mu, sigma, A = fit_result['parameters']
    
    print(f"True distribution: μ=5, σ=2")
    print(f"Fitted distribution: μ={mu:.2f}, σ={sigma:.2f}")
    print(f"R²: {fit_result['r_squared']:.6f}")
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
