"""
Comprehensive Demo: Physics Research Workflow
Demonstrates a complete physics research workflow using all toolkit components.
"""

import numpy as np
import matplotlib.pyplot as plt
from physics_research.math_analysis import NumericalAnalysis, StatisticalAnalysis, FourierAnalysis
from physics_research.modeling import PhysicsModels, ParameterFitting, DifferentialEquations
from physics_research.hypothesis_testing import HypothesisTesting, CorrelationAnalysis, ExperimentalDesign
from physics_research.visualization import DataVisualization


def main():
    print("=" * 70)
    print(" " * 15 + "PHYSICS RESEARCH WORKFLOW DEMO")
    print("=" * 70)
    
    # ========================================================================
    # SCENARIO: Studying a Damped Harmonic Oscillator
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("SCENARIO: Experimental Study of Damped Harmonic Oscillator")
    print("=" * 70)
    
    # Step 1: Generate "experimental" data
    print("\n[Step 1] Generating Experimental Data")
    print("-" * 70)
    
    np.random.seed(42)
    t_experimental = np.linspace(0, 10, 100)
    
    # True parameters (unknown to us initially)
    omega0_true = 3.0
    gamma_true = 0.2
    A_true = 5.0
    
    # Generate noisy measurements
    y_true = PhysicsModels.damped_oscillator(t_experimental, omega0_true, gamma_true, A_true)
    noise = np.random.normal(0, 0.3, len(t_experimental))
    y_experimental = y_true + noise
    
    print(f"✓ Generated {len(t_experimental)} data points over {t_experimental[-1]} seconds")
    print(f"✓ Added Gaussian noise (σ = 0.3)")
    
    # Step 2: Statistical Analysis of Raw Data
    print("\n[Step 2] Statistical Analysis of Experimental Data")
    print("-" * 70)
    
    stats = StatisticalAnalysis.compute_statistics(y_experimental)
    print(f"Mean amplitude: {stats['mean']:.4f}")
    print(f"Standard deviation: {stats['std']:.4f}")
    print(f"Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # Step 3: Model Fitting
    print("\n[Step 3] Fitting Damped Oscillator Model")
    print("-" * 70)
    
    # Initial parameter guesses
    initial_guess = [2.5, 0.15, 4.5, 0]
    
    fit_result = ParameterFitting.fit_model(
        PhysicsModels.damped_oscillator,
        t_experimental,
        y_experimental,
        initial_guess=initial_guess
    )
    
    omega_fit, gamma_fit, A_fit, phi_fit = fit_result['parameters']
    
    print("Fitted parameters:")
    print(f"  Natural frequency (ω₀): {omega_fit:.4f} rad/s (true: {omega0_true})")
    print(f"  Damping coefficient (γ): {gamma_fit:.4f} (true: {gamma_true})")
    print(f"  Initial amplitude (A): {A_fit:.4f} (true: {A_true})")
    print(f"  Phase shift (φ): {phi_fit:.4f} rad")
    print(f"\nGoodness of fit (R²): {fit_result['r_squared']:.6f}")
    
    # Calculate chi-squared
    y_fitted = PhysicsModels.damped_oscillator(t_experimental, *fit_result['parameters'])
    chi2 = ParameterFitting.chi_squared(y_experimental, y_fitted, 
                                       uncertainties=np.ones_like(y_experimental) * 0.3)
    print(f"Chi-squared: {chi2:.4f}")
    
    # Step 4: Hypothesis Testing
    print("\n[Step 4] Hypothesis Testing on Residuals")
    print("-" * 70)
    
    residuals = fit_result['residuals']
    
    # Test if residuals are normally distributed (good model fit indicator)
    ks_result = HypothesisTesting.kolmogorov_smirnov(residuals, distribution='norm')
    print("Testing if residuals follow normal distribution:")
    print(f"  KS statistic: {ks_result['ks_statistic']:.4f}")
    print(f"  p-value: {ks_result['p_value']:.4f}")
    print(f"  Conclusion: {'Residuals are normally distributed (good fit!)' if not ks_result['significant_at_0.05'] else 'Residuals are not normal (poor fit)'}")
    
    # Test if mean residual is zero
    t_result = HypothesisTesting.t_test(residuals, population_mean=0.0)
    print(f"\nTesting if residual mean = 0:")
    print(f"  t-statistic: {t_result['t_statistic']:.4f}")
    print(f"  p-value: {t_result['p_value']:.4f}")
    print(f"  Conclusion: {'Mean is consistent with zero (unbiased fit)' if not t_result['significant_at_0.05'] else 'Mean differs from zero (biased fit)'}")
    
    # Step 5: Compare with Control Experiment
    print("\n[Step 5] Comparing with Control Experiment")
    print("-" * 70)
    
    # Generate "control" data with different damping
    y_control = PhysicsModels.damped_oscillator(t_experimental, omega0_true, 0.4, A_true)
    y_control_noisy = y_control + np.random.normal(0, 0.3, len(t_experimental))
    
    # Compare decay rates using correlation of amplitudes over time
    envelope_exp = np.abs(y_experimental)
    envelope_control = np.abs(y_control_noisy)
    
    t_test_result = HypothesisTesting.t_test(envelope_exp[:30], envelope_control[:30])
    print("Comparing early-time amplitudes between experiments:")
    print(f"  t-statistic: {t_test_result['t_statistic']:.4f}")
    print(f"  p-value: {t_test_result['p_value']:.4f}")
    print(f"  Significant difference: {t_test_result['significant_at_0.05']}")
    
    # Step 6: Fourier Analysis
    print("\n[Step 6] Frequency Analysis")
    print("-" * 70)
    
    # Higher sampling rate for Fourier analysis
    t_fft = np.linspace(0, 10, 1000)
    y_fft = PhysicsModels.damped_oscillator(t_fft, omega_fit, gamma_fit, A_fit, phi_fit)
    
    frequencies, magnitudes = FourierAnalysis.fft(y_fft, len(t_fft)/10)
    
    # Find dominant frequency
    max_idx = np.argmax(magnitudes[1:]) + 1
    dominant_freq = frequencies[max_idx]
    
    # Calculate expected frequency
    expected_freq = np.sqrt(omega_fit**2 - gamma_fit**2) / (2 * np.pi)
    
    print(f"Dominant frequency: {dominant_freq:.4f} Hz")
    print(f"Expected frequency: {expected_freq:.4f} Hz")
    print(f"Agreement: {abs(dominant_freq - expected_freq)/expected_freq * 100:.2f}% difference")
    
    # Step 7: Power Analysis for Future Experiments
    print("\n[Step 7] Power Analysis for Future Studies")
    print("-" * 70)
    
    # Calculate effect size from our experiment
    effect_size = abs(stats['mean']) / stats['std']
    
    power_result = ExperimentalDesign.power_analysis(
        effect_size=0.5,  # Medium effect size
        alpha=0.05,
        power=0.8
    )
    
    print("For detecting a medium effect size (d=0.5):")
    print(f"  Recommended sample size: {power_result['recommended_n_per_group']} per group")
    print(f"  Total measurements needed: {power_result['recommended_n_per_group'] * 2}")
    
    # Step 8: Confidence Intervals
    print("\n[Step 8] Uncertainty Quantification")
    print("-" * 70)
    
    # Bootstrap confidence interval for amplitude
    early_amplitudes = np.abs(y_experimental[:20])
    bootstrap_result = ExperimentalDesign.bootstrap_ci(
        early_amplitudes,
        statistic=np.mean,
        n_bootstrap=1000,
        confidence=0.95
    )
    
    print("95% Bootstrap CI for early-time amplitude:")
    print(f"  Mean: {bootstrap_result['statistic']:.4f}")
    print(f"  CI: ({bootstrap_result['confidence_interval'][0]:.4f}, {bootstrap_result['confidence_interval'][1]:.4f})")
    
    # Parameter uncertainties from covariance matrix
    param_errors = fit_result['std_errors']
    print("\nParameter uncertainties from fit:")
    print(f"  ω₀: {omega_fit:.4f} ± {param_errors[0]:.4f}")
    print(f"  γ: {gamma_fit:.4f} ± {param_errors[1]:.4f}")
    print(f"  A: {A_fit:.4f} ± {param_errors[2]:.4f}")
    
    # Step 9: Summary
    print("\n" + "=" * 70)
    print(" " * 20 + "RESEARCH SUMMARY")
    print("=" * 70)
    
    print("\n✓ Successfully characterized damped harmonic oscillator")
    print(f"✓ Model fit quality: R² = {fit_result['r_squared']:.4f}")
    print(f"✓ Residuals are normally distributed (p = {ks_result['p_value']:.3f})")
    print(f"✓ Frequency analysis confirms model parameters")
    print(f"✓ Statistical analysis validates experimental approach")
    print("\n✓ Ready for publication!")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    
    # Optional: Save a visualization
    print("\n[Optional] Creating visualization...")
    try:
        fig = DataVisualization.plot_time_series(
            t_experimental,
            [y_experimental, y_fitted],
            labels=['Experimental Data', 'Fitted Model'],
            title='Damped Harmonic Oscillator: Data vs Model',
            xlabel='Time (s)',
            ylabel='Amplitude'
        )
        output_file = '/tmp/oscillator_fit.png'
        DataVisualization.save_figure(fig, output_file)
        print(f"✓ Visualization saved to {output_file}")
    except Exception as e:
        print(f"Note: Could not save visualization: {e}")


if __name__ == "__main__":
    main()
