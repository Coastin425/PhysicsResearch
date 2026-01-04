"""
Example: Hypothesis Testing
Demonstrates statistical hypothesis testing and correlation analysis.
"""

import numpy as np
from physics_research.hypothesis_testing import (
    HypothesisTesting, CorrelationAnalysis, ExperimentalDesign
)


def main():
    print("=" * 60)
    print("Physics Research Toolkit - Hypothesis Testing Examples")
    print("=" * 60)
    
    # Example 1: T-Test
    print("\n1. T-Test: Comparing Two Experimental Groups")
    print("-" * 40)
    
    np.random.seed(42)
    # Control group: mean = 100
    control = np.random.normal(100, 15, 50)
    # Treatment group: mean = 105 (small effect)
    treatment = np.random.normal(105, 15, 50)
    
    result = HypothesisTesting.t_test(control, treatment)
    
    print(f"Control group mean: {np.mean(control):.2f}")
    print(f"Treatment group mean: {np.mean(treatment):.2f}")
    print(f"t-statistic: {result['t_statistic']:.4f}")
    print(f"p-value: {result['p_value']:.4f}")
    print(f"Significant at α=0.05: {result['significant_at_0.05']}")
    
    # Example 2: Chi-Squared Test
    print("\n2. Chi-Squared Test: Goodness of Fit")
    print("-" * 40)
    
    # Observed frequencies from dice rolls
    observed = np.array([18, 22, 19, 17, 21, 23])
    # Expected frequencies for fair dice (uniform)
    expected = np.array([20, 20, 20, 20, 20, 20])
    
    result = HypothesisTesting.chi_squared_test(observed, expected)
    
    print("Testing if a dice is fair (6 faces, 120 rolls)")
    print(f"Observed frequencies: {observed}")
    print(f"Expected frequencies: {expected}")
    print(f"χ² statistic: {result['chi2_statistic']:.4f}")
    print(f"p-value: {result['p_value']:.4f}")
    print(f"Significant at α=0.05: {result['significant_at_0.05']}")
    print("Conclusion: Dice appears fair" if not result['significant_at_0.05'] else "Conclusion: Dice may be biased")
    
    # Example 3: ANOVA Test
    print("\n3. ANOVA: Comparing Multiple Groups")
    print("-" * 40)
    
    # Three experimental conditions
    group1 = np.random.normal(100, 10, 30)
    group2 = np.random.normal(105, 10, 30)
    group3 = np.random.normal(110, 10, 30)
    
    result = HypothesisTesting.anova(group1, group2, group3)
    
    print(f"Group 1 mean: {np.mean(group1):.2f}")
    print(f"Group 2 mean: {np.mean(group2):.2f}")
    print(f"Group 3 mean: {np.mean(group3):.2f}")
    print(f"F-statistic: {result['f_statistic']:.4f}")
    print(f"p-value: {result['p_value']:.4e}")
    print(f"Significant at α=0.05: {result['significant_at_0.05']}")
    
    # Example 4: Correlation Analysis
    print("\n4. Correlation Analysis")
    print("-" * 40)
    
    # Generate correlated data
    x = np.random.randn(100)
    y = 2 * x + np.random.randn(100) * 0.5  # Strong positive correlation
    
    pearson_result = CorrelationAnalysis.pearson_correlation(x, y)
    
    print("Testing correlation between two variables")
    print(f"Pearson correlation coefficient: {pearson_result['correlation_coefficient']:.4f}")
    print(f"p-value: {pearson_result['p_value']:.4e}")
    print(f"Correlation strength: {pearson_result['strength']}")
    print(f"Significant at α=0.05: {pearson_result['significant_at_0.05']}")
    
    # Example 5: Power Analysis
    print("\n5. Power Analysis: Sample Size Determination")
    print("-" * 40)
    
    # Calculate required sample size
    effect_size = 0.5  # Medium effect size (Cohen's d)
    alpha = 0.05
    power = 0.8
    
    result = ExperimentalDesign.power_analysis(effect_size, alpha, power)
    
    print(f"Effect size (Cohen's d): {effect_size}")
    print(f"Significance level (α): {alpha}")
    print(f"Desired power: {power}")
    print(f"Recommended sample size per group: {result['recommended_n_per_group']}")
    print(f"Total participants needed: {result['recommended_n_per_group'] * 2}")
    
    # Example 6: Confidence Intervals
    print("\n6. Confidence Interval Calculation")
    print("-" * 40)
    
    data = np.random.normal(100, 15, 50)
    
    ci_result = ExperimentalDesign.confidence_interval(data, confidence=0.95)
    
    print(f"Sample mean: {ci_result['mean']:.2f}")
    print(f"95% confidence interval: ({ci_result['confidence_interval'][0]:.2f}, {ci_result['confidence_interval'][1]:.2f})")
    print(f"Margin of error: ±{ci_result['margin_of_error']:.2f}")
    
    # Example 7: Bootstrap Confidence Interval
    print("\n7. Bootstrap Confidence Interval")
    print("-" * 40)
    
    # For a non-parametric estimate
    data = np.random.exponential(scale=2, size=100)
    
    bootstrap_result = ExperimentalDesign.bootstrap_ci(
        data, 
        statistic=np.median,
        n_bootstrap=1000
    )
    
    print("Bootstrap CI for median (non-parametric)")
    print(f"Sample median: {bootstrap_result['statistic']:.4f}")
    print(f"95% bootstrap CI: ({bootstrap_result['confidence_interval'][0]:.4f}, {bootstrap_result['confidence_interval'][1]:.4f})")
    
    # Example 8: Kolmogorov-Smirnov Test
    print("\n8. Kolmogorov-Smirnov Test: Distribution Testing")
    print("-" * 40)
    
    # Generate data from a normal distribution
    normal_data = np.random.normal(0, 1, 200)
    
    result = HypothesisTesting.kolmogorov_smirnov(normal_data, distribution='norm')
    
    print("Testing if data follows normal distribution")
    print(f"KS statistic: {result['ks_statistic']:.4f}")
    print(f"p-value: {result['p_value']:.4f}")
    print(f"Significant at α=0.05: {result['significant_at_0.05']}")
    print("Conclusion: Data is normally distributed" if not result['significant_at_0.05'] else "Conclusion: Data is not normally distributed")
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
