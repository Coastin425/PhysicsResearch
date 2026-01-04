"""
Hypothesis Testing Module
Provides tools for statistical hypothesis testing and experimental validation.
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional, Union


class HypothesisTesting:
    """Statistical hypothesis testing tools for physics experiments."""
    
    @staticmethod
    def t_test(sample1: np.ndarray, sample2: Optional[np.ndarray] = None,
               population_mean: Optional[float] = None,
               alternative: str = 'two-sided') -> Dict:
        """
        Perform t-test for comparing means.
        
        Args:
            sample1: First sample or single sample
            sample2: Second sample (for two-sample test)
            population_mean: Population mean (for one-sample test)
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            Dictionary with t-statistic, p-value, and conclusion
        """
        if sample2 is not None:
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(sample1, sample2, alternative=alternative)
            test_type = "Two-sample t-test"
        elif population_mean is not None:
            # One-sample t-test
            t_stat, p_value = stats.ttest_1samp(sample1, population_mean, alternative=alternative)
            test_type = "One-sample t-test"
        else:
            raise ValueError("Must provide either sample2 or population_mean")
        
        return {
            'test_type': test_type,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01
        }
    
    @staticmethod
    def chi_squared_test(observed: np.ndarray, expected: np.ndarray) -> Dict:
        """
        Perform chi-squared goodness-of-fit test.
        
        Args:
            observed: Observed frequencies
            expected: Expected frequencies
            
        Returns:
            Dictionary with chi-squared statistic, p-value, and degrees of freedom
        """
        chi2_stat, p_value = stats.chisquare(observed, expected)
        dof = len(observed) - 1
        
        return {
            'test_type': 'Chi-squared test',
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01
        }
    
    @staticmethod
    def anova(*samples: np.ndarray) -> Dict:
        """
        Perform one-way ANOVA test.
        
        Args:
            *samples: Multiple sample arrays
            
        Returns:
            Dictionary with F-statistic and p-value
        """
        f_stat, p_value = stats.f_oneway(*samples)
        
        return {
            'test_type': 'One-way ANOVA',
            'f_statistic': f_stat,
            'p_value': p_value,
            'num_groups': len(samples),
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01
        }
    
    @staticmethod
    def kolmogorov_smirnov(sample: np.ndarray, distribution: str = 'norm',
                          params: Optional[Tuple] = None) -> Dict:
        """
        Perform Kolmogorov-Smirnov test for distribution fitting.
        
        Args:
            sample: Sample data
            distribution: Distribution name ('norm', 'expon', 'uniform', etc.)
            params: Distribution parameters (if None, estimated from data)
            
        Returns:
            Dictionary with KS statistic and p-value
        """
        if params is None:
            if distribution == 'norm':
                params = (np.mean(sample), np.std(sample, ddof=1))
            elif distribution == 'expon':
                params = (0, np.mean(sample))
            elif distribution == 'uniform':
                params = (np.min(sample), np.max(sample) - np.min(sample))
        
        dist = getattr(stats, distribution)
        ks_stat, p_value = stats.kstest(sample, lambda x: dist.cdf(x, *params))
        
        return {
            'test_type': 'Kolmogorov-Smirnov test',
            'distribution': distribution,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01
        }


class CorrelationAnalysis:
    """Tools for analyzing correlations in experimental data."""
    
    @staticmethod
    def pearson_correlation(x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Compute Pearson correlation coefficient.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            Dictionary with correlation coefficient and p-value
        """
        r, p_value = stats.pearsonr(x, y)
        
        return {
            'correlation_type': 'Pearson',
            'correlation_coefficient': r,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'strength': CorrelationAnalysis._interpret_correlation(r)
        }
    
    @staticmethod
    def spearman_correlation(x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Compute Spearman rank correlation coefficient.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            Dictionary with correlation coefficient and p-value
        """
        rho, p_value = stats.spearmanr(x, y)
        
        return {
            'correlation_type': 'Spearman',
            'correlation_coefficient': rho,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'strength': CorrelationAnalysis._interpret_correlation(rho)
        }
    
    @staticmethod
    def _interpret_correlation(r: float) -> str:
        """Interpret correlation coefficient strength."""
        abs_r = abs(r)
        if abs_r < 0.3:
            return "weak"
        elif abs_r < 0.7:
            return "moderate"
        else:
            return "strong"
    
    @staticmethod
    def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """
        Compute partial correlation between x and y controlling for z.
        
        Args:
            x: First variable
            y: Second variable
            z: Control variable
            
        Returns:
            Partial correlation coefficient
        """
        # Correlation matrix
        r_xy = np.corrcoef(x, y)[0, 1]
        r_xz = np.corrcoef(x, z)[0, 1]
        r_yz = np.corrcoef(y, z)[0, 1]
        
        # Partial correlation formula
        numerator = r_xy - r_xz * r_yz
        denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        
        return numerator / denominator


class ExperimentalDesign:
    """Tools for experimental design and power analysis."""
    
    @staticmethod
    def power_analysis(effect_size: float, alpha: float = 0.05, 
                      power: float = 0.8, test_type: str = 't-test') -> Dict:
        """
        Estimate required sample size for desired statistical power.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            alpha: Significance level
            power: Desired statistical power
            test_type: Type of test ('t-test', 'anova', etc.)
            
        Returns:
            Dictionary with recommended sample size
        """
        from scipy.stats import norm
        
        if test_type == 't-test':
            # Simplified power analysis for t-test
            z_alpha = norm.ppf(1 - alpha/2)
            z_beta = norm.ppf(power)
            n = 2 * ((z_alpha + z_beta) / effect_size)**2
            
            return {
                'test_type': test_type,
                'effect_size': effect_size,
                'alpha': alpha,
                'power': power,
                'recommended_n_per_group': int(np.ceil(n))
            }
        else:
            raise NotImplementedError(f"Power analysis for {test_type} not yet implemented")
    
    @staticmethod
    def confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Dict:
        """
        Calculate confidence interval for mean.
        
        Args:
            data: Sample data
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary with mean and confidence interval
        """
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
        
        return {
            'mean': mean,
            'confidence_level': confidence,
            'confidence_interval': ci,
            'margin_of_error': ci[1] - mean
        }
    
    @staticmethod
    def bootstrap_ci(data: np.ndarray, statistic: callable = np.mean,
                    n_bootstrap: int = 10000, confidence: float = 0.95) -> Dict:
        """
        Calculate bootstrap confidence interval.
        
        Args:
            data: Sample data
            statistic: Function to compute statistic (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            Dictionary with statistic value and confidence interval
        """
        bootstrap_stats = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha/2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha/2))
        
        return {
            'statistic': statistic(data),
            'confidence_level': confidence,
            'confidence_interval': (ci_lower, ci_upper),
            'bootstrap_distribution': bootstrap_stats
        }
