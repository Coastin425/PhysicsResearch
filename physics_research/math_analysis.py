"""
Mathematical Analysis Module
Provides tools for deep mathematical analysis including numerical methods,
statistical analysis, and Fourier transforms.
"""

import numpy as np
from scipy import integrate, signal, stats
from typing import Callable, Tuple, Optional, Union


class NumericalAnalysis:
    """Numerical analysis tools for physics research."""
    
    @staticmethod
    def differentiate(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
        """
        Compute numerical derivative using central difference method.
        
        Args:
            f: Function to differentiate
            x: Point at which to compute derivative
            h: Step size for numerical differentiation
            
        Returns:
            Numerical derivative at point x
        """
        return (f(x + h) - f(x - h)) / (2 * h)
    
    @staticmethod
    def integrate(f: Callable[[float], float], a: float, b: float, 
                  method: str = 'quad') -> Tuple[float, float]:
        """
        Compute definite integral using numerical integration.
        
        Args:
            f: Function to integrate
            a: Lower bound
            b: Upper bound
            method: Integration method ('quad' or 'simps')
            
        Returns:
            Tuple of (integral_value, estimated_error)
        """
        if method == 'quad':
            return integrate.quad(f, a, b)
        elif method == 'simps':
            x = np.linspace(a, b, 1000)
            y = np.array([f(xi) for xi in x])
            result = integrate.simpson(y, x=x)
            return (result, 0.0)  # Simpson's rule doesn't provide error estimate
        else:
            raise ValueError(f"Unknown integration method: {method}")
    
    @staticmethod
    def solve_ode(f: Callable[[float, np.ndarray], np.ndarray],
                  y0: Union[float, np.ndarray],
                  t_span: Tuple[float, float],
                  t_eval: Optional[np.ndarray] = None) -> dict:
        """
        Solve ordinary differential equation.
        
        Args:
            f: Function defining dy/dt = f(t, y)
            y0: Initial condition(s)
            t_span: Tuple (t_start, t_end)
            t_eval: Times at which to evaluate solution
            
        Returns:
            Dictionary with 't' and 'y' arrays
        """
        from scipy.integrate import solve_ivp
        
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)
        
        sol = solve_ivp(f, t_span, np.atleast_1d(y0), t_eval=t_eval)
        return {'t': sol.t, 'y': sol.y}


class StatisticalAnalysis:
    """Statistical analysis tools for experimental data."""
    
    @staticmethod
    def compute_statistics(data: np.ndarray) -> dict:
        """
        Compute comprehensive statistics for a dataset.
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary containing statistical measures
        """
        return {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data, ddof=1),
            'var': np.var(data, ddof=1),
            'min': np.min(data),
            'max': np.max(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }
    
    @staticmethod
    def correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Compute Pearson correlation coefficient and p-value.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            Tuple of (correlation_coefficient, p_value)
        """
        return stats.pearsonr(x, y)
    
    @staticmethod
    def linear_regression(x: np.ndarray, y: np.ndarray) -> dict:
        """
        Perform linear regression analysis.
        
        Args:
            x: Independent variable
            y: Dependent variable
            
        Returns:
            Dictionary with slope, intercept, r_value, p_value, std_err
        """
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err
        }


class FourierAnalysis:
    """Fourier analysis tools for signal processing."""
    
    @staticmethod
    def fft(signal_data: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Fast Fourier Transform.
        
        Args:
            signal_data: Input signal
            sample_rate: Sampling rate in Hz
            
        Returns:
            Tuple of (frequencies, magnitudes)
        """
        n = len(signal_data)
        fft_vals = np.fft.fft(signal_data)
        fft_freq = np.fft.fftfreq(n, 1/sample_rate)
        
        # Return only positive frequencies
        positive_freq_idx = fft_freq >= 0
        return fft_freq[positive_freq_idx], np.abs(fft_vals[positive_freq_idx])
    
    @staticmethod
    def power_spectrum(signal_data: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density.
        
        Args:
            signal_data: Input signal
            sample_rate: Sampling rate in Hz
            
        Returns:
            Tuple of (frequencies, power_density)
        """
        frequencies, power = signal.periodogram(signal_data, sample_rate)
        return frequencies, power
    
    @staticmethod
    def filter_signal(signal_data: np.ndarray, cutoff: float, 
                     sample_rate: float, filter_type: str = 'low') -> np.ndarray:
        """
        Apply Butterworth filter to signal.
        
        Args:
            signal_data: Input signal
            cutoff: Cutoff frequency in Hz
            sample_rate: Sampling rate in Hz
            filter_type: 'low', 'high', or 'band'
            
        Returns:
            Filtered signal
        """
        nyquist = sample_rate / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype=filter_type, analog=False)
        return signal.filtfilt(b, a, signal_data)
