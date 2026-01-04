"""
Mathematical Modeling Module
Provides tools for creating and analyzing mathematical models of physical systems.
"""

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, Optional, Union, Dict
import sympy as sp


class PhysicsModels:
    """Common physics models and equations."""
    
    @staticmethod
    def harmonic_oscillator(t: np.ndarray, omega: float, A: float, phi: float = 0) -> np.ndarray:
        """
        Simple harmonic oscillator model: x(t) = A*cos(omega*t + phi)
        
        Args:
            t: Time array
            omega: Angular frequency
            A: Amplitude
            phi: Phase shift
            
        Returns:
            Position array
        """
        return A * np.cos(omega * t + phi)
    
    @staticmethod
    def damped_oscillator(t: np.ndarray, omega0: float, gamma: float, 
                         A: float, phi: float = 0) -> np.ndarray:
        """
        Damped harmonic oscillator: x(t) = A*exp(-gamma*t)*cos(omega*t + phi)
        
        Args:
            t: Time array
            omega0: Natural frequency
            gamma: Damping coefficient
            A: Initial amplitude
            phi: Phase shift
            
        Returns:
            Position array
        """
        omega = np.sqrt(omega0**2 - gamma**2)
        return A * np.exp(-gamma * t) * np.cos(omega * t + phi)
    
    @staticmethod
    def exponential_decay(t: np.ndarray, N0: float, lambda_decay: float) -> np.ndarray:
        """
        Exponential decay model: N(t) = N0*exp(-lambda*t)
        
        Args:
            t: Time array
            N0: Initial quantity
            lambda_decay: Decay constant
            
        Returns:
            Quantity array
        """
        return N0 * np.exp(-lambda_decay * t)
    
    @staticmethod
    def gaussian(x: np.ndarray, mu: float, sigma: float, A: float = 1.0) -> np.ndarray:
        """
        Gaussian distribution: f(x) = A*exp(-(x-mu)^2/(2*sigma^2))
        
        Args:
            x: Input array
            mu: Mean
            sigma: Standard deviation
            A: Amplitude
            
        Returns:
            Gaussian values
        """
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


class ParameterFitting:
    """Tools for fitting model parameters to experimental data."""
    
    @staticmethod
    def fit_model(model: Callable, x_data: np.ndarray, y_data: np.ndarray,
                  initial_guess: Optional[list] = None,
                  bounds: Tuple = (-np.inf, np.inf)) -> Dict:
        """
        Fit a model to data using non-linear least squares.
        
        Args:
            model: Model function to fit
            x_data: Independent variable data
            y_data: Dependent variable data
            initial_guess: Initial parameter guesses
            bounds: Parameter bounds (lower, upper)
            
        Returns:
            Dictionary with fitted parameters, covariance, and statistics
        """
        popt, pcov = curve_fit(model, x_data, y_data, p0=initial_guess, bounds=bounds)
        
        # Calculate R-squared
        y_pred = model(x_data, *popt)
        ss_res = np.sum((y_data - y_pred)**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'parameters': popt,
            'covariance': pcov,
            'std_errors': np.sqrt(np.diag(pcov)),
            'r_squared': r_squared,
            'residuals': y_data - y_pred
        }
    
    @staticmethod
    def chi_squared(observed: np.ndarray, expected: np.ndarray, 
                   uncertainties: Optional[np.ndarray] = None) -> float:
        """
        Calculate chi-squared statistic.
        
        Args:
            observed: Observed data
            expected: Expected/predicted data
            uncertainties: Uncertainties in observed data
            
        Returns:
            Chi-squared value
        """
        if uncertainties is None:
            uncertainties = np.ones_like(observed)
        
        return np.sum(((observed - expected) / uncertainties)**2)


class DifferentialEquations:
    """Tools for solving differential equations in physics."""
    
    @staticmethod
    def solve_system(equations: Callable[[float, np.ndarray], np.ndarray],
                    initial_conditions: np.ndarray,
                    time_span: Tuple[float, float],
                    num_points: int = 1000) -> Dict[str, np.ndarray]:
        """
        Solve a system of ordinary differential equations.
        
        Args:
            equations: Function defining the system dy/dt = f(t, y)
            initial_conditions: Initial values for all variables
            time_span: Tuple (t_start, t_end)
            num_points: Number of time points to evaluate
            
        Returns:
            Dictionary with 't' and 'y' arrays
        """
        t_eval = np.linspace(time_span[0], time_span[1], num_points)
        sol = solve_ivp(equations, time_span, initial_conditions, 
                       t_eval=t_eval, method='RK45')
        return {'t': sol.t, 'y': sol.y}
    
    @staticmethod
    def pendulum(t: float, state: np.ndarray, g: float = 9.81, L: float = 1.0) -> np.ndarray:
        """
        Simple pendulum equations: theta'' + (g/L)*sin(theta) = 0
        
        Args:
            t: Time
            state: [theta, omega] where omega = theta'
            g: Gravitational acceleration
            L: Pendulum length
            
        Returns:
            Derivative [theta', omega']
        """
        theta, omega = state
        dtheta_dt = omega
        domega_dt = -(g / L) * np.sin(theta)
        return np.array([dtheta_dt, domega_dt])
    
    @staticmethod
    def damped_driven_oscillator(t: float, state: np.ndarray, 
                                 omega0: float, gamma: float,
                                 F0: float, omega_d: float) -> np.ndarray:
        """
        Damped driven harmonic oscillator equations.
        
        Args:
            t: Time
            state: [x, v] position and velocity
            omega0: Natural frequency
            gamma: Damping coefficient
            F0: Driving force amplitude
            omega_d: Driving frequency
            
        Returns:
            Derivative [v, a]
        """
        x, v = state
        dx_dt = v
        dv_dt = -omega0**2 * x - 2 * gamma * v + F0 * np.cos(omega_d * t)
        return np.array([dx_dt, dv_dt])


class SymbolicAnalysis:
    """Symbolic mathematics for theoretical analysis."""
    
    @staticmethod
    def create_symbols(symbol_string: str) -> Union[sp.Symbol, Tuple[sp.Symbol, ...]]:
        """
        Create symbolic variables.
        
        Args:
            symbol_string: Space-separated string of symbol names
            
        Returns:
            Symbolic variable(s)
        """
        return sp.symbols(symbol_string)
    
    @staticmethod
    def differentiate(expression: sp.Expr, variable: sp.Symbol, order: int = 1) -> sp.Expr:
        """
        Compute symbolic derivative.
        
        Args:
            expression: Symbolic expression
            variable: Variable to differentiate with respect to
            order: Order of differentiation
            
        Returns:
            Derivative expression
        """
        return sp.diff(expression, variable, order)
    
    @staticmethod
    def integrate(expression: sp.Expr, variable: sp.Symbol, 
                 limits: Optional[Tuple] = None) -> sp.Expr:
        """
        Compute symbolic integral.
        
        Args:
            expression: Symbolic expression
            variable: Integration variable
            limits: Optional integration limits (a, b)
            
        Returns:
            Integral expression
        """
        if limits is None:
            return sp.integrate(expression, variable)
        else:
            return sp.integrate(expression, (variable, limits[0], limits[1]))
    
    @staticmethod
    def taylor_series(expression: sp.Expr, variable: sp.Symbol, 
                     point: float, order: int) -> sp.Expr:
        """
        Compute Taylor series expansion.
        
        Args:
            expression: Expression to expand
            variable: Expansion variable
            point: Point around which to expand
            order: Order of expansion
            
        Returns:
            Taylor series expression
        """
        return expression.series(variable, point, order).removeO()
