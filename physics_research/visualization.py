"""
Visualization Module
Provides tools for visualizing research data and analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Tuple, Union, List


class DataVisualization:
    """Visualization tools for experimental data."""
    
    @staticmethod
    def plot_time_series(t: np.ndarray, y: Union[np.ndarray, List[np.ndarray]],
                        labels: Optional[List[str]] = None,
                        title: str = "Time Series",
                        xlabel: str = "Time",
                        ylabel: str = "Value",
                        figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Plot time series data.
        
        Args:
            t: Time array
            y: Data array or list of data arrays
            labels: Labels for each series
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if isinstance(y, list):
            for i, yi in enumerate(y):
                label = labels[i] if labels else f"Series {i+1}"
                ax.plot(t, yi, label=label)
            ax.legend()
        else:
            ax.plot(t, y)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_scatter(x: np.ndarray, y: np.ndarray,
                    title: str = "Scatter Plot",
                    xlabel: str = "X",
                    ylabel: str = "Y",
                    fit_line: bool = False,
                    figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Create scatter plot with optional linear fit.
        
        Args:
            x: X data
            y: Y data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            fit_line: Whether to add linear regression line
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.scatter(x, y, alpha=0.6)
        
        if fit_line:
            from scipy.stats import linregress
            slope, intercept, r_value, _, _ = linregress(x, y)
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, 'r--', 
                   label=f'Fit: y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}')
            ax.legend()
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_histogram(data: np.ndarray,
                      bins: Union[int, str] = 'auto',
                      title: str = "Histogram",
                      xlabel: str = "Value",
                      ylabel: str = "Frequency",
                      density: bool = False,
                      figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Create histogram with optional density overlay.
        
        Args:
            data: Data array
            bins: Number of bins or binning strategy
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            density: Whether to normalize to probability density
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.hist(data, bins=bins, density=density, alpha=0.7, edgecolor='black')
        
        if density:
            # Add normal distribution overlay
            mu, sigma = np.mean(data), np.std(data)
            x = np.linspace(data.min(), data.max(), 100)
            ax.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu)/sigma)**2),
                   'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
            ax.legend()
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        return fig
    
    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                      x: Optional[np.ndarray] = None,
                      title: str = "Residual Plot",
                      figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Create residual plot for model validation.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            x: Optional x-values for residual vs x plot
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Residuals vs predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1].hist(residuals, bins='auto', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_spectrum(frequencies: np.ndarray, magnitudes: np.ndarray,
                     title: str = "Frequency Spectrum",
                     xlabel: str = "Frequency (Hz)",
                     ylabel: str = "Magnitude",
                     log_scale: bool = False,
                     figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Plot frequency spectrum.
        
        Args:
            frequencies: Frequency array
            magnitudes: Magnitude array
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            log_scale: Whether to use log scale for y-axis
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(frequencies, magnitudes)
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_phase_space(x: np.ndarray, v: np.ndarray,
                        title: str = "Phase Space",
                        xlabel: str = "Position",
                        ylabel: str = "Velocity",
                        figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Create phase space plot.
        
        Args:
            x: Position array
            v: Velocity array
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(x, v, linewidth=1.5)
        ax.scatter(x[0], v[0], c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(x[-1], v[-1], c='red', s=100, marker='s', label='End', zorder=5)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_contour(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    title: str = "Contour Plot",
                    xlabel: str = "X",
                    ylabel: str = "Y",
                    levels: int = 20,
                    figsize: Tuple[int, int] = (10, 8)) -> Figure:
        """
        Create contour plot for 2D data.
        
        Args:
            x: X coordinates
            y: Y coordinates
            z: Z values
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            levels: Number of contour levels
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        contour = ax.contourf(x, y, z, levels=levels, cmap='viridis')
        fig.colorbar(contour, ax=ax)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        return fig
    
    @staticmethod
    def save_figure(fig: Figure, filename: str, dpi: int = 300) -> None:
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib figure object
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
