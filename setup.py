from setuptools import setup, find_packages

setup(
    name="physics_research",
    version="0.1.0",
    description="A comprehensive toolkit for deep physics research",
    author="Physics Research Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "sympy>=1.12",
        "pandas>=2.0.0",
    ],
    python_requires=">=3.8",
)
