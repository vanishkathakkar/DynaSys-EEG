from setuptools import setup, find_packages

setup(
    name="dynasys-eeg",
    version="1.0.0",
    description="DynaSys-EEG: Dynamical System-based EEG Dementia Classification",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "torch>=1.10.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "nolds>=0.5.2",
        "antropy>=0.1.6",
    ],
)
