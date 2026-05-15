"""
DynaSys-EEG: A Dynamical System Identification Framework for EEG-based Dementia Classification

This framework reformulates EEG-based dementia classification as a nonlinear
dynamical system identification problem. Disease states (AD, FTD, HC) are
characterized through system-level properties:
  - Lyapunov Exponent (dynamical stability)
  - Sample Entropy (complexity/unpredictability)
  - Diffusion Coefficient (stochastic variability)
  - Energy Landscape (global state organization)
  - Transition Density (state evolution patterns)
"""

__version__ = "1.0.0"
__author__ = "DynaSys-EEG Implementation"

from dynasys_eeg import data, models, features, classification, evaluation, utils
