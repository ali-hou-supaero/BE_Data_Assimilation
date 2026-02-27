#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to perform twin experiments with the EnKF data assimilation algorithm.
The system state evolution is simulated using a Lorenz-95 model.
'Twin experiments' means that the observations are generated using the same
model as the one used for forecasting in the data assimlation framework.

Data assimilation formation at Cerfacs 2023.

Programming: Anthony Fillion, 2017
             Mayeul Destouches, 2021
             Eliott Lumet, 2023

Licensing: this code is distributed under the CeCILL-C license
  Copyright (c) 2021 CERFACS             
"""
# =============================================================================
# Imports
# =============================================================================
# import sys
# sys.path.append('../Model')

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
# The model used to forecast the evolution of the system
from models import lorenz95
# A localization method to improve the EnKF algorithm
from localization import EnKF_localization

# =============================================================================
# Inputs
# =============================================================================
# Model parameters
dt_model = 0.025   # Time stepping for Runge-Kutta
F = 8              # Forcing term
Ns = 40            # State space dimension
exp_length = 10.0   # Experiment duration (time unit)

# Data assimilation parameters
sigmaB = 2.0       # Initial background error std
sigmaR = 1.0       # Observation error std
Ne = 100           # Number of members
Ndt = 4            # Number of model integration between two data assimilations
# The period between two assimilations is thus dt_ad = Ndt*dt_model
Nc = int(exp_length/(Ndt*dt_model))  # Number of cycles

# Observation operator parameters
# By default, we observe one state variable out of 2.
obs_spacing = 2
# Array of observed state coordinates.
obs_indexes = np.arange(0, Ns, obs_spacing)

# Advanced EnKF settings
# Boolean to decide if inflation should be used in the EnKF algorithm
apply_inflation = False
# Boolean to decide if localization should be used in the EnKF algorithm
apply_localization = False
# Inflation coefficient used to increase the background spread
inflation = 1.2
# The localization function is zero for distance larger than radius (in grid points)
localization_radius = 10

# Random sampling parameters
seed = 31415                       # Seed for reproducibility
rng = np.random.default_rng(seed)  # Initialize random number generator
# To draw samples from a normalized normal law, use
# rng.standard_normal()

# Plotting parameters
plot_component = 30                   # State component to track/show
plot_timestep = 5                     # Timestep to show
# Boolean to plot the truth trajectory based on the Lorenz-95 model
plot_model_trajectory = False
# Boolean to plot all states coordinates and their DA estimation at the i-th timestep
plot_state_variables = True
# Boolean to plot the evolution of the truth, background and analysis of the j-th state component
plot_data_assimilation_cycles = True
# Boolean to plot the evolution of the background ensemble spread for the j-th state component
plot_local_ensemble_spread = True
# Boolean to plot the evolution of the Root Mean Square Error (RMSE) of the state estimation
plot_error_evolution = True
# Boolean to plot the last bht and hbht matrices
plot_bht_and_hbht_matrices = False

# =============================================================================
# Observation operator
# =============================================================================


def H(x):
    """
    Observation operator

    Maps the state to the observation space. For example, in NWP state 
    variables are atmospheric thermodynamic variables such as temperature, 
    pressure, velocity while observations can be radiances seen by satellites.
    In this example, the state is directly observed which means that H is 
    linear. Note that it is not always the case.
    The state variables that are observed are defined in the vector 
    'obs_indexes' defined out of the function scope. By default all the 
    state components are observed.

    Parameters
    ----------
    x : np.array (size Ns)
       System state vector.

    Returns
    -------
    y : np.array (size No=len(obs_indexes))
        The interpolated fields.
    """
    global obs_indexes
    return x[obs_indexes]


No = np.size(obs_indexes)  # Observation space dimension
# The obs error covariance is assumed to be diagonal
R = (sigmaR**2)*np.eye(No)

# =============================================================================
# Initializations
# =============================================================================
model = lorenz95(F, dt_model)         # Model definiton

# Vectors definition
xt = np.zeros((Nc, Ns))               # Truth states vector
# Background vector (mean of the background ensemble)
xb_mean = np.zeros((Nc, Ns))
std_xb = np.zeros((Nc, Ns))           # Background spread vector
Eb = np.zeros((Ns, Ne))               # Background ensemble vector
# Background ensemble vector in the observation space
Z = np.zeros((No, Ne))
y = np.zeros((Nc, No))                # Observations vector
# Analysis vector (mean of the analysis ensemble)
xa_mean = np.zeros((Nc, Ns))
t = np.array([k*Ndt*dt_model for k in range(Nc)])  # Assimilation times vector

# Some algebra
I = np.eye(No)                       # Observation space identity matrix

if apply_localization:
    localization = EnKF_localization(localization_radius, H, Ns, No)

# =============================================================================
# Twin experiment setup: truth state trajectory and observations generation
# =============================================================================
xt_init = 3.0 * np.ones(Ns) + rng.standard_normal(Ns)   # Initial true state
tspinup = 100                                           # Model spin-up
xt[0] = model.traj(xt_init, tspinup)

# True state trajectory generation
for i in range(Nc-1):
    # Model integration over the whole time window
    xt[i+1] = model.traj(xt[i], Ndt)

# Generate the observations from the truth state trajectory
for i in range(Nc):
    # A gaussian error is added to mimic sensor error
    y[i] = H(xt[i]) + sigmaR * rng.standard_normal(No)

# =============================================================================
# First background ensemble generation
# =============================================================================
# A gaussian error is added to the truth state to define the initial prior state estimation
xb0 = xt[0] + sigmaB * rng.standard_normal(Ns)
# The ensemble is then sampled using xb and sigmaB
Eb = xb0[:, np.newaxis] + sigmaB * rng.standard_normal((Ns, Ne))

# =============================================================================
# EnKF algorithm: TO COMPLETE
# =============================================================================
for k in range(Nc):  # Loop over the cycles
    print(f"\r{int(100*k/(Nc-1))}%", end=" ")  # progression counter
    # Analysis
    # Compute the mean and anomaly of the current ensemble
    # Use np.mean(., axis=1) for averaging over the second dimension.
    # Use broadcasting with np.new_axis to add arrays of different sizes.
    xb_mean[k] = np.mean(Eb, axis=1)
    Ab = Eb - xb_mean[k][:, np.newaxis]
    # Storage for visualization
    # Save the spread of the background ensemble
    std_xb[k] = np.std(Eb, axis=1)

    # Map the ensemble members to the observation space using the observation operator H
    for i in range(Ne):
        Z[:, i] = H(Eb[:, i])

   # Compute Z_mean the mean of H(xb) and A0 its anomaly (Ao = [Z_1 - Z_mean, Z_2 - Z_mean, ..., Z_Ne - Z_mean])
    Z_mean = np.mean(Z, axis=1)
    Ao = Z - Z_mean[:, np.newaxis]

    # Compute the Kalman Gain using covariance estimation
    # K = BH^T(HBH^T+R)^(-1)
    # With: HBH^T = Var(H(xb))
    #       BH^T = Cov(xb, H(xb))
    # Use np.dot(A, B.T) to multiply the matrix A with the matrix B transpose
    # Use np.linalg.inv(A) to compute A^(-1)
    # Use empirical covariance estimation with anomalies matrixes Ab and Ao (see tutorial notes)
    BHt = np.dot(Ab, Ao.T) / (Ne - 1)
    HBHt = np.dot(Ao, Ao.T) / (Ne - 1) + R

    if apply_localization:
        # To plot every n cycles only: plot=(k%n == 0)
        HBHt = localization.localize_HBHt(HBHt, plot=False)
        BHt = localization.localize_BHt(BHt, plot=False)

    # Kalmain gain estimation
    K = np.dot(BHt, np.linalg.inv(HBHt))

    # Produce an ensemble of perturbed observations from the real obs y
    Y = y[k][:, np.newaxis] + sigmaR * rng.standard_normal((No, Ne))

    # Compute the analysis of each ensemble members
    Ea = Eb + K @ (Y - Z)
    xa_mean[k] = np.mean(Ea, axis=1)

    if apply_inflation:
        Ea = xa_mean[k][:, np.newaxis] + inflation * \
            (Ea - xa_mean[k][:, np.newaxis])

    # Forecast
    if k < Nc-1:  # If not the last cycle
        for i in range(Ne):
            # For each ensemble forecast the system evolution with M(., Ndt) for the next cycle
            # This part can be easily parallelized
            Eb[:, i] = model.traj(Ea[:, i], Ndt)

# =============================================================================
# End of EnKF algorithm (nothing to complete afterwards)
# =============================================================================
# Error computations
rmse_b = np.zeros(Nc)
rmse_a = np.zeros(Nc)
for k in range(Nc):
    rmse_b[k] = np.linalg.norm(xb_mean[k] - xt[k]) / Ns**0.5
    rmse_a[k] = np.linalg.norm(xa_mean[k] - xt[k]) / Ns**0.5

print('\n\nGlobal RMSE (error between the analysis and the truth state averaged over time): {:.3f}'.format(
    (np.mean(rmse_a**2))**0.5))

# =============================================================================
# Plotting
# =============================================================================
if plot_model_trajectory:
    fig, ax = plt.subplots(figsize=(8, 4))
    fieldmin = min(np.min(xt), -np.max(xt))
    fieldmax = max(-np.min(xt), np.max(xt))
    levels = np.linspace(fieldmin, fieldmax, 100)
    im = ax.contourf(np.arange(1, Ndt*Nc + 1, Ndt) / (1/dt_model),
                     np.arange(1, Ns + 1),
                     xt.T, cmap='coolwarm', levels=levels)
    plt.colorbar(im, ticks=np.linspace(fieldmin, fieldmax, 6),
                 label=r'$\mathrm{x}$')
    ax.set_title('Truth state trajectory (Lorenz-95)')
    ax.set_xlabel('Time unit')
    ax.set_ylabel('Space index')
    plt.show()

if plot_state_variables:
    fig, ax = plt.subplots(figsize=(6, 4))
    indexes = np.arange(1, Ns + 1)
    ax.scatter(indexes, xt[1], c='C0', marker="*", label="Truth")
    ax.scatter(obs_indexes + 1, y[1], c='C1', marker='+', label="Observations")
    ax.scatter(indexes, xb_mean[1], c='C2', marker='x', label="Background")
    ax.scatter(indexes, xa_mean[1], c='C3', marker='.', label="Analysis")
    # Add interpolation for the truth:
    f = interp1d(indexes, xt[1], kind='linear')
    points = np.linspace(1, Ns, 1001)
    ax.plot(points, f(points), c='C0', linestyle='--', linewidth=1.)
    ax.set_title(f'Data assimilation variables at timestep = {plot_timestep}')
    ax.set_ylabel(r'$\mathrm{x}$')
    ax.set_xlabel(r'State component ($j$)')
    ax.legend(loc='best')
    plt.show()

if plot_data_assimilation_cycles:
    fig, ax = plt.subplots(figsize=(10, 6))
    # The truth state trajectory is saved with an increased time resolution
    t_hq = np.array([k*dt_model for k in range(Ndt*(Nc-1))])
    xt_hq = np.zeros((Ndt*(Nc-1), Ns))
    xt_hq[0] = xt[0]
    for k in range(Ndt*(Nc-1)-1):
        xt_hq[k+1] = model.traj(xt_hq[k], 1)

    ax.plot(t_hq, xt_hq[:, plot_component], c='C0', linestyle='--', linewidth=1.,
            label="Truth trajectory")
    ax.scatter(t, xt[:, plot_component], s=80,
               c='C0', marker='*', label="Truth")
    if plot_component in obs_indexes:  # Observations are not plotted if the j-th component is not observed
        ax.scatter(t, y[:, np.argwhere(obs_indexes == plot_component)], s=80,
                   c='C1', marker='+', label="Observations")
    ax.scatter(t, xb_mean[:, plot_component], s=80, c='C2', marker='x',
               label="Backgound")
    ax.scatter(t, xa_mean[:, plot_component], s=80,
               c='C3', marker='.', label="Analysis")
    ax.set_title('Data assimilation variables evolution')
    ax.set_ylabel(r'$\mathrm{x}$['+str(plot_component)+']')
    ax.set_xlabel('Time')
    ax.legend(loc='best')
    plt.show()

if plot_local_ensemble_spread:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, std_xb[:, plot_component], marker='.',
            label=r'Ensemble spread $\sigma(X_b[$'+str(plot_component)+r'$])$')
    ax.set_title('Evolution of the local ensemble spread')
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\sigma(X_b[$'+str(plot_component)+r'$])$')
    ax.legend()
    plt.show()

if plot_error_evolution:
    fig, ax = plt.subplots(figsize=(6, 4))
    cycles = np.array([k+1 for k in range(Nc)])
    ax.plot(cycles, rmse_b, c='C2', marker='x', label='Background')
    ax.plot(cycles, rmse_a, c='C3', marker='.', label='Analysis')
    ax.set_title('Evolution of the error in comparison to the truth state')
    ax.set_xlim(1, Nc)
    ax.set_xlabel('Cycles')
    ax.set_ylabel('RMSE')
    ax.legend()
    plt.show()

if plot_bht_and_hbht_matrices:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    vmin = min(np.min(HBHt), np.min(BHt))
    vmax = max(np.max(HBHt), np.max(BHt))
    axes[0].imshow(HBHt, vmin=vmin, vmax=vmax, cmap='Spectral')
    im = axes[1].imshow(BHt, vmin=vmin, vmax=vmax, cmap='Spectral')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    axes[0].set_title(r'$\mathbf{HBH}^{\mathrm{T}}$' +
                      ' matrix'+f'\n(last value)', fontsize=11.0)
    axes[1].set_title(r'$\mathbf{BH}^{\mathrm{T}}$' +
                      ' matrix'+f'\n(last value)', fontsize=11.0)
    plt.show()

print("\nScript execution completed: Successful run")
