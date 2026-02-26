#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 10:38:59 2021

@author: destouches

Purpose: Apply localization in a stochastic EnKF setting. 

Licensing: this code is distributed under the CeCILL-C license
  Copyright (c) 2021 CERFACS
"""

import numpy as np
import matplotlib.pyplot as plt


        
def scalar_quasi_gaussian(x, c):
    ''' Classic Gaspari & Cohn compact support localization function.
    The function goes to zero at distance 2c (so support is 4c wide)'''
    z = np.abs(x) / c
    if z <= 1:
        return -1/4 * z**5 + 1/2 * z**4 + 5/8 * z**3 - 5/3 * z**2 + 1
    elif z <= 2:
        return 1/12*z**5 - 1/2*z**4 + 5/8*z**3 + 5/3*z**2 - 5*z +4 - 2/3*1/z
    elif z > 2:
        return 0


def quasi_gaussian(x, c):
    ''' Vectorized version of `scalar_quasi_gaussian`'''
    if np.size(x) == 1:
        return scalar_quasi_gaussian(x, c)
    else:
        dims = x.shape
        x = np.ravel(x)
        out = np.array([scalar_quasi_gaussian(x_, c) for x_ in x])
        return np.reshape(out, dims)


class EnKF_localization:
    ''' Naive implementation of localization in state space, in observation space 
    or in cross-space of observation and state space. '''
    
    def __init__(self, radius, H, Ns, Nobs, periodic=True):
        '''
        Parameters
        ----------
        - radius: int or float. localization radius in grid points. 
                  The localization function is zero at distance `radius`.
        - H: callable. Observation operator.
        - Ns: int. Dimension of state space.
        - Nobs: int. Dimension of observation space. 
        - periodic: bool, optional. Default is True
        '''
        self.radius = radius
        self.obs_position = self.get_position_of_observations(H, Ns, Nobs)
        self.state_position = np.arange(Ns)
        self.Ns = Ns
        self.Nobs = Nobs

        x1, x2 = np.meshgrid(self.state_position, self.state_position)
        self.distance_xx = np.abs(x1 - x2)
        y1, y2 = np.meshgrid(self.obs_position, self.obs_position)
        self.distance_yy = np.abs(y1 - y2)
        x1, y2 = np.meshgrid(self.obs_position, self.state_position)
        self.distance_xy = np.abs(x1 - y2)

        # Account for periodic boundary conditions:
        self.periodic = periodic
        if self.periodic:
            self.distance_xx = np.minimum(self.distance_xx , self.Ns - self.distance_xx)
            self.distance_yy = np.minimum(self.distance_yy , self.Ns - self.distance_yy)
            self.distance_xy = np.minimum(self.distance_xy , self.Ns - self.distance_xy)
        
        
    def get_position_of_observations(self, H, Ns, Nobs):
        '''
        From the observation operator H from state space of size Ns to observation
        space of size Nobs, 
        retrieves the spatial position of observations as an array of size Nobs.
        Positions are numbered as python indexes, from 0 to Ns -1.    '''
        # Initialize position vector
        positions = - np.ones(Nobs)
        for dirac_position in range(Ns):
            # Create a dirac vector
            dirac = np.zeros(Ns)
            dirac[dirac_position] = 1
            # Apply observation operator
            obs = H(dirac)
            # Test sensitivity to this position
            if np.any(obs):
                # Observation operator is sensitive to position dirac_position.
                # Find number of sensitive observation(s).
                if Nobs == 1:
                    sensitive = [0]
                else:
                    sensitive = np.argwhere(obs)
                for sensitive_obs in sensitive:
                    if positions[sensitive_obs] != -1:
                        # This position has already been visited.
                        print('Warning! Non local observation operator')
                        return 1
                    else:
                        positions[sensitive_obs] = dirac_position
        return positions


    def localize_HBHt(self, HBHt, plot=False):
        ''' Apply localization in observation space to matrix 
        HBHt. 
        Parameters:
        ----------
        - HBHt: np array or np matrix. Matrix to localize.
        - plot: bool, optional. Plot effect of localization. Default is False.
        Returns:
        -------
        np.array: localized matrix the size of HBHt. 
        '''
        loc_matrix = quasi_gaussian(self.distance_yy, self.radius/2)
        HBHt_loc = loc_matrix * HBHt
        
        if plot:
            fig, axs = plt.subplots(ncols=2)
            axs = axs.ravel()
            im = axs[0].imshow(HBHt)
            plt.colorbar(im, ax=axs[0], orientation='horizontal', location='bottom')
            axs[0].set_title('Not localized')
            im = axs[1].imshow(HBHt_loc)
            plt.colorbar(im, ax=axs[1], orientation='horizontal', location='bottom')
            axs[1].set_title(f'localized, radius = {self.radius} grid points')
            for ax in axs:
                ax.set_ylabel('Observation space')
                ax.set_xlabel('Observation space')
            fig.tight_layout()
            plt.show(fig)
            
        return HBHt_loc


    def localize_BHt(self, BHt, plot=False):
        '''
        Parameters:
        ----------
        - BHt: np array or np matrix. Matrix to localize in cross-space of state
               and observations.
        - plot: bool, optional. Plot effect of localization. Default is False.
        Returns:
        -------
        np.array: localized matrix the size of BHt. 
        '''        
        loc_matrix = quasi_gaussian(self.distance_xy, self.radius/2)
        BHt_loc = loc_matrix * BHt
        
        if plot:
            fig, axs = plt.subplots(ncols=2, figsize=(5, 5))
            axs = axs.ravel()
            im = axs[0].imshow(BHt)
            plt.colorbar(im, ax=axs[0], orientation='horizontal', location='bottom')
            axs[0].set_title('Not localized')
    
            im = axs[1].imshow(BHt_loc)
            plt.colorbar(im, ax=axs[1], orientation='horizontal', location='bottom')
            axs[1].set_title(f'localized, radius = {self.radius} grid points')
            for ax in axs:
                ax.set_ylabel('State space')
                ax.set_xlabel('Observation space')
            fig.tight_layout()
            plt.show(fig)
        
        return BHt_loc


    def localize_B(self, B, plot=False):
        '''
        Parameters:
        ----------
        - B: np array or np matrix. Matrix to localize in observation space.
        - plot: bool, optional. Plot effect of localization. Default is False.
        Returns:
        -------
        np.array: localized matrix the size of B. 
        '''
        loc_matrix = quasi_gaussian(self.distance_xx, self.radius/2)
        B_loc = B * loc_matrix
        
        if plot:
            fig, axs = plt.subplots(ncols=2)
            axs = axs.ravel()
            im = axs[0].imshow(B)
            plt.colorbar(im, ax=axs[0], orientation='horizontal', location='bottom')
            axs[0].set_title('Not localized')
    
            im = axs[1].imshow(B_loc)
            plt.colorbar(im, ax=axs[1], orientation='horizontal', location='bottom')
            axs[1].set_title(f'Localized, radius = {self.radius} grid points')
            for ax in axs:
                ax.set_ylabel('State space')
                ax.set_xlabel('State space')
            fig.tight_layout()
            plt.show(fig)
        
        return B_loc
        

    
if __name__ == '__main__':
    # Just some tests to make sure everything is OK. 
    
    # State space dimension
    Ns = 40
    # Test localization radius
    radius = 10
    
# =============================================================================
#     Test method get_position_of_observations
# =============================================================================
    # Test observation operator #1
    H1 = lambda x: x
    loc = EnKF_localization(radius, H1, Ns, Ns)
    assert np.all(loc.obs_position == np.arange(40))

    # Test observation operator #2
    H2 = lambda x: x[::2]
    loc = EnKF_localization(radius, H2, Ns, Ns//2)
    assert np.all(loc.obs_position == np.arange(0, 40, 2))

    # Test observation operator #3
    positions = np.array([0, 0, 9, 3, 4])
    H3 = lambda x: x[positions]
    loc = EnKF_localization(radius, H3, Ns, len(positions))
    assert np.all(loc.obs_position == positions)
    
    # Test non-local observation operator
    H4 = lambda x: x[0] + x[1]
    print('This should print "Warning! Non local observation operator":')
    loc = EnKF_localization(radius, H4, Ns, 1)
    assert loc.obs_position == 1
    
    # Test simple observation operator
    H5 = lambda x: x[7]
    loc = EnKF_localization(radius, H5, Ns, 1)
    assert loc.obs_position == np.array([7])


#%%============================================================================
#     Plot GC function
# =============================================================================
    x = np.linspace(-25, 25, 100)
    plt.figure()
    plt.plot(x, quasi_gaussian(x, c=radius/2))

    
#%%============================================================================
#     Plot localized covariance matrices (state space)
# =============================================================================
    rng = np.random.default_rng(1234)
    
    H = lambda x: x
    loc = EnKF_localization(radius, H, Ns, Ns)
    
    for Ne in (10, 20, 1000):
        E = rng.standard_normal((Ns, Ne))
        X = E - np.outer(np.mean(E, axis=1), np.ones(Ne))
        Bens = 1/(Ne-1) * X @ X.T
        Bloc = loc.localize_B(Bens, plot=True)

#%%============================================================================
#     Plot localized covariance matrices (observation - state space)
# =============================================================================
    rng = np.random.default_rng(1234)
    
    H = lambda x: x[::2]
    loc = EnKF_localization(radius, H, Ns, Ns//2)
    
    for Ne in (10, 20, 1000):
        E = rng.standard_normal((Ns, Ne))
        Eobs = H(E)
        X = E - np.outer(np.mean(E, axis=1), np.ones(Ne))
        Y = Eobs - np.outer(np.mean(Eobs, axis=1), np.ones(Ne))
        BHtens = 1/(Ne-1) * X @ (H(X)).T
        BHtloc = loc.localize_BHt(BHtens, plot=True)