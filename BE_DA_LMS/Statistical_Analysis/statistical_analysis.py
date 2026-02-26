"""
  Module: Statistical Analysis
  Author: Selime Gurol
  Licensing: this code is distributed under the CeCILL-C license
  Copyright (c) 2021 CERFACS
"""

from numpy import zeros, eye, exp, abs
from numpy.linalg import \
(inv,# To invert a matrix
norm)# To compute the Euclidean norm
from scipy.linalg import sqrtm# To compute a square root of a SPD matrix 
from numpy.random import randn # To generate samples from a normalized Gaussian
import matplotlib.pyplot as plt # To plot a grapH


##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (1) Initialization
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n  = 10  # state space dimension 

# Observation operator
I = eye(n)
inds_obs = [2,5,8]
H = I[inds_obs]
m = len(inds_obs) # number of observations

# Observation errors
sigmaR = 0.3 # default 0.3 # observation error std
R = zeros((m,m))
for ii in range(m):
    R[ii,ii] = sigmaR*sigmaR 
 
# Background errors
sigmaB = 0.1 # background error std
L = 1.2 # correlation length scale
btype = 'diagonal'
B = zeros((n,n))
if btype == 'diagonal':
    for ii in range(n):
        B[ii,ii] = sigmaB*sigmaB  
if btype == 'soar':
    for ii in range(n):
        for jj in range(n):
            rij = abs(jj-ii)
            rho = (1 + rij/L)*exp(-rij/L)
            B[ii,jj] = sigmaB*sigmaB*rho          

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (2) Generate the truth
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xt = randn(n) # the true state

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (3) Generate the background
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xb = xt + sqrtm(B).dot(randn(n)) # the background state

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (4) Generate the observations
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y = H.dot(xt) + sqrtm(R).dot(randn(m)) # the observations

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (5) Apply BLUE
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Kalman Gain matrix
K = B.dot(H.T).dot(inv(H.dot(B).dot(H.T) + R))

# Analysis
xa =  xb + K.dot(y - H.dot(xb)) # the analysis state




##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (5) Diagnostics
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print('')
print('||xt - xb||_2 / ||xt||_2 = ', norm(xt - xb)/norm(xt))
print('||xt - xa||_2 / ||xt||_2 = ', norm(xt - xa)/norm(xt))
print('\n')

# Analysis covariance matrix
Sinv = inv(B) + (H.T).dot(inv(R).dot(H))
S = inv(Sinv)
print('Analysis covariance matrix: \n')
for ii in range(n):
    print('S[{}, {}]: {}'.format(ii, ii, S[ii,ii]))

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (6) Plots
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xrange = range(0,n)
fig, (ax_abs, ax_err) = plt.subplots(ncols = 2, figsize = (20,6))
ax_abs.plot(xrange,xt, '-+k', label='Truth', linewidth = 0.5)
ax_abs.plot(xrange, xb, '-db', label='Background', linewidth = 0.5)
ax_abs.plot(inds_obs, y, 'og', label='Observations')
ax_abs.plot(xrange, xa, '-xr', label='Analysis', linewidth = 0.5)
leg = ax_abs.legend()
plt.xlabel('x-coordinate')
plt.ylabel('Temperature')
##
ax_err.plot(xrange, abs(xb - xt), '-db', label='Background error', linewidth = 0.5)
ax_err.plot(inds_obs, abs(y - H.dot(xt)), 'og', label='Observation error')
ax_err.plot(xrange, abs(xa - xt), '-xr', label='Analysis error', linewidth = 0.5)
leg = ax_err.legend()
plt.xlabel('x-coordinate')
plt.ylabel('Temperature')
##
plt.show()






