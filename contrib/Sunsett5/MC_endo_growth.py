#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 11:58:52 2023

@author: setthakorntanom
"""
# import modules
import math
import numpy as np
import matplotlib.pyplot as plt

# Parameter Values

# Number of Countries
N = 60
# Number of Industries
M = 30
# Number of firms (each industry)
S = 20
# Sectoral demand shares
d_h = 1/30
# Capital-output ratio
B = 3
# Mark-up adjustment parameter
v = 0.04
# R&D investment propensity
rho = 0.04
# R&D allocation parameter
lamb = 0.5
# Firs search capabalities
xi_IN = 0.08
# Firs search capabalities
xi_IM = 0.08
# First stage probabilities upper bound
theta_max = 0.75
# Beta distribution parameter
(alp1, beta1) = (1,5)
# Beta distribution support
(x1_l, x1_u) = (-0.05,0.25)
# Beta distribution parameter (ent.)
(alp2, beta2) = (1,5)
# Beta distribution parameter (ent.)
(x2_l, x2_u) = (-0.03,0.15)
# Foreign imitation penalty
eps = 5
# Foreign competition penalty
tau = 0.05
# Replicator dynamics parameter
chi = 1
# Wage sensitivity parameters
(psi1, psi2, psi3) =(1,0,0)
# Exchange rates flexbilty
gam = 0.1
# Exchange rates shocks std. dev.
sige = 0.002
# Depreciation rate
delta = 0.02
# Monte-Carlo replications
mc_sims = 50
# Time Steps
T = 500

'''def Euc_dist_inv_scalar(x,y):
    
    if x == y: return 0
       
    else: return 1/abs(x-y)'''


def Euc_dist_inv(A,foreign,ind):
    
    Eu = np.divide(1,abs(A-A[ind]), out = np.zeros_like(A), where=A[ind]!=A)
    Eu *= foreign
    
    return Eu

# Initiate Variables

# Sales
SS = np.zeros((T,N,M,S))
SS[0]=1
# R&D expenditure
RD = np.zeros((T,N,M,S))
# Productivity
A = np.zeros((T,N,M,S))
A[0]=1
A_IN = np.zeros((T,N,M,S))
A_IM = np.zeros((T,N,M,S))
# Innovative expenditure and Imitative expenditure
IN = np.zeros((T,N,M,S))
IM = np.zeros((T,N,M,S))
# Success rate of IN, IM
theta_IN = np.zeros((T,N,M,S))
theta_IM = np.zeros((T,N,M,S))

for t in range(1,T):
    SS[t] = SS[t-1]*1.1
    
    # R&D Expenditure
    RD[t] = rho*SS[t-1]
    IN[t] = lamb*RD[t]
    IM[t] = (1-lamb)*RD[t]
    
    # 1st step: determine success rate of IN and IM search
    theta_IN[t] = np.minimum(theta_max, 1 -np.exp(-xi_IN *IN[t]))
    theta_IM[t] = np.minimum(theta_max, 1 -np.exp(-xi_IN *IN[t]))
    
    # 2nd tep: Maximum of A, A_IN,, A_IM
    A_IN[t] = A[t-1]*(1+np.random.binomial(1, theta_IN[t])* \
              (x1_l+np.random.beta(alp1, beta1, (N,M,S)))*(x1_u -x1_l))
        
    for i in N:
        # Augment foreign imitation penalty
        foreign = np.fill(shape = (T,N,S), fill_value = 1/eps)
        foreign[:,i,:] = 1
        
        for h in M:
            for j in S:
                # Euclidian distance
                Eu_inv = Euc_dist_inv(A[t-1,:,h,:], foreign, (t-1,i,h,j))
                A_IM[t,i,h,j] = A[t-1,:,h,:][t-1np.random.multinomial(1, Eu_inv/sum(Eu_inv))
    
    A[t] = np.maximum(A[t-1], A_IN[t])
    
print(A_IN[:,0,0,0])
    