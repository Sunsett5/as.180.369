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
    
    
    Eu = np.divide(1,abs(A-A[ind]), out = np.full(np.shape(A),-1, dtype='float64'), where=A[ind]!=A)
    
    # Prevent ZeroDivisionError by setting the result equal to the maximum value
    Eu[Eu == -1] = np.max(Eu)
    Eu[ind] = 0
    
    Eu *= foreign
    
    return Eu

# Initiate Variables

# Sales
SS = np.zeros((T,M,N,S))
SS[0]=10
# R&D expenditure
RD = np.zeros((T,M,N,S))
# Productivity
A = np.zeros((T,M,N,S))
A[0]=1
A_IN = np.zeros((T,M,N,S))
A_IM = np.zeros((T,M,N,S))
# Innovative expenditure and Imitative expenditure
IN = np.zeros((T,M,N,S))
IM = np.zeros((T,M,N,S))
# Success rate of IN, IM
theta_IN = np.zeros((T,M,N,S))
theta_IM = np.zeros((T,M,N,S))
# Market share
f = np.zeros((T,M,N,S))
f = np.full((T,M,N,S), 1/S) #check!!
# Capital Stock
K = np.zeros((T,M,N,S))
K[0:2]=1
# Desired capital stock
Kd = np.zeros((T,M,N,S))
# Price of consumer goods
p = np.zeros((T,M,N,S))
# Mark up price
m = np.zeros((T,M,N,S))
m = np.full((T,M,N,S), 1.2) #remove
# Nominal wage
W = np.zeros((T,M,N,S))
W = np.full((T,M,N,S), 1)
# Desired Production
Qd = np.zeros((T,M,N,S))
# Actual Production
Q = np.zeros((T,M,N,S))
# Demand
D = np.zeros((T,M,N,S))
D = np.full((T,M,N,S), 1.2) #remove
# Expansion Investment
Ie = np.zeros((T,M,N,S))
# Replacement Investment
Ir = np.zeros((T,M,N,S))
# Total domestic production of capital sector
Qk = np.zeros((T,N))
# Labor employed in capital sector
Lk = np.zeros((T,N))

for t in range(1,T):
    
    # Provisionary dynamics
    SS[t] = SS[t-1]*1.1
    
    # R&D Expenditure
    RD[t] = rho*SS[t-1]
    IN[t] = lamb*RD[t]
    IM[t] = (1-lamb)*RD[t]
    
    # 1st step: determine success rate of IN and IM search
    theta_IN[t] = np.minimum(theta_max, 1 -np.exp(-xi_IN *IN[t]))
    theta_IM[t] = np.minimum(theta_max, 1 -np.exp(-xi_IN *IN[t]))
    
    # Success of imitation
    IM_success = np.random.binomial(1, theta_IM[t])
    
    # 2nd tep: Maximum of A, A_IN,, A_IM
    A_IN[t] = A[t-1]*(1+np.random.binomial(1, theta_IN[t])* \
              (x1_l+np.random.beta(alp1, beta1, (M,N,S)))*(x1_u -x1_l))
        
    for i in range(N):
        # Augment foreign imitation penalty
        foreign = np.full(shape = (N,S), fill_value = 1/eps)
        foreign[i,:] = 1
        
        for h in range(M):
            for j in range(S):
                
                # Find target firm if imitation is successful
                if IM_success[h,i,j] == 1:                        
                
                    # Euclidian distance
                    Eu_inv = Euc_dist_inv(A[t-1,h,:,:], foreign, (i,j))
                    
                    # Multinomial Result
                    mult = np.random.multinomial(1, np.reshape(Eu_inv/np.sum(Eu_inv),N*S))
                    (i_IM,j_IM) = tuple(np.argwhere(np.reshape(mult,(N,S))==1)[0])
                    
                    A_IM[t,h,i,j] = A[t-1,h,i_IM,j_IM]
                    
                else: A_IM[t,h,i,j] = A[t-1,h,i,j]
                
    
    A[t] = np.maximum(A[t-1], A_IN[t], A_IM[t])
    print(A[:,0,0,4])
    print(t)
    
    # Evolution of mark-up ratio
    m[t] = m[t-1]*(1+v*(f[t-1]-f[t-2])/f[t-2])
    
    # Price with mark-up
    p[t] = (1+m[t])*W[t]/A[t]
    
    # Desired Production (Myopic)
    Qd[t] = D[t-1]
    
    # Actual Production
    Q[t] = np.minimum(Qd[t], K[t]/B)
    
    # Desired capital
    Kd[t] = B*Qd[t]
    
    # Expansion Investment due to limited capital
    Ie[t] = np.minimum(0, Kd[t]-K[t])
    
    # Replacement Investment due to capital depreciation
    Ir[t] = delta*K[t]
    
    # Dynamics of Capital Stock
    K[t+1] = K[t] + Ie[t]
    
    # Total domestic capital production
    Qk[t,:] = np.sum(Ie[t]+Ir[t], axis=(0,2))
    
    # Labor employed in the capital sector
    Lk[t] = Qk[t]/np.average(A[t], axis=(0,2))
    
    # Price tracks unit cost of production ??
    pass

    
    
print(A_IN[:,0,0,0])
    