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

# Define Functions
def Euc_inv(A):
    
    A = A.reshape(N, S, 1, 1)               # Reshape A to N x S x 1 x 1
    Euc = abs(A - A.T).transpose(0,1,3,2)   # Compute pairwise differences
    if np.all(Euc==0): 
        Euc = np.ones_like(Euc, dtype='float64')
    else:
        Euc[Euc==0] = np.min(Euc[Euc>0])
    
    Euc *= F
    Euc[in_i[:, np.newaxis], in_j[np.newaxis, :], in_i[:, np.newaxis], in_j[np.newaxis, :]] = -1.0
    Euc_inverse = Euc**(-1)
    Euc_inverse[in_i[:, np.newaxis], in_j[np.newaxis, :], in_i[:, np.newaxis], in_j[np.newaxis, :]] = 0
    
    return Euc_inverse

def e_pair(e_i):
    
    e_i = e_i.reshape(N,1)
    return e_i*(e_i.T**(-1))

# Parameter Values

# Number of Industries
M = 30
# Number of Countries
N = 60
# Number of firms (each industry)
S = 20
# Sectoral demand shares
d_h = 1/M
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
# Foreign Penalty Matrix
in_i, in_j = np.arange(N), np.arange(S)
F = np.full((N,S,N,S),eps)
# No penalty for same country
F[in_i[:, np.newaxis], :, in_i[:, np.newaxis], :] = 1
# Prevent ZeroDivisionError for same firm
#F[i[:, np.newaxis], j[np.newaxis, :], i[:, np.newaxis], j[np.newaxis, :]] = -1
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
T = 505
T_start = 2

  

# Initiate Variables

# Sales
#SS = np.zeros((T,M,N,S))
#SS[0:T_start]=20
# R&D expenditure
RD = np.zeros((T,M,N,S))
# Productivity
A = np.zeros((T,M,N,S), dtype='float64')
A[0:T_start]=1.0
A_IN = np.zeros((T,M,N,S))
A_IM = np.zeros((T,M,N,S))
A_av = np.zeros((T,N)) # National average productivity
A_av[0:T_start]=1.0
A_av_g = np.zeros((T,N))
# Innovative expenditure and Imitative expenditure
IN = np.zeros((T,M,N,S))
IM = np.zeros((T,M,N,S))
# Success rate of IN, IM
theta_IN = np.zeros((T,M,N,S))
theta_IM = np.zeros((T,M,N,S))
# Exchange Rate Matrix
e_i = np.zeros((T,N))
e_i[0:T_start+1] = 1
e = np.ones((T,N,N))
# Market share
    # f(t,h,i,j,k) = Market share of Time t, Industry h, Country of origin i, firm j, Operating in country k
f = np.zeros((T,M,N,S,N)) 
f[0:T_start] = 1/(N*S) #fix to dynamics
f_j = np.zeros((T,M,N,S)) # Average Market share of Time t, Industry h, Country of origin i, firm j
f_j[0:T_start] = 1/(N*S)
# Cost of Production
CGS = np.zeros((T,M,N,S))
# Competitiveness (international)
    # E(t,h,i,j,k) = Competitveness of Time t, Industry h, Country of origin i, firm j, Operating in country k
E = np.zeros((T,M,N,S,N))
    # E_av(t,h,k) = Average Competitiveness of Time t, Industry h, in Market Country k
E_av = np.zeros((T,M,N))
# Desired capital stock
Kd = np.zeros((T,M,N,S))
# Mark up price
m = np.zeros((T,M,N,S))
m[0:T_start] = 0.2 #check value
# Nominal wage
W = np.zeros((T,N))
W[0:T_start+1] = 1 # fix
# Price of consumer goods
p = np.zeros((T,M,N,S))
p[0:T_start+1]= (1+m[0])*W[0,np.newaxis,:,np.newaxis]/A[0]
# Desired Production
Qd = np.zeros((T,M,N,S))
# Actual Production
Q = np.zeros((T,M,N,S))
# Demand
D = np.zeros((T,M,N,S))
D[0:T_start] = 10
D_mat = np.zeros((T,M,N,S,N))
D_int = np.zeros((T,M,N,S))
D_exp = np.zeros((T,M,N,S))
# Capital Stock
K = np.zeros((T,M,N,S))
K[0:T_start+1]= D[0]*B
# Expansion Investment
Ie = np.zeros((T,M,N,S))
# Replacement Investment
Ir = np.zeros((T,M,N,S))
# Labor employed in consumer goods sector
Lc = np.zeros((T,M,N,S))
# Labor employed in capital sector
Lk = np.zeros((T,N))
# Total Labor employed
L = np.zeros((T,N))
# Aggregate Demand
AD = np.zeros((T,N))
# National Consumption
C = np.zeros((T,N))
# National Investment (domestic production of capital sector)
I = np.zeros((T,N))
# National Exports
EXP = np.zeros((T,N))
# National Imports
IMP = np.zeros((T,N))
# Trade Balance
TB = np.zeros((T,N))
# GDP
Y = np.zeros((T,N))
Y_world = np.zeros(T)

for t in range(T_start,T-1):
    
    # R&D Expenditure
    RD[t] = rho*Q[t-1]
    IN[t] = lamb*RD[t]
    IM[t] = (1-lamb)*RD[t]
    
    # 1st step: determine success rate of IN and IM search
    theta_IN[t] = np.minimum(theta_max, 1 -np.exp(-xi_IN *IN[t]))
    theta_IM[t] = np.minimum(theta_max, 1 -np.exp(-xi_IM *IM[t]))
    
    
    # Success of Innovation and Imitation
    IN_success = np.random.binomial(1, theta_IN[t])
    IM_success = np.random.binomial(1, theta_IM[t])
    
    # 2nd tep: Maximum of A, A_IN,, A_IM
    A_IN[t] = A[t-1]*IM_success* \
              ((x1_l+np.random.beta(alp1, beta1, (M,N,S)))*(x1_u -x1_l))
        
    
    for h in range(M):
        
        Eu_inv_Mat = Euc_inv(A[t-1,h,:,:])
        
        for i in range(N):
            
            for j in range(S):
                
                # Find target firm if imitation is successful
                if IM_success[h,i,j] == 1:                        
                
                    # Euclidian distance
                    Eu_inv = Eu_inv_Mat[i,j]
                    
                    # Multinomial Result
                    mult = np.random.multinomial(1, np.reshape(Eu_inv/np.sum(Eu_inv),N*S))
                    (i_IM,j_IM) = tuple(np.argwhere(np.reshape(mult,(N,S))==1)[0])
                    
                    A_IM[t,h,i,j] = A[t-1,h,i_IM,j_IM]
                    
                else: A_IM[t,h,i,j] = A[t-1,h,i,j]
                
    
    A[t] = np.maximum(A[t-1], A_IN[t], A_IM[t])
    
    
    # Desired Production (Myopic)
    Qd[t] = D[t-1]
    
    # Actual Production
    Q[t] = np.minimum(Qd[t], K[t]/B)
    
    # Labor employed in the consumer goods sector
    Lc[t] = Q[t]/A[t]
    
    # Cost of Production
    #CGS[t] = 
    
    # National Average Productivity A_av[t] = np.average(A[t], axis=(0,2))
    A_av[t] = np.sum(Q[t],axis=(0,2))/np.sum(Lc[t],axis=(0,2))
    
    # Growth rate of National Average Productivity
    A_av_g[t] = A_av[t]/A_av[t-1] -1
    
    # Evolution of mark-up ratio
    m[t] = m[t-1]*(1+v*(f_j[t-1]-f_j[t-2])/f_j[t-2])
        
    
    # Desired capital
    Kd[t] = B*Qd[t]
    
    # Expansion Investment due to limited capital
    Ie[t] = np.maximum(0, Kd[t]-K[t])
    
    # Replacement Investment due to capital depreciation
    Ir[t] = delta*K[t]
    
    # Dynamics of Capital Stock
    K[t+1] = K[t] + Ie[t]
    
    # Total domestic capital production
    I[t] = np.sum(Ie[t]+Ir[t], axis=(0,2))
    
    # Labor employed in the capital sector
    Lk[t] = I[t]/A_av[t]
    
    # Totla Labor employed
    L[t] = np.sum(Lc[t], axis=(0,2)) + Lk[t] #???? check
    
    # Dynamics of Wage
    W[t] = W[t-1]*(1+ psi1*A_av_g[t])
    #W[t] = 
    
    # Price with mark-up
    p[t] = (1+m[t])*W[t,np.newaxis,:,np.newaxis]/A[t]
    
    # Price tracks unit cost of production ??
    pass

    # Calculate Sales for current time step
    #SS[t] = Q[t]
    
    # Aggregate Consumption
    C[t] = W[t]*L[t]
    
    # Dynamics of Exchange Rates
    if t > T_start:
        e_i[t] = e_i[t-1]*(1+gam*TB[t-1]/Y_world[t-1] +np.random.multivariate_normal(np.ones(N),sige*np.eye(N)))
    e[t] = e_pair(e_i[t])  
    
    print(t)
    
    # Competitiveness of internationally operated firms
    E[t] = ((1+tau)*p[t,:,:,:,np.newaxis]*e[t,np.newaxis,:,np.newaxis,:])**(-1)
    E_av[t] = np.sum(E[t]*f[t-1], axis = (1,2))
    
    # Dynamics of Market Shares
    f[t] = f[t-1]*(1 -chi+ chi*(E[t]/E_av[t,:,np.newaxis,np.newaxis,:]))
    f_j[t] = np.average(f[t], axis = 3)
    
    # Exit and Entry
    
    
    # Demand Matrix for the whole system
    D_mat[t] = d_h *  W[t,np.newaxis,np.newaxis,np.newaxis,:] * L[t,np.newaxis,:,np.newaxis,np.newaxis] * e[t,np.newaxis,:,np.newaxis,:]* f[t]
    D_int[t] = D_mat[t,:,in_i,:,in_i].transpose(1,0,2)
    D_exp[t] = np.sum(D_mat[t], axis=(3)) - D_int[t]
    D[t] = D_int[t] + D_exp[t]
    
    
    # Aggregate Export/ Import
    EXP[t] = np.sum(D_exp[t], axis=(0,2))
    IMP[t] = C[t] - np.sum(D_int[t], axis=(0,2))
    TB[t] = EXP[t] - IMP[t]
    
    #AD[t] = EXP[t] -IMP[t] + C[t]
    
    # GDP
    Y[t] = C[t] + I[t] + TB[t]
    Y_world[t] = np.sum(Y[t],axis=0)

    
    #print(np.sum(TB[t]*e[t]))
    #print(np.max(f_j[t]))
    #print(IN_success)
    print(np.max(f[t,1]))

    
#print(A_IN[:,0,0,0])
    