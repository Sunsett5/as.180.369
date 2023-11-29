#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 01:06:32 2023

@author: setthakorntanom
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Define Functions

def dist(A):
    return np.log(A)

def Euc_inv(A):
    
    A = A.reshape(N, S, 1, 1)               # Reshape A to N x S x 1 x 1
    Euc = abs(dist(A) - dist(A.T)).transpose(0,1,3,2)   # Compute pairwise differences
    if np.all(Euc==0): 
        Euc = np.ones_like(Euc, dtype='float64')
    else:
        Euc[Euc==0] = np.min(Euc[Euc>0])
    
    Euc *= F
    Euc[in_i[:, np.newaxis], in_j[np.newaxis, :], in_i[:, np.newaxis], in_j[np.newaxis, :]] = -1.0
    Euc_inverse = Euc**(-1)
    Euc_inverse[in_i[:, np.newaxis], in_j[np.newaxis, :], in_i[:, np.newaxis], in_j[np.newaxis, :]] = 0
    
    return Euc_inverse

def Euc(A):
    
    A = A.reshape(N, S, 1, 1)               # Reshape A to N x S x 1 x 1
    Euc = abs(dist(A) - dist(A.T)).transpose(0,1,3,2)   # Compute pairwise differences
    if np.all(Euc==0): 
        Euc = np.ones_like(Euc, dtype='float64')
    else:
        Euc[Euc==0] = np.min(Euc[Euc>0])
    
    Euc *= F
    Euc[in_i[:, np.newaxis], in_j[np.newaxis, :], in_i[:, np.newaxis], in_j[np.newaxis, :]] = 0
    
    return Euc
    

def Ent(c,noise):
    A[0:T_start+1] = A[-1]
    j_max = np.argmax(A[0],1)
    A[0][P[-2]<np.percentile(P[-2],10)] = 0
    
    for i in range(N):
        n_s = np.size(A[0,i][A[0,i]>0])
        A_s = A[0,i][A[0,i]>0]
        rho_s = rho[c-1,i][A[0,i]>0]
        lamb_s = lamb[c-1,i][A[0,i]>0]
        #p_mat = np.fill((S),1/n_s)
        #p_mat[A[0,i] == 0]=0
        if n_s == 0: 
            A[0,i] = A[-1,i,j_max[i]]
            rho[c,i]=rho[-1,i,j_max[i]]
            lamb[c,i]=lamb[-1,i,j_max[i]]
            
            print(f"[{c},{i}]extinct")
            
            continue
        p_mat = [1/n_s]*n_s
        A_new = np.random.multinomial(1,p_mat,S-n_s)
        A_new_in = np.argwhere(A_new>0)
        #print(S-n_s, A_new_in[:,1])
        #print(rho_s[A_new_in[:,1]])
        rho[c,i]=rho[c-1,i]
        rho[c,i][A[0,i]==0] = rho_s[A_new_in[:,1]]
        #rho[c,i]*=np.exp(np.random.multivariate_normal(np.full((S),noise[0]), np.eye(S)*noise[1]))
        lamb[c,i]=lamb[c-1,i]
        lamb[c,i][A[0,i]==0] = lamb_s[A_new_in[:,1]]
        #lamb[c,i]*=np.exp(np.random.multivariate_normal(np.full((S),noise[0]), np.eye(S)*noise[1]))
        A[0,i][A[0,i]==0] = A_s[A_new_in[:,1]]

    b_rho = (alp-1)/rho[c] +2 - alp
    b_lamb = (alp-1)/lamb[c] +2 - alp
    rho[c]=np.random.beta(alp, b_rho)
    lamb[c]=np.random.beta(alp, b_lamb)
    

# Number of Countries
N = 20
# Number of firms (each industry)
S = 40
# Time Steps
T = 40
T_start = 0
# Cycle
C = 11
C_start = 1
# Monte-Carlo replications
mc_sims = 1

# beta-convergence
beta = np.zeros((mc_sims,C))
sd = np.zeros((mc_sims,C))

inv = 0
noise=(0,0)
elim = 1
plateau = 1
beta_ind = 1
div_start = 1

# Firms' search capabalities
#xi_IN = 1.386
xi_IN = 0.4
# Firms' search capabalities
xi_IM = 0.4
#xi_IM = 0.4


# Foreign imitation penalty
eps = 5 if inv else 1/5
# Foreign Penalty Matrix
in_i, in_j = np.arange(N), np.arange(S)
F = np.full((N,S,N,S),eps)
# No penalty for same country
F[in_i[:, np.newaxis], :, in_i[:, np.newaxis], :] = 1

# R&D expenditure
RD = np.zeros((T,N,S))
# Innovative expenditure and Imitative expenditure
IN = np.zeros((T,N,S))
IM = np.zeros((T,N,S))
# First stage probabilities upper bound
theta_max = 0.75
# Success rate of IN, IM
theta_IN = np.zeros((T,N,S))
theta_IM = np.zeros((T,N,S))

# Beta distribution parameter
alp = 100
alp_start = 10
(alp1, beta1) = (1,5)
# Beta distribution support
(x1_l, x1_u) = (-0.05,0.25)
# Beta distribution parameter (ent.)
(alp2, beta2) = (1,5)
# Beta distribution parameter (ent.)
(x2_l, x2_u) = (-0.03,0.15)


for isim in range(mc_sims):
    
    # R&D investment propensity
    rho = np.zeros((C,N,S))
    rho[:] = np.random.uniform(0,1,(N,S))
    # R&D allocation parameter
    lamb = np.zeros((C,N,S))
    lamb[:] = np.random.uniform(0,1,(N,S))
    
    # Productivity
    A = np.zeros((T,N,S), dtype='float64')
    if div_start:
        A[0:T_start+1] = np.random.beta(alp_start,1/alp_start,N)[np.newaxis,:, np.newaxis]
    else: A[0:T_start+1]=1.0
    A_IN = np.zeros((T,N,S))
    A_IM = np.zeros((T,N,S))
    A_av = np.zeros((C,N)) # National average productivity
    
    for c in range(C):
        
        if c==0: pass
        elif elim: Ent(c,noise)
        else: A[0:T_start+1] = A[-1]
        
        # Accumulated Profit
        P = np.zeros((T,N,S), dtype='float64')
        
        for t in range(T_start,T-1):
            
            P[t]=P[t-1]+(1-rho[c])*A[t]
            
            if plateau:
                
                theta_IN[t] = np.minimum(0.99, 1 -np.exp(-xi_IN *rho[c]*lamb[c]))
                theta_IM[t] = np.minimum(0.99, 1 -np.exp(-xi_IM *rho[c]*(1-lamb[c])))
                
            else:
                # R&D Expenditure
                RD[t] = rho[c]*A[t]
                IN[t] = lamb[c]*RD[t]
                IM[t] = (1-lamb[c])*RD[t]
                # 1st step: determine success rate of IN and IM search
                theta_IN[t] = np.minimum(theta_max, 1 -np.exp(-xi_IN *IN[t]))
                theta_IM[t] = np.minimum(theta_max, 1 -np.exp(-xi_IM *IM[t]))
                
               
            
            
            # Success of Innovation and Imitation
            IN_success = np.random.binomial(1, theta_IN[t])
            IM_success = np.random.binomial(1, theta_IM[t])
            
            #print(IN_success)
            
            # 2nd step: Maximum of A, A_IN,, A_IM
            A_IN[t+1] = A[t]*IN_success* \
                      (1+((x1_l+np.random.beta(alp1, beta1, (N,S)))*(x1_u -x1_l))  ) 
        
            if inv: Euc_Mat = Euc_inv(A[t,:,:])
            else: Euc_Mat = Euc(A[t,:,:])
            
                
            for i in range(N):
                    
                for j in range(S):
                        
                    # Find target firm if imitation is successful
                    if IM_success[i,j]:    
                        #print(f'{t}Imitated')                    
                        # Euclidian distance
                        Eu = Euc_Mat[i,j]    
                        # Multinomial Result
                        mult = np.random.multinomial(1, (Eu/np.sum(Eu)).flatten())
                        #mult = np.zeros((N*S,))
                        #mult[np.argmax((Eu/np.sum(Eu)).flatten())] = 1
                        (i_IM,j_IM) = tuple(np.argwhere(np.reshape(mult,(N,S))==1)[0])
                            
                        A_IM[t+1,i,j] = A[t,i_IM,j_IM]
                            
                    else: A_IM[t+1,i,j] = A[t,i,j]
            
            A[t+1] = np.maximum(np.maximum(A[t], A_IN[t+1]), A_IM[t+1])
           # print(t)
            
        A_av[c] = np.average(A[-1],1)
        
        sd[isim,c] = np.std(np.log(A[-1]))
        
        if c >= C_start:
        
            #print(A[0])
            
            if beta_ind:
                g = np.reshape(np.log(A[-1])-np.log(A[0]),(-1))
                A0 = np.reshape(A[0],(-1,1))
                model = LinearRegression().fit(A0,g)
                beta[isim,c] = model.coef_
            
            else:
                g = np.reshape(np.average(np.log(A[-1])-np.log(A[0]),0),(-1))
                A0 = np.reshape(np.average(A[0],0),(-1,1))
                model = LinearRegression().fit(A0,g)
                beta[isim,c] = model.coef_
        
        print(f"Monte Carlo Simulation {isim}, Cycle {c}")
           
    if mc_sims == 1:
        rho_all = rho[-1].flatten()
        lamb_all = lamb[-1].flatten()
        A_all = A[-1].flatten()
        P_all = P[-2].flatten()
        
        A_all_in = np.log(A_all)/np.std(np.log(A_all))
        P_all_in = np.log(P_all)/np.std(np.log(P_all))
        
        country = np.indices((N, S))
        country_ind = country[0].flatten()
        
        fig2, ax2 = plt.subplots()
        fig2.set_size_inches(8, 6)
        ax2.scatter(rho_all, lamb_all, s= 0.5*(A_all_in), color = 'black') #cmap = 'viridis')
        ax2.set_title("log(Productivity) of Surviving Firms")
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.set_xlim([0,1])
        ax2.set_ylim([0,1])
        ax2.set_xlabel('R&D Expenditure (ρ)', fontsize = 14)
        ax2.set_ylabel('Innovation Expenditure (λ)', fontsize = 14)
        fig2.savefig(f'A_survive_{N}_{S}_{T}_{C}_{xi_IN}_{xi_IM}.png', dpi = 300)
       
        fig3, ax3 = plt.subplots()
        fig3.set_size_inches(8, 6)
        ax3.scatter(rho_all, lamb_all, s = 0.1*P_all_in, color='black')
        ax3.set_title("log(Profits) of Surviving Firms")
        ax3.tick_params(axis='both', which='major', labelsize=12)
        ax3.set_xlim([0,1])
        ax3.set_ylim([0,1])
        ax3.set_xlabel('R&D Expenditure (ρ)', fontsize = 14)
        ax3.set_ylabel('Innovation Expenditure (λ)', fontsize = 14)
        fig3.savefig(f'profits_{N}_{S}_{T}_{C}_{xi_IN}_{xi_IM}.png', dpi = 300)
        #ax.set_zlabel('Profit')
        
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(8, 6)
        ax1.set_title('log(Productivity)')
        ax1.set_xlabel("Cycle")
        ax1.plot(range(C_start,C),np.log(A_av[C_start:C]))
        ax1.legend
        fig1.savefig(f'A_trend_{N}_{S}_{T}_{C}_{xi_IN}_{xi_IM}.png', dpi = 300)
        
        '''
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        fig.set_size_inches(8, 6)
        ax.view_init(30, 130)
        ax.plot_trisurf(rho_all, lamb_all, np.log(A_all), cmap='viridis', edgecolor='none');
        #ax.plot_trisurf(rho[c,0], lamb[c,0], P[-2,0], cmap='viridis', edgecolor='none');
        ax.set_title("Distribution of Surviving Firms")
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.set_xlabel('R&D Expenditure (ρ)', fontsize = 13)
        ax.set_ylabel('Innovation Expenditure (λ)', fontsize = 13)
        ax.set_zlabel('log(Productivity) (logA)', fontsize = 13)
        #ax.set_zlabel('Profit')
        '''

    
    
    
    
    
    
    continue

    fig2, ax2 = plt.subplots()
    ax2.plot(range(C_start,C),beta[isim,C_start:])
    ax2.set_title("β-convergence")
    #ax2.set_ylabel("beta")
    ax2.set_xlabel("Cycle")
    
    fig3, ax3 = plt.subplots()
    ax3.plot(range(C_start,C),sd[isim])
    ax3.legend
    
    
if mc_sims > 3: 
    fig4, ax4 = plt.subplots()
    fig4.set_size_inches(8, 6)
    ax4.set_title("β-convergence")
    ax4.set_xlabel("Cycle")
    ax4.plot(range(C_start,C),np.average(beta[:,C_start:],0),color='black')
    ax4.fill_between(range(C_start,C), np.percentile(beta[:,C_start:],2.5,0), np.percentile(beta[:,C_start:],97.5,0), color='black', alpha=.1)
    ax4.legend
    fig4.savefig(f'beta_{N}_{S}_{T}_{C}_{xi_IN}_{xi_IM}.png', dpi = 300)
    
print(f"xi_IN: {xi_IN}, xi_IM: {xi_IM}")
