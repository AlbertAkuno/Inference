#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:24:19 2023

@author: albertakuno
"""

from __future__ import division
from scipy.integrate import odeint
from scipy import integrate,stats,special
#from xlrd import open_workbook
import numpy as np
#import scipy as sp
import pylab as pl
import pytwalk
import scipy.stats as ss
import matplotlib.pyplot as plt
import pandas as pd
#import emcee
#import time
#import math
#from scipy.special import gamma
import random
import corner
import seaborn as sns
import statsmodels.api as sm
from scipy.stats.mstats import mquantiles
#import warnings


plt.style.use('seaborn-talk') # beautify the plots!

TotalNumIter =  1000000#600000#600000
burnin       =  360000#100000
LastNumIter  =    30000
NumEqs       =      24
NumParams    =      24

#the path for saving the figures
save_results_to = '/Volumes/F/Hermosillo_four_regions data/figures_twalk' 


dat_one_alpha_FF = pd.read_csv("/Volumes/F/Hermosillo_AGEBs_data/one_alpha_POBTOT/alphas_FS.csv") 
df_4zones2 = pd.read_excel(r"/Volumes/F/Hermosillo_four_regions data/Ageb_zonas_updated.xlsx")


#frame = [dat_one_alpha_FF,df_4zones2]

#dat_one_alpha_FF2 = dat_one_alpha_FF
dat_one_alpha_FF["Zona"] = df_4zones2["Zona"]

#POBTOT_zone2 = dat_one_alpha_FF[["POBTOT", "Zona"]]
#Nbar_data2 = POBTOT_zone2.groupby('Zona').sum().reset_index()
#Nbar2 = np.array(Nbar_data2["POBTOT"])
#N1 = Nbar2[0]
#N2 = Nbar2[1]
#N3 = Nbar2[2]
#N4 = Nbar2[3]

#new_df = pd.concat(frame, axis=1)

#POBTOT_zone = new_df[["POBTOT", "Zona"]]
POBTOT_zone = dat_one_alpha_FF[["POBTOT", "Zona"]]

#df_zona_POBTOT = pd.concat([dat_one_alpha_FF["POBTOT"], df_4zones2["Zona"] ])

Nbar_data = POBTOT_zone.groupby('Zona').sum().reset_index()

zone1_cases = np.load("/Volumes/F/Hermosillo_four_regions data/zonal_COVID-19_data/zone1_COVID-19_data.npy")
zone2_cases = np.load("/Volumes/F/Hermosillo_four_regions data/zonal_COVID-19_data/zone2_COVID-19_data.npy")
zone3_cases = np.load("/Volumes/F/Hermosillo_four_regions data/zonal_COVID-19_data/zone3_COVID-19_data.npy")
zone4_cases = np.load("/Volumes/F/Hermosillo_four_regions data/zonal_COVID-19_data/zone4_COVID-19_data.npy")

zone1_cases_inflated = zone1_cases*15
zone2_cases_inflated = zone2_cases*15
zone3_cases_inflated = zone3_cases*15
zone4_cases_inflated = zone4_cases*15

zone1_cases_including_pred_data = np.load("/Volumes/F/Hermosillo_four_regions data/zonal_COVID-19_data/zone1_COVID-19_including_pred_data.npy")
zone2_cases_including_pred_data = np.load("/Volumes/F/Hermosillo_four_regions data/zonal_COVID-19_data/zone2_COVID-19_including_pred_data.npy")
zone3_cases_including_pred_data = np.load("/Volumes/F/Hermosillo_four_regions data/zonal_COVID-19_data/zone3_COVID-19_including_pred_data.npy")
zone4_cases_including_pred_data = np.load("/Volumes/F/Hermosillo_four_regions data/zonal_COVID-19_data/zone4_COVID-19_including_pred_data.npy")

zone1_cases_including_pred_data_inflated = zone1_cases_including_pred_data*15
zone2_cases_including_pred_data_inflated = zone2_cases_including_pred_data*15
zone3_cases_including_pred_data_inflated = zone3_cases_including_pred_data*15
zone4_cases_including_pred_data_inflated = zone4_cases_including_pred_data*15


def generate_actual_pstar(p, alpha):
    pstar = p * alpha.reshape(-1, 1) + np.diag(
        1
        - alpha.reshape(
            -1,
        )
    )
    return pstar


##alpha and pstar for the third period first part
alpha = np.array( [0.9668, 0.9265, 0.9692, 0.9680])
p = np.array([ [0.8164, 0.1289, 0.0372, 0.0175], [0.1222, 0.8119, 0.0215, 0.0444], [0.0722, 0.0504, 0.7293, 0.1481], [0.0313, 0.1166, 0.1278, 0.7243] ])
pstar = generate_actual_pstar(p, alpha)

mu = np.ones([4])* (0.06/(1000*365))
tau = np.ones([4])* (1/180)
Nbar = np.array(Nbar_data["POBTOT"])
Lambda = np.ones([4])* (15.7/(1000*365))
phi = np.ones([4])*0.00385

ttime  = np.linspace(0.0,float(len(zone1_cases_inflated))+1,len(zone1_cases_inflated)+1)
ttime2 = np.linspace(0.0,float(len(zone1_cases_inflated)),len(zone1_cases_inflated))
times_pred = np.linspace(0.0,230,230)
times_pred2 = np.linspace(0.0,229,229)

n_days=len(zone1_cases_inflated)
n_pred=len(times_pred)


# Sistema de ecuaciones del modelo SEIRS.

def deriv(Y, t, beta1, beta2, beta3, beta4,kappa1, kappa2, kappa3, kappa4, gamma1, gamma2, gamma3, gamma4):
    S1, S2, S3, S4, E1, E2, E3, E4, I1, I2, I3, I4, Y1, Y2, Y3, Y4, R1, R2, R3, R4, N1, N2, N3, N4 = Y
    S = np.array([S1, S2, S3, S4])
    E = np.array([E1, E2, E3, E4])
    I = np.array([I1, I2, I3, I4])
    R = np.array([R1, R2, R3, R4])
    N = np.array([N1, N2, N3, N4])
    beta = np.array([beta1,beta2,beta3, beta4])
    kappa = np.array([kappa1, kappa2 ,kappa3 ,kappa4])
    gamma = np.array([gamma1,gamma2,gamma3,gamma4])
    Ntilde=np.transpose(pstar)@ Nbar
    M1 = np.diag(Lambda)@N -np.diag(S) @ pstar@ np.diag(beta) @ np.linalg.inv(np.diag(Ntilde)) @ np.transpose(pstar)@I\
     - np.diag(mu)@S + np.diag(tau)@R
    M2 = np.diag(S) @ pstar@ np.diag(beta) @ np.linalg.inv(np.diag(Ntilde)) @ np.transpose(pstar)@I \
    -np.diag(kappa+mu)@E 
    M3 = np.diag(kappa)@E - np.diag(gamma + phi + mu)@I
    M4 = np.diag(kappa)@E - np.diag(gamma + phi + mu)@I + np.diag(gamma + phi + mu)@I
    M5 = np.diag(gamma)@I - np.diag(tau+mu)@R
    M6 = np.diag(Lambda)@N - np.diag(mu)@N - np.diag(phi)@I
    
    dS1dt = M1[0]
    dS2dt = M1[1]
    dS3dt = M1[2]
    dS4dt = M1[3]
    
    dE1dt = M2[0]
    dE2dt = M2[1]
    dE3dt = M2[2]
    dE4dt = M2[3]
    
    dI1dt = M3[0]
    dI2dt = M3[1]
    dI3dt = M3[2]
    dI4dt = M3[3]
    
    dY1dt = M4[0]
    dY2dt = M4[1]
    dY3dt = M4[2]
    dY4dt = M4[3]
    
    dR1dt = M5[0]
    dR2dt = M5[1]
    dR3dt = M5[2]
    dR4dt = M5[3]
    
    dN1dt = M6[0]
    dN2dt = M6[1]
    dN3dt = M6[2]
    dN4dt = M6[3]


    return np.array([dS1dt, dS2dt, dS3dt, dS4dt, dE1dt, dE2dt, dE3dt, dE4dt, dI1dt, dI2dt, dI3dt, dI4dt, dY1dt, dY2dt, dY3dt, dY4dt, dR1dt, dR2dt, dR3dt, dR4dt, dN1dt, dN2dt, dN3dt, dN4dt])


def solve(par):
    beta1 = par[0]
    beta2 = par[1]
    beta3 = par[2]
    beta4 = par[3]
    kappa1 = par[4]
    kappa2 = par[5]
    kappa3 = par[6]
    kappa4 = par[7]
    gamma1 = par[8]
    gamma2 = par[9]
    gamma3 = par[10]
    gamma4 = par[11]
    e01 = par[12]
    i01 = par[13]
    e02 = par[14]
    i02 = par[15]
    e03 = par[16]
    i03 = par[17]
    e04 = par[18]
    i04 = par[19]
    
    R01 = 0; R02 = 0; R03 = 0; R04 = 0
    
    N01 =Nbar[0]
    N02 =Nbar[1]
    N03 =Nbar[2]
    N04 =Nbar[3]
    
    S01 = N01 - e01 - i01 - R01
    S02 = N02 - e02 - i02 - R02
    S03 = N03 - e03 - i03 - R03
    S04 = N04 - e04 - i04 - R04
    
    Y01 = e01 + i01 + R01
    Y02 = e02 + i02 + R02
    Y03 = e03 + i03 + R03
    Y04 = e04 + i04 + R04


    
    y0 = S01, S02, S03, S04, e01, e02, e03, e04, i01, i02,  i03, i04, Y01,  Y02, Y03, Y04, R01, R02, R03, R04, N01, N02, N03, N04
    
    ret = odeint(deriv, y0, ttime, args=(beta1, beta2, beta3, beta4, kappa1, kappa2, kappa3, kappa4, gamma1, gamma2, gamma3, gamma4,))
    
    result_InsP1 = np.diff(ret[:,12])
    result_InsP2 = np.diff(ret[:,13])
    result_InsP3 = np.diff(ret[:,14])
    result_InsP4 = np.diff(ret[:,15])
    
    return result_InsP1, result_InsP2, result_InsP3, result_InsP4

    
def solve_pred(par):
    
    beta1 = par[0]
    beta2 = par[1]
    beta3 = par[2]
    beta4 = par[3]
    kappa1 = par[4]
    kappa2 = par[5]
    kappa3 = par[6]
    kappa4 = par[7]
    gamma1 = par[8]
    gamma2 = par[9]
    gamma3 = par[10]
    gamma4 = par[11]
    e01 = par[12]
    i01 = par[13]
    e02 = par[14]
    i02 = par[15]
    e03 = par[16]
    i03 = par[17]
    e04 = par[18]
    i04 = par[19]
    
    R01 = 0; R02 = 0; R03 = 0; R04 = 0
    
    N01 =Nbar[0]
    N02 =Nbar[1]
    N03 =Nbar[2]
    N04 =Nbar[3]
    
    S01 = N01 - e01 - i01 - R01
    S02 = N02 - e02 - i02 - R02
    S03 = N03 - e03 - i03 - R03
    S04 = N04 - e04 - i04 - R04
    
    Y01 = e01 + i01 + R01
    Y02 = e02 + i02 + R02
    Y03 = e03 + i03 + R03
    Y04 = e04 + i04 + R04


    
    y0 = S01, S02, S03, S04, e01, e02, e03, e04, i01, i02,  i03, i04, Y01,  Y02, Y03, Y04, R01, R02, R03, R04, N01, N02, N03, N04
    
    ret = odeint(deriv, y0, times_pred, args=(beta1, beta2, beta3, beta4, kappa1, kappa2, kappa3, kappa4, gamma1, gamma2, gamma3, gamma4,))
    
    result_InsP1 = np.diff(ret[:,12])
    result_InsP2 = np.diff(ret[:,13])
    result_InsP3 = np.diff(ret[:,14])
    result_InsP4 = np.diff(ret[:,15])
    
    return result_InsP1, result_InsP2, result_InsP3, result_InsP4


def ratioGamma_beta(y,phi):
    """
    Function to compute Gamma(y+phi)/Gamma(y) based on the beta function
    y>0 and phi>0
    """
    epsilon = 1e-8
    res = 1/special.beta(y+1,phi+epsilon)*special.gamma(phi+epsilon)
    res[y==0] = 1/special.beta(2,phi+epsilon)*special.gamma(phi+epsilon)
    return np.array(res)

#def ratioGamma_beta1(y,phi):
#    """
#    Function to compute Gamma(y+phi)/Gamma(y) based on the beta function
#    y>0 and phi>0
#    """
#    epsilon = 1e-8
#    res = 1/special.beta(y+1,phi+epsilon)*special.gamma(phi+epsilon)
#    return np.array(res)



def lognorm_mu_s(x, mu, s):
    tempX = x / np.exp(mu)
    return ss.lognorm.pdf(tempX, s)

#ss.lognorm.rvs(s = 0.2803,scale= np.exp(-0.0529))

def energy(par):
    
    """
    function of overdispersion parameters
    par[24]: nu1; 
    par[25]; nu2; 
    par[26]: nu3; 
    par[27]: nu4
    and the rest of the parameters
    """
    beta1 = par[0]
    beta2 = par[1]
    beta3 = par[2]
    beta4 = par[3]
    kappa1 = par[4]
    kappa2 = par[5]
    kappa3 = par[6]
    kappa4 = par[7]
    gamma1 = par[8]
    gamma2 = par[9]
    gamma3 = par[10]
    gamma4 = par[11]
    e01 = par[12]
    i01 = par[13]
    e02 = par[14]
    i02 = par[15]
    e03 = par[16]
    i03 = par[17]
    e04 = par[18]
    i04 = par[19]
    nu1 = par[20]
    nu2 = par[21]
    nu3 = par[22]
    nu4 = par[23]
    
    if support(par):
        my_soln_InsP1,my_soln_InsP2,my_soln_InsP3,my_soln_InsP4 = solve(par)        
        
#        log_likelihood1 = -np.sum(my_soln_s-Suspect*np.log(my_soln_s))
#        log_likelihood1 = np.sum(Suspect*np.log(my_soln_s)
#                            -(p[10] + Suspect)*np.log(p[10] + my_soln_s) ) \
#             -len(Suspect)*(-p[10]*np.log(p[10])+ np.log(gamma(p[10]) )  ) 
#        log_likelihood2 = np.sum(np.log( ss.nbinom.pmf(Suspect,p[10],
#                                         p[10]/(p[10] + my_soln_Q )) ) )
        #np.log(special.gamma(phi1 + zone1_cases_sim)) +
        
        ratioGamma_betaP1 = ratioGamma_beta(zone1_cases_inflated,nu1)
        ratioGamma_betaP2 = ratioGamma_beta(zone2_cases_inflated,nu2)
        ratioGamma_betaP3 = ratioGamma_beta(zone3_cases_inflated,nu3)
        ratioGamma_betaP4 = ratioGamma_beta(zone4_cases_inflated,nu4)
        
        #log_gamma_y1_nu1 = np.log(special.gamma(nu1 + zone1_cases_inflated))
        
        #log_gamma_y2_nu2 = np.log(special.gamma(nu2 + zone2_cases_inflated))
        
        #log_gamma_y3_nu3 = np.log(special.gamma(nu3 + zone3_cases_inflated))
        
        #log_gamma_y4_nu4 = np.log(special.gamma(nu4 + zone1_cases_inflated))
        
        
        log_likelihood1 = np.sum(np.log(ratioGamma_betaP1) + zone1_cases_inflated*np.log(my_soln_InsP1) \
                            -(nu1 + zone1_cases_inflated)*np.log(nu1 + my_soln_InsP1) ) \
             -len(zone1_cases_inflated)*(-nu1*np.log(nu1)+ np.log(special.gamma(nu1) )  )
        
        log_likelihood2 = np.sum(np.log(ratioGamma_betaP2) + zone2_cases_inflated*np.log(my_soln_InsP2) \
                            -(nu2 + zone2_cases_inflated)*np.log(nu2 + my_soln_InsP2) ) \
             -len(zone2_cases_inflated)*(-nu2*np.log(nu2)+ np.log(special.gamma(nu2) )  )
#        
        log_likelihood3 = np.sum( np.log(ratioGamma_betaP3) + zone3_cases_inflated*np.log(my_soln_InsP3)
                            -(nu3 + zone3_cases_inflated)*np.log(nu3 + my_soln_InsP3) ) \
             -len(zone3_cases_inflated)*(-nu3*np.log(nu3)+ np.log(special.gamma(nu3) )  ) 
             
        log_likelihood4 = np.sum(np.log(ratioGamma_betaP4) + zone4_cases_inflated*np.log(my_soln_InsP4)
                            -(nu4 + zone4_cases_inflated)*np.log(nu4 + my_soln_InsP4) ) \
             -len(zone4_cases_inflated)*(-nu4*np.log(nu4)+ np.log(special.gamma(nu4) )  ) 

       
        # gamma distribution parameters for p[4] = kappa1
        k0 = 4.5264
        theta0 = 19.1006
        # gamma distribution parameters for p[5] = kappa2
        k1 = 4.5264
        theta1 = 19.1006
        # gamma distribution parameters for p[6] = kappa3
        k2 = 4.5264
        theta2 = 19.1006
        # gamma distribution parameters for p[7] = kappa4
        k3 = 4.5264
        theta3 = 19.1006
        
        
        
        # gamma distribution parameters for p[8] = gamma1
        k4 = 1.9826
        theta4 = 3.6943
        # gamma distribution parameters for p[9] = gamma2
        k5 = 1.9826
        theta5 = 3.6943
        # gamma distribution parameters for p[10] = gamma3
        k6 = 1.9826
        theta6 = 3.6943
        
        # gamma distribution parameters for p[11] = gamma4
        k7 = 1.9826
        theta7 = 3.6943
        
        
        
        # gamma distribution parameters for p[24] = nu1
        k8 = 0.5761
        theta8 = 0.2714
        
        # gamma distribution parameters for p[25] = nu2
        k9 = 0.5761
        theta9 = 0.2714
        
        # gamma distribution parameters for p[26] = nu3
        k10 = 0.5761
        theta10 = 0.2714
        # gamma distribution parameters for p[27] = nu4
        k11 = 0.5761
        theta11 = 0.2714
        
        #dist=ss.lognorm([0.4394], loc = -0.3425) #dist=ss.lognorm([std], loc=mean)
        
        #a0 = dist.pdf(beta1)
        #a1 = dist.pdf(beta2)
        #a2 = dist.pdf(beta3)
        #a3 = dist.pdf(beta4)
        
        a0 = np.log(lognorm_mu_s(beta1, -0.0529, 0.2803))
        a1 = np.log(lognorm_mu_s(beta2, -0.0975, 0.2804))
        a2 = np.log(lognorm_mu_s(beta3, -0.8562, 0.2803))
        a3 = np.log(lognorm_mu_s(beta4, 0.2583, 0.2803))
    
        #a0 = ss.lognorm.pdf(beta1, -0.3425, 0.4394)
        #a1 = ss.lognorm.pdf(beta2, -0.3425, 0.4394)
        #a2 = ss.lognorm.pdf(beta3, -0.3425, 0.4394)
        #a3 = ss.lognorm.pdf(beta4, -0.3425, 0.4394)
        
        a4 = (k0-1)*np.log(kappa1)- (theta0*kappa1) 
        a5 = (k1-1)*np.log(kappa2)- (theta1*kappa2)
        a6 = (k2-1)*np.log(kappa3)- (theta2*kappa3)
        a7 = (k3-1)*np.log(kappa4)- (theta3*kappa4)
#        a0 = (k0-1)*np.log(p[0])- (p[0]/theta0)
#        a1 = (k1-1)*np.log(p[1])- (p[1]/theta1)
        a8 = (k4-1)*np.log(gamma1)- (theta4*gamma1) 
        a9 = (k5-1)*np.log(gamma2)- (theta5*gamma2)
        a10 = (k6-1)*np.log(gamma3)- (theta6*gamma3)
        a11 = (k7-1)*np.log(gamma4)- (theta7*gamma4)
#        a5 = (k5-1)*np.log(p[5])- (p[5]/theta5)
        
        a12 = np.log(ss.uniform.pdf(e01, 50., 65.))
        a13 = np.log(ss.uniform.pdf(i01, 0., 10.))
        
        a14 = np.log(ss.uniform.pdf(e02, 70, 85.))
        a15 = np.log(ss.uniform.pdf(i02, 0., 10.))
        
        a16 = np.log(ss.uniform.pdf(e03, 15, 30.))
        a17 = np.log(ss.uniform.pdf(i03, 0., 10.))
        
        a18 = np.log(ss.uniform.pdf(e04, 0., 10.))
        a19 = np.log(ss.uniform.pdf(i04, 0., 10.))
        
        
        a20 = (k8-1)*np.log(nu1)- (theta8*nu1)
        a21 =(k9-1)*np.log(nu2)- (theta9*nu2)
        a22 = (k10-1)*np.log(nu3)- (theta10*nu3)
        a23 = (k11-1)*np.log(nu4)- (theta11*nu4)
        
#        a8 = ss.uniform.pdf(p[8], 0., 0.3)
#        a5 = (k5-1)*np.log(p[5])- (p[5]/theta5)
#        a6 = (k6-1)*np.log(p[6])- (p[6]/theta6)
#        a7 = (k7-1)*np.log(p[7])- (p[7]/theta7)

#        a11= (k11-1)*np.log(p[11])- (p[11]/theta11)
#        a12= (k12-1)*np.log(p[12])- (p[12]/theta10)
 
        log_prior = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + \
                    a12 + a13 + a14 + a15 + a16 + a17 + a18 + a19 + a20 + a21 + a22 + a23
                    
        return -log_likelihood1 - log_likelihood2 - log_likelihood3 - log_likelihood4 - log_prior
    return -np.infty




def support(par):
    
    beta1 = par[0]
    beta2 = par[1]
    beta3 = par[2]
    beta4 = par[3]
    kappa1 = par[4]
    kappa2 = par[5]
    kappa3 = par[6]
    kappa4 = par[7]
    gamma1 = par[8]
    gamma2 = par[9]
    gamma3 = par[10]
    gamma4 = par[11]
    e01 = par[12]
    i01 = par[13]
    e02 = par[14]
    i02 = par[15]
    e03 = par[16]
    i03 = par[17]
    e04 = par[18]
    i04 = par[19]
    nu1 = par[20]
    nu2 = par[21]
    nu3 = par[22]
    nu4 = par[23]
    
    rt = True
    
    rt &= (0.5475 < beta1 < 1.6425)  #beta1
    rt &= (0.5237 < beta2 < 1.57098)  #beta2
    rt &= (0.2452 < beta3 < 0.7356)  #beta3
    rt &= (0.7476 < beta4 < 2.2427)  #beta4
    rt &= (0.0714 < kappa1 < 0.5)  #kappa1
    rt &= (0.0714 < kappa2 < 0.5)  #kappa2
    rt &= (0.0714 < kappa3 < 0.5)  #kappa3
    rt &= (0.0714 < kappa4 < 0.5)  #kappa4
    rt &= (0.0641 < gamma1 < 1.5)  #gamma1
    rt &= (0.0641 < gamma2 < 1.5)  #gamma2
    rt &= (0.0641 < gamma3 < 1.5)  #gamma3
    rt &= (0.0641 < gamma4 < 1.5)  #gamma4
    rt &= (50 < e01 < 65)  #E01
    rt &= (0  < i01 < 10)  #I01 [0,20]
    rt &= (70 < e02 < 85)  #E02 [50,100]
    rt &= (0  < i02 < 10)  #I02 [0,20]
    rt &= (15 < e03 < 30)  #E03 [0,50]
    rt &= (0 < i03 < 10)  #I03 [0,20]
    rt &= (0 < e04 < 10)  #E04 [0,50]
    rt &= (0 < i04 < 10)  #I04[0,20]
    rt &= ( 0.1 < nu1 < 10)  #nu1
    rt &= ( 0.1 < nu2 < 10)  #nu2
    rt &= ( 0.1 < nu3 < 10)  #nu3
    rt &= ( 0.1 < nu4 < 10)  #nu4
    
    return rt


def init0():
    par = np.zeros(NumParams)
    par[0] = 1.0950 #beta1
    par[1] = 1.0473 #beta2
    par[2] = 0.4904 #beta3
    par[3] = 1.4951 #beta4
    par[4] = 0.30 #kappa1
    par[5] = 0.30 #kappa2
    par[6] = 0.30 #kappa3
    par[7] = 0.30 #kappa4
    par[8] = 1.0 #gamma1
    par[9] = 1.0 #gamma2
    par[10] = 1.0 #gamma3
    par[11] = 1.0 #gamma4
    par[12] = 58.9925 #E01
    par[13] = 6.3662 #I01
    par[14] = 78.1313 #E02
    par[15] = 5.5063 #I02
    par[16] = 22.9404 #E03
    par[17] = 5.7946 #I03
    par[18] = 4.4192 #E04
    par[19] = 2.6886 #I04
    par[20] = 8.9 #nu1
    par[21] = 8.2 #nu2
    par[22] = 4.6 #nu3
    par[23] = 5.4 #nu4
    
    return par



def init1():
    par = np.zeros(NumParams)
    par[0] = 1.100 #beta1
    par[1] = 1.051 #beta2
    par[2] = 0.50 #beta3
    par[3] = 1.50 #beta4
    par[4] = 0.31 #kappa1
    par[5] = 0.33 #kappa2
    par[6] = 0.29 #kappa3
    par[7] = 0.28 #kappa4
    par[8] = 1.01 #gamma1
    par[9] = 0.99 #gamma2
    par[10] = 1.02 #gamma3
    par[11] = 0.98 #gamma4
    par[12] = 59.000 #E01
    par[13] = 6.500 #I01
    par[14] = 78.000 #E02
    par[15] = 5.5500 #I02
    par[16] = 23.000 #E03
    par[17] = 5.8200 #I03
    par[18] = 4.5000 #E04
    par[19] = 2.700 #I04
    par[20] = 9.0 #nu1
    par[21] = 8.0 #nu2
    par[22] = 5.0 #nu3
    par[23] = 5.6 #nu4
    
    return par



def init():
    par = np.zeros(NumParams)
    par[0] = np.random.uniform(low=0.5475,high=1.6425) #beta1
    par[1] = np.random.uniform(low=0.5237,high=1.57098) #beta2
    par[2] = np.random.uniform(low=0.2452,high=0.7356) #beta3
    par[3] = np.random.uniform(low=0.7476,high=2.2427) #beta4
    par[4] = np.random.uniform(low=0.0714,high=0.5) #kappa1
    par[5] = np.random.uniform(low=0.0714,high=0.5) #kappa2
    par[6] = np.random.uniform(low=0.0714,high=0.5) #kappa3
    par[7] = np.random.uniform(low=0.0714,high=0.5) #kappa4
    par[8] = np.random.uniform(low=0.0641,high=1.5) #gamma1
    par[9] = np.random.uniform(low=0.0641,high=1.5) #gamma2
    par[10] = np.random.uniform(low=0.0641,high=1.5) #gamma3
    par[11] = np.random.uniform(low=0.0641,high=1.5) #gamma4
    par[12] = np.random.uniform(low=50,high=100) #E01
    par[13] = np.random.uniform(low=0,high=20) #I01
    par[14] = np.random.uniform(low=70,high=100) #E02
    par[15] = np.random.uniform(low=0,high=10) #I02
    par[16] = np.random.uniform(low=15,high=50) #E03
    par[17] = np.random.uniform(low=0,high=10) #I03
    par[18] = np.random.uniform(low=0,high=20) #E04
    par[19] = np.random.uniform(low=0,high=10) #I04
    par[20] = np.random.uniform(low=0.1,high=10) #nu1
    par[21] = np.random.uniform(low=0.1,high=10) #nu2
    par[22] = np.random.uniform(low=0.1,high=10) #nu3
    par[23] = np.random.uniform(low=0.1,high=10) #nu4
    
    return par

def euclidean(v1, v2):
    return sum((q1-q2)**2 for q1, q2 in zip(v1, v2))**.5

if __name__=="__main__": 
     #random.seed(64)
     #warnings.filterwarnings('ignore')
     SEIRS = pytwalk.pytwalk(n=NumParams,U=energy,Supp=support)
     SEIRS.Run(T=TotalNumIter,x0=init0(),xp0=init1())



#effective sample size
def ess(data, stepSize=1):
    """ Effective sample size, as computed by BEAST Tracer."""
    samples = len( data )

    assert len( data ) > 1, "no stats for short sequences"

    maxLag = min( samples // 3, 1000 )

    gammaStat = [0, ] * maxLag
    # varGammaStat = [0,]*maxLag

    varStat = 0.0;

    if type( data ) != np.ndarray:
        data = np.array( data )

    normalizedData = data - data.mean()

    for lag in range( maxLag ):
        v1 = normalizedData[:samples - lag]
        v2 = normalizedData[lag:]
        v = v1 * v2
        gammaStat[lag] = sum( v ) / len( v )
        # varGammaStat[lag] = sum(v*v) / len(v)
        # varGammaStat[lag] -= gammaStat[0] ** 2

        # print lag, gammaStat[lag], varGammaStat[lag]

        if lag == 0:
            varStat = gammaStat[0]
        elif lag % 2 == 0:
            s = gammaStat[lag - 1] + gammaStat[lag]
            if s > 0:
                varStat += 2.0 * s
            else:
                break

    # standard error of mean
    # stdErrorOfMean = Math.sqrt(varStat/samples);

    # auto correlation time
    act = stepSize * varStat / gammaStat[0]

    # effective sample size
    ess = (stepSize * samples) / act

    return ess

########################## summary ######################################
def Analysis(x, taus=np.array( [100, 200, 300, 400, 500, 700, 900, 1100]  ), alpha=0.05 ):
    mean=np.mean(x)
    var=np.var(x)
    std=np.std(x)
    eff=ess(x) #effective sample size takes time
    effn=eff/len(x)    #effective sample proportion
    

    acx=np.zeros(len(taus))
    for i in range(len(taus)):
        acx[i]=pd.Series.autocorr(pd.Series(x), lag=taus[i])
        
    qs=mquantiles(x, prob=np.array([alpha/2, alpha*10/2, alpha*10, 1-alpha*10/2, 1-alpha/2]))    
    
    summary= {
        "ESS":eff,
        "ESSn": effn, #eff/n#
        "Mean": mean,
        "Variance": var,
        "SD": std,
        "mVar": var/eff,
        "taus": taus,
        "ac": acx,
        "quantiles "+ str(100*(1-alpha))+ " %": qs,
        }
    
    return summary
#############################################################################

#np.save("/Volumes/F/Hermosillo_four_regions data/twalk_output/toutput_real_data.npy", SEIRS.Output)




SEIRSOutput = np.load("/Volumes/F/Hermosillo_four_regions data/twalk_output/toutput_real_data.npy")

#output_burn =Output[ burnin: ,: ]

#tau=20
#indexes=np.arange(start=0,stop=TotalNumIter-burnin, step=tau)

tau=20
indexes=np.arange(start=0,stop=TotalNumIter-burnin, step=tau)

#toutput = SEIRS.Output[ burnin: ,: ] #output
toutput2 = SEIRSOutput[ burnin: ,: ] #output

toutput = toutput2[indexes, :]


#toutput2 = SEIRSOutput[ burnin:400000 ,: ] #output


fig0= plt.figure()
ax0 = plt.subplot(111)
#SEIRS.Ana(start=burnin)
ax0.plot(-toutput[:,24])
ax0.set_ylabel(r"Log of Objective")
ax0.set_xlabel(r"Iteration")
ax0.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.savefig('/Volumes/F/Hermosillo_four_regions data/figures_twalk/log_objective_fn_real_data.png', dpi = 300 )





#plt.plot(-toutput[:,28])


beta1 = toutput[:,0]
beta2 = toutput[:,1]
beta3 = toutput[:,2]
beta4 = toutput[:,3]

kappa1 = toutput[:,4]
kappa2 = toutput[:,5]
kappa3 = toutput[:,6]
kappa4 = toutput[:,7]

gamma1 = toutput[:,8]
gamma2 = toutput[:,9]
gamma3 = toutput[:,10]
gamma4 = toutput[:,11]

E01 = toutput[:,12]
I01 = toutput[:,13]


E02 = toutput[:,14]
I02 = toutput[:,15]

E03 = toutput[:,16]
I03 = toutput[:,17]


E04 = toutput[:,18]
I04 = toutput[:,19]


nu1 = toutput[:,20]
nu2 = toutput[:,21]
nu3 = toutput[:,22]
nu4 = toutput[:,23]


#Trace plots after burn in
fig, ax = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
#fig.tight_layout()
ax[0][0].plot(beta1)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[0][0].set_ylabel( r'$\beta_{1}$' )
ax[0][0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[0][0].axhline(beta1.mean(), color='k', linestyle='dashed', linewidth=2)
#plt.title( r'$\beta$'+' Traceplot' )
#plt.savefig('SEIRbeta-tp.png', dpi=450)
#plt.show()

ax[0][1].plot(beta2)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[0][1].set_ylabel( r'$\beta_{2}$' )
ax[0][1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[0][1].axhline(beta2.mean(), color='k', linestyle='dashed', linewidth=2)

ax[0][2].plot(beta3)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[0][2].set_ylabel( r'$\beta_{3}$' )
ax[0][2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[0][2].axhline(beta3.mean(), color='k', linestyle='dashed', linewidth=2)

ax[0][3].plot(beta4)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[0][3].set_ylabel( r'$\beta_{4}$' )
ax[0][3].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[0][3].axhline(beta4.mean(), color='k', linestyle='dashed', linewidth=2)

ax[1][0].plot(kappa1)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[1][0].set_ylabel( r'$\kappa_{1}$' )
ax[1][0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[1][0].axhline(kappa1.mean(), color='k', linestyle='dashed', linewidth=2)

ax[1][1].plot(kappa2)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[1][1].set_ylabel( r'$\kappa_{2}$' )
ax[1][1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[1][1].axhline(kappa2.mean(), color='k', linestyle='dashed', linewidth=2)

ax[1][2].plot(kappa3)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[1][2].set_ylabel( r'$\kappa_{3}$' )
ax[1][2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[1][2].axhline(kappa3.mean(), color='k', linestyle='dashed', linewidth=2)

ax[1][3].plot(kappa4)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[1][3].set_ylabel( r'$\kappa_{4}$' )
ax[1][3].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[1][3].axhline(kappa4.mean(), color='k', linestyle='dashed', linewidth=2)

ax[2][0].plot(gamma1)
ax[2][0].set_xlabel( r'$Iteration$' )
ax[2][0].set_ylabel( r'$\gamma_{1}$' )
ax[2][0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[2][0].axhline(gamma1.mean(), color='k', linestyle='dashed', linewidth=2)

ax[2][1].plot(gamma2)
ax[2][1].set_xlabel( r'$Iteration$' )
ax[2][1].set_ylabel( r'$\gamma_{2}$' )
ax[2][1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[2][1].axhline(gamma2.mean(), color='k', linestyle='dashed', linewidth=2)

ax[2][2].plot(gamma3)
ax[2][2].set_xlabel( r'$Iteration$' )
ax[2][2].set_ylabel( r'$\gamma_{3}$' )
ax[2][2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[2][2].axhline(gamma3.mean(), color='k', linestyle='dashed', linewidth=2)

ax[2][3].plot(gamma4)
ax[2][3].set_xlabel( r'$Iteration$' )
ax[2][3].set_ylabel( r'$\gamma_{4}$' )
ax[2][3].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[2][3].axhline(gamma4.mean(), color='k', linestyle='dashed', linewidth=2)
plt.savefig('/Volumes/F/Hermosillo_four_regions data/figures_twalk/traceplots_beta_kappa_gamma_real_data.png', dpi = 300 )


fig, ax = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)

ax[0][0].plot(E01)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[0][0].set_ylabel( r'E01' )
ax[0][0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[0][0].axhline(E01.mean(), color='k', linestyle='dashed', linewidth=2)

ax[0][1].plot(E02)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[0][1].set_ylabel( r'E02' )
ax[0][1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[0][1].axhline(E02.mean(), color='k', linestyle='dashed', linewidth=2)

ax[0][2].plot(E03)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[0][2].set_ylabel( r'E03' )
ax[0][2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[0][2].axhline(E03.mean(), color='k', linestyle='dashed', linewidth=2)

ax[0][3].plot(E04)
#ax[1][2].set_xlabel( r'$Iteration$' )
ax[0][3].set_ylabel( r'E04' )
ax[0][3].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[0][3].axhline(E04.mean(), color='k', linestyle='dashed', linewidth=2)


ax[1][0].plot(I01)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[1][0].set_ylabel( r'I01' )
ax[1][0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[1][0].axhline(I01.mean(), color='k', linestyle='dashed', linewidth=2)


ax[1][1].plot(I02)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[1][1].set_ylabel( r'I02' )
ax[1][1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[1][1].axhline(I02.mean(), color='k', linestyle='dashed', linewidth=2)


ax[1][2].plot(I03)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[1][2].set_ylabel( r'I03' )
ax[1][2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[1][2].axhline(I03.mean(), color='k', linestyle='dashed', linewidth=2)


ax[1][3].plot(I04)
#ax[1][3].set_xlabel( r'$Iteration$' )
ax[1][3].set_ylabel( r'I04' )
ax[1][3].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[1][3].axhline(I04.mean(), color='k', linestyle='dashed', linewidth=2)

ax[2][0].plot(nu1)
ax[2][0].set_xlabel( r'$Iteration$' )
ax[2][0].set_ylabel( r'$\nu_{1}$' )
ax[2][0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[2][0].axhline(nu1.mean(), color='k', linestyle='dashed', linewidth=2)

ax[2][1].plot(nu2)
ax[2][1].set_xlabel( r'$Iteration$' )
ax[2][1].set_ylabel( r'$\nu_{2}$' )
ax[2][1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[2][1].axhline(nu2.mean(), color='k', linestyle='dashed', linewidth=2)

ax[2][2].plot(nu3)
ax[2][2].set_xlabel( r'$Iteration$' )
ax[2][2].set_ylabel( r'$\nu_{3}$' )
ax[2][2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[2][2].axhline(nu3.mean(), color='k', linestyle='dashed', linewidth=2)

ax[2][3].plot(nu4)
ax[2][3].set_xlabel( r'$Iteration$' )
ax[2][3].set_ylabel( r'$\nu_{4}$' )
ax[2][3].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[2][3].axhline(nu4.mean(), color='k', linestyle='dashed', linewidth=2)
plt.savefig('/Volumes/F/Hermosillo_four_regions data/figures_twalk/traceplots_E0_I0_nu_real_data.png', dpi = 300 )

#summary   
sbeta1 = Analysis(beta1)
sbeta2 = Analysis(beta2) 
sbeta3 = Analysis(beta3)   
sbeta4 = Analysis(beta4) 

skappa1 = Analysis(kappa1)
skappa2 = Analysis(kappa2) 
skappa3 = Analysis(kappa3)   
skappa4 = Analysis(kappa4)        
      
sgamma1 = Analysis(gamma1)
sgamma2 = Analysis(gamma2) 
sgamma3 = Analysis(gamma3)   
sgamma4 = Analysis(gamma4)  

sE01 = Analysis(E01) 
sI01 = Analysis(I01)   


sE02 = Analysis(E02) 
sI02 = Analysis(I02)   


sE03 = Analysis(E03) 
sI03 = Analysis(I03)   


sE04 = Analysis(E04) 
sI04 = Analysis(I04)   


snu1 = Analysis(nu1)
snu2 = Analysis(nu2) 
snu3 = Analysis(nu3)   
snu4 = Analysis(nu4)   

###########plotting the ACF for beta################
fig, ax = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)

sm.graphics.tsa.plot_acf(beta1, ax=ax[0][0])
#ax[0][0].set_xlabel('Lags') 
ax[0][0].set_ylabel('ACF') 
ax[0][0].set_title(r"$\beta_{1}$") 

sm.graphics.tsa.plot_acf(beta2, ax=ax[0][1])
#ax[0][1].set_xlabel('Lags') 
#ax[0][1].set_ylabel('ACF') 
ax[0][1].set_title(r"$\beta_{2}$") 

sm.graphics.tsa.plot_acf(beta3, ax=ax[0][2])
#ax[0][2].set_xlabel('Lags') 
#ax[0][2].set_ylabel('ACF') 
ax[0][2].set_title(r"$\beta_{3}$") 

sm.graphics.tsa.plot_acf(beta4, ax=ax[0][3])
#ax[0][3].set_xlabel('Lags') 
#ax[0][3].set_ylabel('ACF') 
ax[0][3].set_title(r"$\beta_{4}$") 

sm.graphics.tsa.plot_acf(kappa1, ax=ax[1][0])
#ax[0][5].set_xlabel('Lags') 
ax[1][0].set_ylabel('ACF') 
ax[1][0].set_title(r"$\kappa_{1}$") 

sm.graphics.tsa.plot_acf(kappa2, ax=ax[1][1])
#ax[0][5].set_xlabel('Lags') 
#ax[0][5].set_ylabel('ACF') 
ax[1][1].set_title(r"$\kappa_{2}$") 

sm.graphics.tsa.plot_acf(kappa3, ax=ax[1][2])
#ax[0][6].set_xlabel('Lags') 
#ax[0][6].set_ylabel('ACF') 
ax[1][2].set_title(r"$\kappa_{3}$") 

sm.graphics.tsa.plot_acf(kappa4, ax=ax[1][3])
#ax[1][0].set_xlabel('Lags') 
ax[1][3].set_ylabel('ACF') 
ax[1][3].set_title(r"$\kappa_{4}$") 

sm.graphics.tsa.plot_acf(gamma1, ax=ax[2][0])
ax[2][0].set_xlabel('Lags') 
ax[2][0].set_ylabel('ACF') 
ax[2][0].set_title(r"$\gamma_{1}$") 

sm.graphics.tsa.plot_acf(gamma2, ax=ax[2][1])
ax[2][1].set_xlabel('Lags') 
#ax[1][2].set_ylabel('ACF') 
ax[2][1].set_title(r"$\gamma_{2}$") 

sm.graphics.tsa.plot_acf(gamma3, ax=ax[2][2])
ax[2][2].set_xlabel('Lags') 
#ax[1][3].set_ylabel('ACF') 
ax[2][2].set_title(r"$\gamma_{3}$") 

sm.graphics.tsa.plot_acf(gamma4, ax=ax[2][3])
ax[2][3].set_xlabel('Lags') 
#ax[1][4].set_ylabel('ACF') 
ax[2][3].set_title(r"$\gamma_{4}$") 
plt.savefig('/Volumes/F/Hermosillo_four_regions data/figures_twalk/ACF_beta_kappa_gamma_real_data.png', dpi = 300 )



fig, ax = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)

sm.graphics.tsa.plot_acf(E01, ax=ax[0][0])
#ax[0][0].set_ylabel('Lags') 
#ax[1][5].set_xlabel('Lags') 
ax[0][0].set_ylabel('ACF') 
ax[0][0].set_title(r"E01") 

sm.graphics.tsa.plot_acf(E02, ax=ax[0][1])
#ax[2][1].set_xlabel('Lags') 
#ax[2][1].set_ylabel('ACF') 
ax[0][1].set_title(r"E02") 

sm.graphics.tsa.plot_acf(E03, ax=ax[0][2])
#ax[2][4].set_xlabel('Lags') 
#ax[0][2].set_ylabel('ACF') 
ax[0][2].set_title(r"E03") 

sm.graphics.tsa.plot_acf(E04, ax=ax[0][3])
#ax[1][2].set_xlabel('Lags') 
#ax[1][2].set_ylabel('ACF') 
ax[0][3].set_title(r"E04") 

sm.graphics.tsa.plot_acf(I01, ax=ax[1][0])
#ax[1][6].set_xlabel('Lags') 
ax[1][0].set_ylabel('ACF') 
ax[1][0].set_title(r"I01") 

sm.graphics.tsa.plot_acf(I02, ax=ax[1][1])
#ax[2][2].set_xlabel('Lags') 
#ax[2][2].set_ylabel('ACF') 
ax[1][1].set_title(r"I02") 


sm.graphics.tsa.plot_acf(I03, ax=ax[1][2])
#ax[1][1].set_xlabel('Lags') 
#ax[2][5].set_ylabel('ACF') 
ax[1][2].set_title(r"I03") 

sm.graphics.tsa.plot_acf(I04, ax=ax[1][3])
#ax[1][3].set_xlabel('Lags') 
#ax[3][1].set_ylabel('ACF') 
ax[1][3].set_title(r"I04") 


sm.graphics.tsa.plot_acf(nu1, ax=ax[2][0])
ax[2][0].set_xlabel('Lags') 
ax[2][0].set_ylabel('ACF') 
ax[2][0].set_title(r"$\nu_{1}$") 

sm.graphics.tsa.plot_acf(nu2, ax=ax[2][1])
ax[2][1].set_xlabel('Lags') 
#ax[3][4].set_ylabel('ACF') 
ax[2][1].set_title(r"$\nu_{2}$") 

sm.graphics.tsa.plot_acf(nu3, ax=ax[2][2])
ax[2][2].set_xlabel('Lags') 
#ax[3][5].set_ylabel('ACF') 
ax[2][2].set_title(r"$\nu_{3}$") 

sm.graphics.tsa.plot_acf(nu4, ax=ax[2][3])
ax[2][3].set_xlabel('Lags') 
#ax[3][6].set_ylabel('ACF') 
ax[2][3].set_title(r"$\nu_{4}$") 
plt.savefig('/Volumes/F/Hermosillo_four_regions data/figures_twalk/ACF_E0_I0_nu_real_data.png', dpi = 300 )




ppc_samples_InsP1 = np.zeros((LastNumIter,len(times_pred)-1))
ppc_samples_InsP2 = np.zeros((LastNumIter,len(times_pred)-1))
ppc_samples_InsP3 = np.zeros((LastNumIter,len(times_pred)-1))
ppc_samples_InsP4 = np.zeros((LastNumIter,len(times_pred)-1))
    
start_date = "2020-02-26"
end_date = "2020-09-06"

pred_date = "2020-10-17"

working_covidcases_zone1 = pd.DataFrame(pd.date_range(start_date, end_date), columns=['FECINISI'])
working_covidcases_pred = pd.DataFrame(pd.date_range(start_date, pred_date), columns=['FECINISI'])

tt = np.array(working_covidcases_zone1['FECINISI'][6:194])

t_pred = np.array(working_covidcases_pred['FECINISI'][6:235])
    
fig, ax = plt.subplots(2, 2, figsize=(18, 15), constrained_layout=True)
qq = toutput[toutput[:,-1].argsort()] # MAP
my_soln_InsP1,my_soln_InsP2, my_soln_InsP3, my_soln_InsP4, = solve_pred(qq[0,:]) # solve for MAP
ax[0][0].plot(t_pred,my_soln_InsP1,'b', label="MAP model incidence")
ax[0][1].plot(t_pred,my_soln_InsP2,'b', label="MAP model incidence")
ax[1][0].plot(t_pred,my_soln_InsP3,'b', label="MAP model incidence")
ax[1][1].plot(t_pred,my_soln_InsP4,'b', label="MAP model incidence")
#plt.savefig(save_results_to + 'MAP.eps')


for k in np.arange(LastNumIter): # last 3000 samples
    ppc_samples_InsP1[k],ppc_samples_InsP2[k],ppc_samples_InsP3[k], ppc_samples_InsP4[k]= \
    solve_pred(toutput[k,:])
#        sample_s, sample_Q,sample_P = solve(sir.Output[-k,:]) 
    ax[0][0].plot(t_pred,ppc_samples_InsP1[k],"#888888", alpha=.25) 
    ax[0][1].plot(t_pred,ppc_samples_InsP2[k],"#888888", alpha=.25) 
    ax[1][0].plot(t_pred,ppc_samples_InsP3[k],"#888888", alpha=.25)
    ax[1][1].plot(t_pred,ppc_samples_InsP4[k],"#888888", alpha=.25)

#ax[0][0].plot(tt,zone1_cases_inflated,'r.', label= "Observed incidence")
#ax[0][1].plot(tt,zone2_cases_inflated,'r.', label= "Observed incidence")
#ax[1][0].plot(tt,zone3_cases_inflated,'r.', label= "Observed incidence")
#ax[1][1].plot(tt,zone4_cases_inflated,'r.', label= "Observed incidence")
ax[0][0].plot(t_pred, zone1_cases_including_pred_data_inflated, "r.", label="Observed daily incidence"  )
ax[0][1].plot(t_pred, zone2_cases_including_pred_data_inflated, "r.", label="Observed daily incidence"  )
ax[1][0].plot(t_pred, zone3_cases_including_pred_data_inflated, "r.", label="Observed daily incidence"  )
ax[1][1].plot(t_pred, zone4_cases_including_pred_data_inflated, "r.", label="Observed daily incidence"  )
ax[0][0].set_title(" Zone 1", weight = "bold")
ax[0][1].set_title(" Zone 2", weight = "bold")
ax[1][0].set_title(" Zone 3", weight = "bold")
ax[1][1].set_title(" Zone 4", weight = "bold")
ax[0][0].axvline(pd.Timestamp("2020-09-07"), color='g', label ="Prediction start date")
ax[0][1].axvline(pd.Timestamp("2020-09-07"), color='g', label ="Prediction start date")
ax[1][0].axvline(pd.Timestamp("2020-09-07"), color='g', label ="Prediction start date")
ax[1][1].axvline(pd.Timestamp("2020-09-07"), color='g', label ="Prediction start date")
ax[0][0].text(pd.Timestamp("2020-08-23"), .85, "2020-09-07", transform=ax[0][0].get_xaxis_transform(), color='k', weight = "bold")
ax[0][1].text(pd.Timestamp("2020-08-23"), .85, "2020-09-07", transform=ax[0][1].get_xaxis_transform(), color='k', weight = "bold")
ax[1][0].text(pd.Timestamp("2020-08-23"), .85, "2020-09-07", transform=ax[1][0].get_xaxis_transform(), color='k', weight = "bold")
ax[1][1].text(pd.Timestamp("2020-08-23"), .85, "2020-09-07", transform=ax[1][1].get_xaxis_transform(), color='k', weight = "bold")
ax[0][0].set_ylabel(r"Incidence", fontsize=15)
ax[0][1].set_ylabel(r"Incidence", fontsize=15)
ax[1][0].set_ylabel(r"Incidence", fontsize=15)
ax[1][1].set_ylabel(r"Incidence", fontsize=15)
ax[0][0].set_xlabel(r"Time", fontsize=15)
ax[0][1].set_xlabel(r"Time", fontsize=15)
ax[1][0].set_xlabel(r"Time", fontsize=15)
ax[1][1].set_xlabel(r"Time", fontsize=15)
ax[0][0].grid(True)
#ax[0][0].legend(loc ="best", title="Predictive check", title_fontsize="xx-large", edgecolor="k", shadow=True, fancybox=True)
ax[0][1].grid(True)
#ax[0][1].legend(loc ="best", title="Predictive check", title_fontsize="xx-large", edgecolor="k", shadow=True, fancybox=True)
ax[1][0].grid(True)
#ax[1][0].legend(loc ="best", title="Predictive check", title_fontsize="xx-large", edgecolor="k", shadow=True, fancybox=True)
ax[1][1].grid(True)
#ax[1][1].legend(loc ="best", title="Predictive check", title_fontsize="xx-large", edgecolor="k", shadow=True, fancybox=True)
#plt.savefig(save_results_to + 'data_vs_samples.eps')
plt.savefig('/Volumes/F/Hermosillo_four_regions data/figures_twalk/data_vs_samples_real_data.png', dpi = 300 )



#fig, ax = plt.subplots(2, 2, figsize=(25, 17))
fig, ax = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
ax[0][0].plot(tt,zone1_cases_inflated,'bo', label= "Observed daily incidence")
ax[0][1].plot(tt,zone2_cases_inflated,'bo', label= "Observed daily incidence")
ax[1][0].plot(tt,zone3_cases_inflated,'bo', label= "Observed daily incidence")
ax[1][1].plot(tt,zone4_cases_inflated,'bo', label= "Observed daily incidence")
ax[0][0].set_title(" Zone 1", fontsize=20, weight = "bold")
ax[0][1].set_title(" Zone 2", fontsize=20, weight = "bold")
ax[1][0].set_title(" Zone 3", fontsize=20, weight = "bold")
ax[1][1].set_title(" Zone 4", fontsize=20, weight = "bold")
ax[0][0].set_ylabel(r"Incidence", fontsize=15)
ax[0][1].set_ylabel(r"Incidence", fontsize=15)
ax[1][0].set_ylabel(r"Incidence", fontsize=15)
ax[1][1].set_ylabel(r"Incidence", fontsize=15)
ax[0][0].set_xlabel(r"Time", fontsize=15)
ax[0][1].set_xlabel(r"Time", fontsize=15)
ax[1][0].set_xlabel(r"Time", fontsize=15)
ax[1][1].set_xlabel(r"Time", fontsize=15)
ax[0][0].grid(True)
#ax[0][0].legend(loc="best", fontsize=25)
ax[0][1].grid(True)
#ax[0][1].legend(loc="best", fontsize=20, edgecolor="k", shadow=True, fancybox=True)
ax[1][0].grid(True)
#ax[1][0].legend(loc="best", fontsize=20, edgecolor="k", shadow=True, fancybox=True)
ax[1][1].grid(True)
#ax[1][1].legend(loc="best", fontsize=20, edgecolor="k", shadow=True, fancybox=True)
#plt.savefig(save_results_to + 'data_vs_samples.eps')
plt.savefig('/Volumes/F/Hermosillo_four_regions data/figures_twalk/observed_incidence_real_data.png', dpi = 300 )



#samples = SEIRS.Output[burnin:,:-1]

#samples = SEIRSOutput[burnin:,:-1]

samples = toutput[:,:-1]

#samples[:,1] *= N
#samples[:,2] *= N
map = qq[0,:-1]

#map1 = qq[0,0:12]
#map[1] *= N
#map[2] *= N    



median_ppc_InsP1 = np.percentile(ppc_samples_InsP1,q=50.0,axis=0)
median_ppc_InsP2 = np.percentile(ppc_samples_InsP2,q=50.0,axis=0)
median_ppc_InsP3 = np.percentile(ppc_samples_InsP3,q=50.0,axis=0)
median_ppc_InsP4 = np.percentile(ppc_samples_InsP4,q=50.0,axis=0)

mean_ppc_InsP1 = np.mean(ppc_samples_InsP1, axis=0)
mean_ppc_InsP2 = np.mean(ppc_samples_InsP2, axis=0)
mean_ppc_InsP3 = np.mean(ppc_samples_InsP3, axis=0)
mean_ppc_InsP4 = np.mean(ppc_samples_InsP4, axis=0)

#median_ppc_I  = np.median(ppc_samples_I,axis=0)
#median_ppc_D  = np.median(ppc_samples_D,axis=0)

CriL_ppc_InsP1 = np.percentile(ppc_samples_InsP1,q=2.5,axis=0)
CriU_ppc_InsP1 = np.percentile(ppc_samples_InsP1,q=97.5,axis=0)

CriL_ppc_InsP2 = np.percentile(ppc_samples_InsP2,q=2.5,axis=0)
CriU_ppc_InsP2 = np.percentile(ppc_samples_InsP2,q=97.5,axis=0)

CriL_ppc_InsP3 = np.percentile(ppc_samples_InsP3,q=2.5,axis=0)
CriU_ppc_InsP3 = np.percentile(ppc_samples_InsP3,q=97.5,axis=0)

CriL_ppc_InsP4 = np.percentile(ppc_samples_InsP4,q=2.5,axis=0)
CriU_ppc_InsP4 = np.percentile(ppc_samples_InsP4,q=97.5,axis=0)


CriL_ppc_InsP1_99p = np.percentile(ppc_samples_InsP1,q=0.5,axis=0)
CriU_ppc_InsP1_99p = np.percentile(ppc_samples_InsP1,q=99.5,axis=0)

CriL_ppc_InsP2_99p = np.percentile(ppc_samples_InsP2,q=0.5,axis=0)
CriU_ppc_InsP2_99p = np.percentile(ppc_samples_InsP2,q=99.5,axis=0)

CriL_ppc_InsP3_99p = np.percentile(ppc_samples_InsP3,q=0.5,axis=0)
CriU_ppc_InsP3_99p = np.percentile(ppc_samples_InsP3,q=99.5,axis=0)

CriL_ppc_InsP4_99p = np.percentile(ppc_samples_InsP4,q=0.5,axis=0)
CriU_ppc_InsP4_99p = np.percentile(ppc_samples_InsP4,q=99.5,axis=0)




median_ppc_beta1   = np.median(samples[:,0],axis=0)
median_ppc_beta2   = np.median(samples[:,1],axis=0)
median_ppc_beta3   = np.median(samples[:,2],axis=0)
median_ppc_beta4   = np.median(samples[:,3],axis=0)
median_ppc_kappa1  = np.median(samples[:,4],axis=0)
median_ppc_kappa2  = np.median(samples[:,5],axis=0)
median_ppc_kappa3  = np.median(samples[:,6],axis=0)
median_ppc_kappa4  = np.median(samples[:,7],axis=0)
median_ppc_gamma1  = np.median(samples[:,8],axis=0)
median_ppc_gamma2  = np.median(samples[:,9],axis=0)
median_ppc_gamma3  = np.median(samples[:,10],axis=0)
median_ppc_gamma4  = np.median(samples[:,11],axis=0)
median_ppc_E01     = np.median(samples[:,12],axis=0)
median_ppc_I01     = np.median(samples[:,13],axis=0)
median_ppc_E02     = np.median(samples[:,14],axis=0)
median_ppc_I02     = np.median(samples[:,15],axis=0)
median_ppc_E03     = np.median(samples[:,16],axis=0)
median_ppc_I03     = np.median(samples[:,17],axis=0)
median_ppc_E04     = np.median(samples[:,18],axis=0)
median_ppc_I04     = np.median(samples[:,19],axis=0)
median_ppc_nu1    = np.median(samples[:,20],axis=0)
median_ppc_nu2    = np.median(samples[:,21],axis=0)
median_ppc_nu3    = np.median(samples[:,22],axis=0)
median_ppc_nu4    = np.median(samples[:,23],axis=0)



CriL_ppc_beta1   = np.percentile(samples[:,0],q=2.5,axis=0)
CriU_ppc_beta1   = np.percentile(samples[:,0],q=97.5,axis=0)

CriL_ppc_beta2   = np.percentile(samples[:,1],q=2.5,axis=0)
CriU_ppc_beta2   = np.percentile(samples[:,1],q=97.5,axis=0)

CriL_ppc_beta3    = np.percentile(samples[:,2],q=2.5,axis=0)
CriU_ppc_beta3    = np.percentile(samples[:,2],q=97.5,axis=0)

CriL_ppc_beta4  = np.percentile(samples[:,3],q=2.5,axis=0)
CriU_ppc_beta4  = np.percentile(samples[:,3],q=97.5,axis=0)



CriL_ppc_kappa1  = np.percentile(samples[:,4],q=2.5,axis=0)
CriU_ppc_kappa1  = np.percentile(samples[:,4],q=97.5,axis=0)

CriL_ppc_kappa2  = np.percentile(samples[:,5],q=2.5,axis=0)
CriU_ppc_kappa2  = np.percentile(samples[:,5],q=97.5,axis=0)

CriL_ppc_kappa3  = np.percentile(samples[:,6],q=2.5,axis=0)
CriU_ppc_kappa3  = np.percentile(samples[:,6],q=97.5,axis=0)

CriL_ppc_kappa4  = np.percentile(samples[:,7],q=2.5,axis=0)
CriU_ppc_kappa4  = np.percentile(samples[:,7],q=97.5,axis=0)



CriL_ppc_gamma1  = np.percentile(samples[:,8],q=2.5,axis=0)
CriU_ppc_gamma1  = np.percentile(samples[:,8],q=97.5,axis=0)

CriL_ppc_gamma2  = np.percentile(samples[:,9],q=2.5,axis=0)
CriU_ppc_gamma2  = np.percentile(samples[:,9],q=97.5,axis=0)

CriL_ppc_gamma3  = np.percentile(samples[:,10],q=2.5,axis=0)
CriU_ppc_gamma3  = np.percentile(samples[:,10],q=97.5,axis=0)

CriL_ppc_gamma4  = np.percentile(samples[:,11],q=2.5,axis=0)
CriU_ppc_gamma4  = np.percentile(samples[:,11],q=97.5,axis=0)



CriL_ppc_E01  = np.percentile(samples[:,12],q=2.5,axis=0)
CriU_ppc_E01  = np.percentile(samples[:,12],q=97.5,axis=0)

CriL_ppc_I01  = np.percentile(samples[:,13],q=2.5,axis=0)
CriU_ppc_I01  = np.percentile(samples[:,13],q=97.5,axis=0)


CriL_ppc_E02  = np.percentile(samples[:,14],q=2.5,axis=0)
CriU_ppc_E02  = np.percentile(samples[:,14],q=97.5,axis=0)

CriL_ppc_I02  = np.percentile(samples[:,15],q=2.5,axis=0)
CriU_ppc_I02  = np.percentile(samples[:,15],q=97.5,axis=0)




CriL_ppc_E03  = np.percentile(samples[:,16],q=2.5,axis=0)
CriU_ppc_E03  = np.percentile(samples[:,16],q=97.5,axis=0)

CriL_ppc_I03  = np.percentile(samples[:,17],q=2.5,axis=0)
CriU_ppc_I03  = np.percentile(samples[:,17],q=97.5,axis=0)



CriL_ppc_E04  = np.percentile(samples[:,18],q=2.5,axis=0)
CriU_ppc_E04  = np.percentile(samples[:,18],q=97.5,axis=0)

CriL_ppc_I04  = np.percentile(samples[:,19],q=2.5,axis=0)
CriU_ppc_I04  = np.percentile(samples[:,19],q=97.5,axis=0)


CriL_ppc_nu1  = np.percentile(samples[:,20],q=2.5,axis=0)
CriU_ppc_nu1  = np.percentile(samples[:,20],q=97.5,axis=0)

CriL_ppc_nu2  = np.percentile(samples[:,21],q=2.5,axis=0)
CriU_ppc_nu2  = np.percentile(samples[:,21],q=97.5,axis=0)

CriL_ppc_nu3  = np.percentile(samples[:,22],q=2.5,axis=0)
CriU_ppc_nu3  = np.percentile(samples[:,22],q=97.5,axis=0)

CriL_ppc_nu4  = np.percentile(samples[:,23],q=2.5,axis=0)
CriU_ppc_nu4  = np.percentile(samples[:,23],q=97.5,axis=0)




print(median_ppc_beta1)
print(median_ppc_beta2)
print(median_ppc_beta3)
print(median_ppc_beta4)

print(median_ppc_kappa1)
print(median_ppc_kappa2)
print(median_ppc_kappa3)
print(median_ppc_kappa4)

print(median_ppc_gamma1)
print(median_ppc_gamma2)
print(median_ppc_gamma3)
print(median_ppc_gamma4)

print(median_ppc_E01)
print(median_ppc_I01)

print(median_ppc_E02)
print(median_ppc_I02)

print(median_ppc_E03)
print(median_ppc_I03)

print(median_ppc_E04)
print(median_ppc_I04)

print(median_ppc_nu1)
print(median_ppc_nu2)
print(median_ppc_nu3)
print(median_ppc_nu4)


fig, ax = plt.subplots(2, 2, figsize=(15,10 ), constrained_layout = True)
#ax2 = plt.subplot(111)
#ax[0][0].plot(tt, zone1_cases_inflated, "r.", label="Observed daily incidence"  )
ax[0][0].plot(t_pred, zone1_cases_including_pred_data_inflated, "r.", label="Observed daily incidence"  )
ax[0][0].plot(t_pred,my_soln_InsP1,color='mediumvioletred', lw=1.5, label = "MAP model incidence")
ax[0][0].plot(t_pred,median_ppc_InsP1,color='mediumblue', lw=1.5, label = "Median model incidence")
ax[0][0].plot(t_pred,mean_ppc_InsP1,color='black', lw=1.5, label = "Mean model incidence")
ax[0][0].fill_between(t_pred, CriL_ppc_InsP1, CriU_ppc_InsP1, color='blue', alpha=0.3, label="95% CI")
ax[0][0].axvline(pd.Timestamp("2020-09-07"), color='g', label ="Prediction start date")
ax[0][0].text(pd.Timestamp("2020-08-23"), .85, "2020-09-07", transform=ax[0][0].get_xaxis_transform(), color='k', weight = "bold")
#ax[0][0].set_xlabel('Time (days)')  # Add an x-label to the axes.
ax[0][0].set_xlabel(r"Time", fontsize=15)  # Add an x-label to the axes.
ax[0][0].set_ylabel('Incidence', fontsize=15)
ax[0][0].grid(True)
#ax[0][0].legend(loc ="best", title="Predictive check", title_fontsize="xx-large", edgecolor="k", shadow=True, fancybox=True)  # Add a legend.
ax[0][0].set_title('Zone 1', weight="bold", fontsize = 20)  



ax[0][1].plot(t_pred, zone2_cases_including_pred_data_inflated, "r.", label="Observed daily incidence"  )
ax[0][1].plot(t_pred,my_soln_InsP2,color='mediumvioletred', lw=1.5, label = "MAP model incidence")
ax[0][1].plot(t_pred,median_ppc_InsP2,color='mediumblue', lw=1.5, label = "Median model incidence")
ax[0][1].plot(t_pred,mean_ppc_InsP2,color='black', lw=1.5, label = "Mean model incidence")
ax[0][1].fill_between(t_pred, CriL_ppc_InsP2, CriU_ppc_InsP2, color='blue', alpha=0.3, label="95% CI")
ax[0][1].axvline(pd.Timestamp("2020-09-07"), color='g', label ="Prediction start date")
ax[0][1].text(pd.Timestamp("2020-08-23"), .85, "2020-09-07", transform=ax[0][1].get_xaxis_transform(), color='k', weight = "bold")
#ax[0][1].set_xlabel('Time (days)')  # Add an x-label to the axes.
#ax[0][1].set_ylabel('count')
ax[0][1].set_xlabel(r"Time", fontsize=15)  # Add an x-label to the axes.
ax[0][1].set_ylabel('Incidence', fontsize=15)
ax[0][1].grid(True)
#ax[0][1].legend(loc ="best", title="Predictive check", title_fontsize="xx-large", edgecolor="k", shadow=True, fancybox=True)  # Add a legend.
ax[0][1].set_title('Zone 2', weight="bold", fontsize = 20)  


ax[1][0].plot(t_pred, zone3_cases_including_pred_data_inflated, "r.", label="Observed daily incidence"  )
ax[1][0].plot(t_pred,my_soln_InsP3,color='mediumvioletred', lw=1.5, label = "MAP model incidence")
ax[1][0].plot(t_pred,median_ppc_InsP3,color='mediumblue', lw=1.5, label = "Median model incidence")
ax[1][0].plot(t_pred,mean_ppc_InsP3,color='black', lw=1.5, label = "Mean model incidence")
ax[1][0].fill_between(t_pred, CriL_ppc_InsP3, CriU_ppc_InsP3, color='blue', alpha=0.3, label="95% CI")
ax[1][0].axvline(pd.Timestamp("2020-09-07"), color='g', label ="Prediction start date")
ax[1][0].text(pd.Timestamp("2020-08-23"), .85, "2020-09-07", transform=ax[1][0].get_xaxis_transform(), color='k', weight = "bold")
ax[1][0].set_xlabel(r"Time", fontsize=12)  # Add an x-label to the axes.
ax[1][0].set_ylabel('Incidence', fontsize=15)
ax[1][0].grid(True)
#ax[1][0].legend(loc ="best", title="Predictive check", title_fontsize="xx-large", edgecolor="k", shadow=True, fancybox=True) # Add a legend.
ax[1][0].set_title('Zone 3', weight="bold", fontsize = 20)  

ax[1][1].plot(t_pred, zone4_cases_including_pred_data_inflated, "r.", label="Observed daily incidence"  )
ax[1][1].plot(t_pred,my_soln_InsP4,color='mediumvioletred', lw=1.5, label = "MAP model incidence")
ax[1][1].plot(t_pred,median_ppc_InsP4,color='mediumblue', lw=1.5, label = "Median model incidence")
ax[1][1].plot(t_pred,mean_ppc_InsP4,color='black', lw=1.5, label = "Mean model incidence")
ax[1][1].fill_between(t_pred, CriL_ppc_InsP4, CriU_ppc_InsP4, color='blue', alpha=0.3, label="95% CI")
ax[1][1].axvline(pd.Timestamp("2020-09-07"), color='g', label ="Prediction start date")
ax[1][1].text(pd.Timestamp("2020-08-23"), .85, "2020-09-07", transform=ax[1][1].get_xaxis_transform(), color='k', weight = "bold")
ax[1][1].set_xlabel(r"Time", fontsize=15)  # Add an x-label to the axes.
ax[1][1].grid(True)
#ax[1][1].legend(loc ="best", title="Predictive check", title_fontsize="xx-large", edgecolor="k", shadow=True, fancybox=True)# Add a legend.
ax[1][1].set_title('Zone 4', weight="bold", fontsize = 20)  
ax[1][1].set_ylabel('Incidence', fontsize=15)
#plt.savefig(save_results_to + 'BandsPrediction_InsP1_2_3_4.pdf')
handles, labels = ax[0][0].get_legend_handles_labels()
#fig.legend(handles, labels, title="Predictive check",title_fontsize="xx-large", bbox_to_anchor=(0.73, 0.90, 0, -0.3),loc=4, ncol=5,fancybox=True, bbox_transform=fig.transFigure)
plt.savefig('/Volumes/F/Hermosillo_four_regions data/figures_twalk/BandsPrediction_InsP1_2_3_4_real_data.png', dpi = 300 )



fig, ax = plt.subplots(2, 2, figsize=(20,17 ))
#ax2 = plt.subplot(111)
#ax[0][0].plot(tt, zone1_cases_inflated, "r.", label="Observed daily incidence"  )
ax[0][0].plot(t_pred, zone1_cases_including_pred_data_inflated, "r.", label="Observed daily incidence"  )
ax[0][0].plot(t_pred,my_soln_InsP1,color='mediumvioletred', lw=1.5, label = "MAP model incidence")
ax[0][0].plot(t_pred,median_ppc_InsP1,color='mediumblue', lw=1.5, label = "Median model incidence")
ax[0][0].plot(t_pred,mean_ppc_InsP1,color='black', lw=1.5, label = "Mean model incidence")
ax[0][0].fill_between(t_pred, CriL_ppc_InsP1_99p, CriU_ppc_InsP1_99p, color='blue', alpha=0.3, label="99% CI")
ax[0][0].axvline(pd.Timestamp("2020-09-07"), color='g', label ="Prediction start date")
ax[0][0].text(pd.Timestamp("2020-08-23"), .85, "2020-09-07", transform=ax[0][0].get_xaxis_transform(), color='k', weight = "bold")
ax[0][0].set_xlabel(r"Time", fontsize=15)  # Add an x-label to the axes.
#ax[0][0].set_xlabel('Time (days)')  # Add an x-label to the axes.
ax[0][0].set_ylabel('Incidence', fontsize=15)
ax[0][0].grid(True)
#ax[0][0].legend(loc ="best", title="Predictive check", title_fontsize="xx-large", edgecolor="k", shadow=True, fancybox=True) # Add a legend.
ax[0][0].set_title('Zone 1', weight="bold", fontsize = 20)  


#ax[0][1].plot(tt, zone2_cases_inflated, "r.", label="Observed daily incidence"  )
ax[0][1].plot(t_pred, zone2_cases_including_pred_data_inflated, "r.", label="Observed daily incidence"  )
ax[0][1].plot(t_pred,my_soln_InsP2,color='mediumvioletred', lw=1.5, label = "MAP model incidence")
ax[0][1].plot(t_pred,median_ppc_InsP2,color='mediumblue', lw=1.5, label = "Median model incidence")
ax[0][1].plot(t_pred,mean_ppc_InsP2,color='black', lw=1.5, label = "Mean model incidence")
ax[0][1].fill_between(t_pred, CriL_ppc_InsP2_99p, CriU_ppc_InsP2_99p, color='blue', alpha=0.3, label="99% CI")
ax[0][1].axvline(pd.Timestamp("2020-09-07"), color='g', label ="Prediction start date")
ax[0][1].text(pd.Timestamp("2020-08-23"), .85, "2020-09-07", transform=ax[0][1].get_xaxis_transform(), color='k', weight = "bold")
#ax[0][1].set_xlabel('Time (days)')  # Add an x-label to the axes.
ax[0][1].set_xlabel(r"Time", fontsize=15)  # Add an x-label to the axes.
ax[0][1].set_ylabel('Incidence', fontsize=15)
ax[0][1].grid(True)
#ax[0][1].legend(loc ="best", title="Predictive check", title_fontsize="xx-large", edgecolor="k", shadow=True, fancybox=True)  # Add a legend.
ax[0][1].set_title('Zone 2', weight="bold", fontsize = 20)  

#ax[1][0].plot(tt, zone3_cases_inflated, "r.",label="Observed daily incidence"  )
ax[1][0].plot(t_pred, zone3_cases_including_pred_data_inflated, "r.", label="Observed daily incidence"  )
#ax[1][0].stem(ttime2, zone3_cases_sim, linefmt='tomato', markerfmt=" ",basefmt=" ",label="zone 3 daily incidence"  )
#    ax2.plot(ttime,Sick,linestyle='dashed', marker='o', color='mediumblue',label="Confirmed Cases")
ax[1][0].plot(t_pred,my_soln_InsP3,color='mediumvioletred', lw=1.5, label = "MAP model incidence")
ax[1][0].plot(t_pred,median_ppc_InsP3,color='mediumblue', lw=1.5, label = "Median model incidence")
ax[1][0].plot(t_pred,mean_ppc_InsP3,color='black', lw=1.5, label = "Mean model incidence")
ax[1][0].fill_between(t_pred, CriL_ppc_InsP3_99p, CriU_ppc_InsP3_99p, color='blue', alpha=0.3, label="99% CI")
ax[1][0].axvline(pd.Timestamp("2020-09-07"), color='g', label ="Prediction start date")
ax[1][0].text(pd.Timestamp("2020-08-23"), .85, "2020-09-07", transform=ax[1][0].get_xaxis_transform(), color='k', weight = "bold")
ax[1][0].set_xlabel(r"Time", fontsize=15)  # Add an x-label to the axes.
ax[1][0].set_ylabel('Incidence', fontsize=15)
ax[1][0].grid(True)
#ax[1][0].legend(loc ="best", title="Predictive check", title_fontsize="xx-large", edgecolor="k", shadow=True, fancybox=True) # Add a legend.
ax[1][0].set_title('Zone 3', weight="bold", fontsize = 20)  

#ax[1][1].plot(tt, zone4_cases_inflated,"r.",label="Observed daily incidence"  )
ax[1][1].plot(t_pred, zone4_cases_including_pred_data_inflated, "r.", label="Observed daily incidence"  )
ax[1][1].plot(t_pred,my_soln_InsP4,color='mediumvioletred', lw=1.5, label = "MAP model incidence")
ax[1][1].plot(t_pred,median_ppc_InsP4,color='mediumblue', lw=1.5, label = "Median model incidence")
ax[1][1].plot(t_pred,mean_ppc_InsP4,color='black', lw=1.5, label = "Mean model incidence")
ax[1][1].fill_between(t_pred, CriL_ppc_InsP4_99p, CriU_ppc_InsP4_99p, color='blue', alpha=0.3, label="99% CI")
ax[1][1].axvline(pd.Timestamp("2020-09-07"), color='g', label ="Prediction start date")
ax[1][1].text(pd.Timestamp("2020-08-23"), .85, "2020-09-07", transform=ax[1][1].get_xaxis_transform(), color='k', weight = "bold")
ax[1][1].set_xlabel(r"Time", fontsize=15)  # Add an x-label to the axes.
ax[1][1].grid(True)
ax[1][1].set_ylabel('Incidence', fontsize=15)
#ax[1][1].legend(loc ="best", title="Predictive check", title_fontsize="xx-large", edgecolor="k", shadow=True, fancybox=True)  # Add a legend.
ax[1][1].set_title('Zone 4', weight="bold", fontsize = 20)  
#plt.savefig(save_results_to + 'BandsPrediction_InsP1_2_3_4.pdf')
handles, labels = ax[0][0].get_legend_handles_labels()
fig.legend(handles, labels, title="Predictive check",title_fontsize="xx-large", bbox_to_anchor=(0.73, 0.90, 0, -0.3),loc=4, ncol=5,fancybox=True, bbox_transform=fig.transFigure)
plt.savefig('/Volumes/F/Hermosillo_four_regions data/figures_twalk/BandsPrediction_InsP1_2_3_4_99p_real_data.png', dpi = 300 )


""" define a function that computes the proportion of prediction incidence covered by 95% CI """

def count_observed_in_interval(minvec, obsvec, maxvec):
    n = len(obsvec) 
    count = 0
    for i in range(len(obsvec)):
        if obsvec[i] > minvec[i] and obsvec[i]<maxvec[i]:
            count += 1
    print("prop_observed_within_interval: ", count/n)


""" compute lower and upper CIs for zone incidences"""
zone1_lower_pred_incidence = CriL_ppc_InsP1[193:229]
zone1_upper_pred_incidence = CriU_ppc_InsP1[193:229]
zone2_lower_pred_incidence = CriL_ppc_InsP2[193:229]
zone2_upper_pred_incidence = CriU_ppc_InsP2[193:229]
zone3_lower_pred_incidence = CriL_ppc_InsP3[193:229]
zone3_upper_pred_incidence = CriU_ppc_InsP3[193:229]
zone4_lower_pred_incidence = CriL_ppc_InsP4[193:229]
zone4_upper_pred_incidence = CriU_ppc_InsP4[193:229]

""" obtain observed zonal incidences """
observed_zone1_prediction_incidence = zone1_cases_including_pred_data_inflated[193:229]
observed_zone2_prediction_incidence = zone2_cases_including_pred_data_inflated[193:229]
observed_zone3_prediction_incidence = zone3_cases_including_pred_data_inflated[193:229]
observed_zone4_prediction_incidence = zone4_cases_including_pred_data_inflated[193:229]

""" compute zonal observed incidences covered by 95% CIs"""
zone1_prop_covered = count_observed_in_interval(zone1_lower_pred_incidence,observed_zone1_prediction_incidence,zone1_upper_pred_incidence )
zone2_prop_covered = count_observed_in_interval(zone2_lower_pred_incidence,observed_zone2_prediction_incidence,zone2_upper_pred_incidence )
zone3_prop_covered = count_observed_in_interval(zone3_lower_pred_incidence,observed_zone3_prediction_incidence,zone3_upper_pred_incidence )
zone4_prop_covered = count_observed_in_interval(zone4_lower_pred_incidence,observed_zone4_prediction_incidence,zone4_upper_pred_incidence )



"""plotting the prior and posterior distributions""" 

def lognormMu(x, mu, s):
    tempX = x / np.exp(mu)
    return ss.lognorm.pdf(tempX, s)


s1 = 0.2803
mu1 = - 0.0529
s2 = 0.2803
mu2 = - 0.0975
s3 = 0.2803
mu3 = - 0.8562
s4 = 0.2803
mu4 = 0.2583
u=np.linspace(0.,3,200)
t=np.linspace(0.,1.5,200)
x=np.linspace(0.,2,200)
r=np.linspace(0.,0.75,200)
v=np.linspace(0.,2.5,200)

y = ss.gamma.pdf(r, a=4.5264, scale=1/19.1006) #gamma distribution for kappa parameters
z = ss.gamma.pdf(v, a=1.9826, scale=1/3.6943) # gamma distribution for gamma parameters


fig, ax = plt.subplots(3,4, figsize=(15, 10), constrained_layout=True)

sns.kdeplot(data=samples[:,0].squeeze(), ax=ax[0,0], color="blue", fill=True, label = "posterior distribution")
ax[0,0].plot(x,lognormMu(x, mu1, s1), label="prior distribution", color="red")
ax[0,0].set_xlabel(r"$\beta_{1}$")

ax[0,1].plot(x,lognormMu(x, mu2, s2), label="prior distribution", color="red")
sns.kdeplot(data=samples[:,1].squeeze(), ax=ax[0,1], color="blue", fill=True, label = "posterior distribution")
ax[0,1].set_xlabel(r"$\beta_{2}$")

ax[0,2].plot(t,lognormMu(t, mu3, s3), label="prior distribution", color="red")
sns.kdeplot(data=samples[:,2].squeeze(), ax=ax[0,2], color="blue", fill=True, label = "posterior distribution")
ax[0,2].set_xlabel(r"$\beta_{3}$")

ax[0,3].plot(u,lognormMu(u, mu4, s4), label="prior distribution", color="red")
sns.kdeplot(data=samples[:,3].squeeze(), ax=ax[0,3], color="blue", fill=True, label = "posterior distribution")
ax[0,3].set_xlabel(r"$\beta_{4}$")

#create plot for kappa
ax[1,0].plot(r, y, label="prior distribution", color="red")
sns.kdeplot(data=samples[:,4].squeeze(), ax=ax[1,0], color="blue", fill=True, label = "posterior distribution")
ax[1,0].set_xlabel(r"$\kappa_{1}$")

#create plot for kappa
ax[1,1].plot(r, y, label="prior distribution", color="red")
sns.kdeplot(data=samples[:,5].squeeze(), ax=ax[1,1], color="blue", fill=True, label = "posterior distribution")
ax[1,1].set_xlabel(r"$\kappa_{2}$")

#create plot for kappa
ax[1,2].plot(r, y, label="prior distribution", color="red")
sns.kdeplot(data=samples[:,6].squeeze(), ax=ax[1,2], color="blue", fill=True, label = "posterior distribution")
ax[1,2].set_xlabel(r"$\kappa_{3}$")


#create plot for kappa
ax[1,3].plot(r, y, label="prior distribution", color="red")
sns.kdeplot(data=samples[:,7].squeeze(), ax=ax[1,3], color="blue", fill=True, label = "posterior distribution")
ax[1,3].set_xlabel(r"$\kappa_{4}$")

#create plot of gamma parameters
ax[2,0].plot(v, z, label="prior distribution", color="red")
sns.kdeplot(data=samples[:,8].squeeze(), ax=ax[2,0], color="blue", fill=True, label = "posterior distribution")
ax[2,0].set_xlabel(r"$\gamma_{1}$")

#create plot of gamma parameters
ax[2,1].plot(v, z, label="prior distribution", color="red")
sns.kdeplot(data=samples[:,9].squeeze(), ax=ax[2,1], color="blue", fill=True, label = "posterior distribution")
ax[2,1].set_xlabel(r"$\gamma_{2}$")

#create plot of gamma parameters
ax[2,2].plot(v, z, label="prior distribution", color="red")
sns.kdeplot(data=samples[:,10].squeeze(), ax=ax[2,2], color="blue", fill=True, label = "posterior distribution")
ax[2,2].set_xlabel(r"$\gamma_{3}$")

#create plot of gamma parameters
ax[2,3].plot(v, z, label="prior distribution", color="red")
sns.kdeplot(data=samples[:,11].squeeze(), ax=ax[2,3], color="blue", fill=True, label = "posterior distribution")
ax[2,3].set_xlabel(r"$\gamma_{4}$")
handles, labels = ax[0][0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.78, 1.0, 0, -0.4), fontsize=20, loc=4, ncol=4, bbox_transform=fig.transFigure)

plt.savefig('/Volumes/F/Hermosillo_four_regions data/figures_twalk/prior_vs_posterior_distributions.png', dpi = 300, bbox_inches='tight' )


# Define the borders
b1     = samples[:,0]
b2     = samples[:,1]
b3     = samples[:,2]
b4     = samples[:,3]
k1     = samples[:,4]
k2     = samples[:,5]
k3     = samples[:,6]
k4     = samples[:,7]
g1     = samples[:,8]
g2     = samples[:,9]
g3     = samples[:,10]
g4     = samples[:,11]
e01    = samples[:,12]
i01    = samples[:,13]
e02    = samples[:,14]
i02    = samples[:,15]
e03    = samples[:,16]
i03    = samples[:,17]
e04    = samples[:,18]
i04    = samples[:,19]
nu_1  = samples[:,20]
nu_2  = samples[:,21]
nu_3  = samples[:,22]
nu_4  = samples[:,23]


fig, ax = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
sns.kdeplot(data=b1, ax=ax[0][0])
x_values_beta1 = [CriL_ppc_beta1, CriU_ppc_beta1]
y_values_beta1 = [0, 0]
ax[0][0].plot(x_values_beta1, y_values_beta1,color='r', lw=6, label="95% CI") 
ax[0][0].axvline(b1.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label = "mean") 
#ax[0][0].axvline(x=1.3, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value") 
ax[0][0].axvline(x=1.0950, color='g', linestyle='dashed', linewidth=4, alpha=1,label = "GA")   
ax[0][0].set_xlabel(r"$\beta_{1}$")

sns.kdeplot(data=b2, ax=ax[0][1])
x_values_beta2 = [CriL_ppc_beta2, CriU_ppc_beta2]
y_values_beta2 = [0, 0]
ax[0][1].plot(x_values_beta2, y_values_beta2,color='r', lw=6, label ="95% CI")  
ax[0][1].axvline(b2.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label="mean") 
#ax[0][1].axvline(x=1.4, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value") 
ax[0][1].axvline(x=1.0473 , color='g', linestyle='dashed', linewidth=4,alpha=1, label = "GA")   
ax[0][1].set_xlabel(r"$\beta_{2}$")

sns.kdeplot(data=b3, ax=ax[0][2])
x_values_beta3 = [CriL_ppc_beta3, CriU_ppc_beta3]
y_values_beta3 = [0, 0]
ax[0][2].plot(x_values_beta3, y_values_beta3,color='r', lw=6, label = "95% CI") 
ax[0][2].axvline(b3.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label ="mean")  
#ax[0][2].axvline(x=0.95, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value")  
ax[0][2].axvline(x=0.4904, color='g', linestyle='dashed', linewidth=4, alpha=1,label = "GA")  
ax[0][2].set_xlabel(r"$\beta_{3}$")
   
sns.kdeplot(data=b4, ax=ax[0][3])
x_values_beta4 = [CriL_ppc_beta4, CriU_ppc_beta4]
y_values_beta4 = [0, 0]
ax[0][3].plot(x_values_beta4, y_values_beta4,color='r', lw=6, label = "95% CI")  
ax[0][3].axvline(b4.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label = "mean") 
#ax[0][3].axvline(x=0.80, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value") 
ax[0][3].axvline(x=1.4951, color='g', linestyle='dashed', linewidth=4, alpha=1,label = "GA")  
ax[0][3].set_xlabel(r"$\beta_{4}$") 

sns.kdeplot(data=k1, ax=ax[1][0])
x_values_kappa1 = [CriL_ppc_kappa1, CriU_ppc_kappa1]
y_values_kappa1 = [0, 0]
ax[1][0].plot(x_values_kappa1, y_values_kappa1,color='r', lw=6, label="95% CI") 
ax[1][0].axvline(k1.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label = "mean") 
#ax[0][4].axvline(x=1/12, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value")
ax[1][0].axvline(x=0.3, color='g', linestyle='dashed', linewidth=4,alpha=1, label = "GA")     
ax[1][0].set_xlabel(r"$\kappa_{1}$")

sns.kdeplot(data=k2, ax=ax[1][1])
x_values_kappa2 = [CriL_ppc_kappa2, CriU_ppc_kappa2]
y_values_kappa2 = [0, 0]
ax[1][1].plot(x_values_kappa2, y_values_kappa2,color='r', lw=6, label="95% CI")   
ax[1][1].axvline(k2.mean(), color='k', linestyle='dashed', linewidth=4, alpha=1, label = "mean") 
#ax[0][5].axvline(x=1/14, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value") 
ax[1][1].axvline(x=0.3, color='g', linestyle='dashed', linewidth=4, alpha=1, label = "GA")   
ax[1][1].set_xlabel(r"$\kappa_{2}$")

sns.kdeplot(data=k3, ax=ax[1][2])
x_values_kappa3 = [CriL_ppc_kappa3, CriU_ppc_kappa3]
y_values_kappa3 = [0, 0]
ax[1][2].plot(x_values_kappa3, y_values_kappa3,color='r', lw=6, label="95% CI")  
ax[1][2].axvline(k3.mean(), color='k', linestyle='dashed', linewidth=4, alpha=1, label = "mean") 
#ax[0][6].axvline(x=2/27, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value")
ax[1][2].axvline(x=0.3, color='g', linestyle='dashed', linewidth=4,alpha=1, label = "GA")   
ax[1][2].set_xlabel(r"$\kappa_{3}$")  

sns.kdeplot(data=k4, ax=ax[1][3])
x_values_kappa4 = [CriL_ppc_kappa4, CriU_ppc_kappa4]
y_values_kappa4 = [0, 0]
ax[1][3].plot(x_values_kappa4, y_values_kappa4,color='r', lw=6, label="95% CI")   
ax[1][3].axvline(k4.mean(), color='k', linestyle='dashed', linewidth=4, alpha=1,label = "mean") 
#ax[1][0].axvline(x=1/10, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value") 
ax[1][3].axvline(x=0.3, color='g', linestyle='dashed', linewidth=4,alpha=1, label = "GA")   
ax[1][3].set_xlabel(r"$\kappa_{4}$")

sns.kdeplot(data=g1, ax=ax[2][0] )
x_values_gamma1 = [CriL_ppc_gamma1, CriU_ppc_gamma1]
y_values_gamma1 = [0, 0]
ax[2][0].plot(x_values_gamma1, y_values_gamma1,color='r', lw=6, label="95% CI") 
ax[2][0].axvline(g1.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label = "mean") 
#ax[1][1].axvline(x=1/6, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value") 
ax[2][0].axvline(x=1, color='g', linestyle='dashed', linewidth=4, alpha=1, label = "GA")   
ax[2][0].set_xlabel(r"$\gamma_{1}$")  

sns.kdeplot(data=g2, ax=ax[2][1])
x_values_gamma2 = [CriL_ppc_gamma2, CriU_ppc_gamma2]
y_values_gamma2 = [0, 0]
ax[2][1].plot(x_values_gamma2, y_values_gamma2,color='r', lw=6, label="95% CI") 
ax[2][1].axvline(g2.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label = "mean")
#ax[1][2].axvline(x=1/7, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value")
ax[2][1].axvline(x=1, color='g', linestyle='dashed', linewidth=4, alpha=1,label = "GA")     
ax[2][1].set_xlabel(r"$\gamma_{2}$")  

sns.kdeplot(data=g3, ax=ax[2][2])
x_values_gamma3 = [CriL_ppc_gamma3, CriU_ppc_gamma3]
y_values_gamma3 = [0, 0]
ax[2][2].plot(x_values_gamma3, y_values_gamma3,color='r', lw=6, label="95% CI") 
ax[2][2].axvline(g3.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label = "mean") 
#ax[1][3].axvline(x=2/11, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value")
ax[2][2].axvline(x=1.0, color='g', linestyle='dashed', linewidth=4,alpha=1, label = "GA")    
ax[2][2].set_xlabel(r"$\gamma_{3}$")    

sns.kdeplot(data=g4,ax=ax[2][3] )
x_values_gamma4 = [CriL_ppc_gamma4, CriU_ppc_gamma4]
y_values_gamma4 = [0, 0]
ax[2][3].plot(x_values_gamma4, y_values_gamma4,color='r', lw=6, label="95% CI") 
ax[2][3].axvline(g4.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label = "mean") 
#ax[1][4].axvline(x=1/5, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value") 
ax[2][3].axvline(x=0.7853, color='g', linestyle='dashed', linewidth=4,alpha=1, label = "GA")     
ax[2][3].set_xlabel(r"$\gamma_{4}$")  
handles, labels = ax[0][0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.65, 1.0, 0, -0.4), fontsize=20, loc=4, ncol=4, bbox_transform=fig.transFigure)
#ax.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(-2.0, -0.12), ncol=4)  
plt.savefig('/Volumes/F/Hermosillo_four_regions data/figures_twalk/BayesianInterval_beta_kappa_gamma_real_data.png', dpi = 300, bbox_inches='tight' )



fig, ax = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
sns.kdeplot(data=e01, ax=ax[0][0])
x_values_E01 = [CriL_ppc_E01, CriU_ppc_E01]
y_values_E01 = [0, 0]
ax[0][0].plot(x_values_E01, y_values_E01,color='r', lw=6, label="95% CI") 
ax[0][0].axvline(e01.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label = "mean")
#ax[1][5].axvline(x=10, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value") 
ax[0][0].axvline(x=58.9925, color='g', linestyle='dashed', linewidth=4, alpha=1, label = "GA")   
ax[0][0].set_xlabel(r"E01")   

sns.kdeplot(data=e02,ax=ax[0][1] )
x_values_E02 = [CriL_ppc_E02, CriU_ppc_E02]
y_values_E02 = [0, 0]
ax[0][1].plot(x_values_E02, y_values_E02,color='r', lw=6, label="95% CI") 
ax[0][1].axvline(e02.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label = "mean") 
#ax[2][1].axvline(x=20, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value")
ax[0][1].axvline(x=78.1213, color='g', linestyle='dashed', linewidth=4, alpha=1, label = "GA")    
ax[0][1].set_xlabel(r"E02") 

sns.kdeplot(data=e03, ax=ax[0][2])
x_values_E03 = [CriL_ppc_E03, CriU_ppc_E03]
y_values_E03 = [0, 0]
ax[0][2].plot(x_values_E03, y_values_E03,color='r', lw=6, label="95% CI") 
ax[0][2].axvline(e03.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label = "mean") 
#ax[2][4].axvline(x=5, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value")
ax[0][2].axvline(x=22.9404, color='g', linestyle='dashed', linewidth=4,alpha=1, label = "GA")     
ax[0][2].set_xlabel(r"Y02")   

sns.kdeplot(data=e04, ax=ax[0][3])
x_values_E04 = [CriL_ppc_E04, CriU_ppc_E04]
y_values_E04 = [0, 0]
ax[0][3].plot(x_values_E04, y_values_E04,color='r', lw=6, label="95% CI")
ax[0][3].axvline(e04.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label = "mean") 
#ax[3][0].axvline(x=5, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value") 
ax[0][3].axvline(x=4.4192, color='g', linestyle='dashed', linewidth=4, alpha=1,label = "GA")     
ax[0][3].set_xlabel(r"E04")    

  
sns.kdeplot(data=i01, ax=ax[1][0])
x_values_I01 = [CriL_ppc_I01, CriU_ppc_I01]
y_values_I01 = [0, 0]
ax[1][0].plot(x_values_I01, y_values_I01,color='r', lw=6, label="95% CI")
ax[1][0].axvline(i01.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label = "mean") 
#ax[1][6].axvline(x=1, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value") 
ax[1][0].axvline(x=0, color='g', linestyle='dashed', linewidth=4,alpha=1, label = "GA")     
ax[1][0].set_xlabel(r"I01")   


sns.kdeplot(data=i02, ax=ax[1][1])
x_values_I02 = [CriL_ppc_I02, CriU_ppc_I02]
y_values_I02 = [0, 0]
ax[1][1].plot(x_values_I02, y_values_I02,color='r', lw=6, label="95% CI") 
ax[1][1].axvline(i02.mean(), color='k', linestyle='dashed', linewidth=4, alpha=1, label = "mean")
#ax[2][2].axvline(x=2, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value")
ax[1][1].axvline(x=0, color='g', linestyle='dashed', linewidth=4, alpha=1, label = "GA")     
ax[1][1].set_xlabel(r"I02")   


sns.kdeplot(data=i03, ax=ax[1][2])
x_values_I03 = [CriL_ppc_I03, CriU_ppc_I03]
y_values_I03 = [0, 0]
ax[1][2].plot(x_values_I03, y_values_I03,color='r', lw=6, label="95% CI") 
ax[1][2].axvline(i03.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label = "mean")
#ax[2][5].axvline(x=0, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value") 
ax[1][2].axvline(x=0, color='g', linestyle='dashed', linewidth=4,alpha=1, label = "GA")      
ax[1][2].set_xlabel(r"I03")   
 

sns.kdeplot(data=i04, ax=ax[1][3])
x_values_I04 = [CriL_ppc_I04, CriU_ppc_I04]
y_values_I04 = [0, 0]
ax[1][3].plot(x_values_I04, y_values_I04,color='r', lw=6, label="95% CI") 
ax[1][3].axvline(i04.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label = "mean") 
#ax[3][1].axvline(x=0, color='b', linestyle='dashed', linewidth=2, label = "real(simulation) value") 
ax[1][3].axvline(x=0, color='g', linestyle='dashed', linewidth=4, alpha=1,label = "GA")   
ax[1][3].set_xlabel(r"I04")    


sns.kdeplot(data=nu_1, ax=ax[2][0])
x_values_nu1 = [CriL_ppc_nu1, CriU_ppc_nu1]
y_values_phi1 = [0, 0]
ax[2][0].plot(x_values_nu1, y_values_phi1,color='r', lw=6, label="95% CI")
ax[2][0].axvline(nu_1.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label = "mean")     
ax[2][0].set_xlabel(r"$\nu_{1}$")    

sns.kdeplot(data=nu_2, ax=ax[2][1])
x_values_nu_2 = [CriL_ppc_nu2, CriU_ppc_nu2]
y_values_nu_2 = [0, 0]
ax[2][1].plot(x_values_nu_2, y_values_nu_2,color='r', lw=6, label="95% CI")
ax[2][1].axvline(nu_2.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label = "mean")     
ax[2][1].set_xlabel(r"$\nu_{2}$")    

sns.kdeplot(data=nu_3, ax=ax[2][2])
x_values_nu3 = [CriL_ppc_nu3, CriU_ppc_nu3]
y_values_nu3 = [0, 0]
ax[2][2].plot(x_values_nu3, y_values_nu3,color='r', lw=6, label="95% CI") 
ax[2][2].axvline(nu_3.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label = "mean")   
ax[2][2].set_xlabel(r"$\nu_{3}$")    

sns.kdeplot(data=nu_4, ax=ax[2][3])
x_values_nu4 = [CriL_ppc_nu4, CriU_ppc_nu4]
y_values_nu4 = [0, 0]
ax[2][3].plot(x_values_nu4, y_values_nu4,color='r', lw=6, label="95% CI") 
ax[2][3].axvline(nu_4.mean(), color='k', linestyle='dashed', linewidth=4,alpha=1, label = "mean")  
ax[2][3].set_xlabel(r"$\nu_{4}$")   

handles, labels = ax[0][0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.65, 1.0, 0, -0.4),fontsize=20,loc=4, ncol=4, bbox_transform=fig.transFigure)
#ax.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(-2.0, -0.12), ncol=4)  
plt.savefig('/Volumes/F/Hermosillo_four_regions data/figures_twalk/BayesianInterval_E0_I0_nu_real_data.png', dpi = 300, bbox_inches='tight' )


    
print('Norm Square of Patch_1 data =')
print(euclidean(median_ppc_InsP1, zone1_cases_inflated)) 
print('Norm Square of Patch_2 data =')
print(euclidean(median_ppc_InsP2, zone2_cases_inflated))
print('Norm Square of Patch_3 data =')
print(euclidean(median_ppc_InsP3, zone3_cases_inflated))
print('Norm Square of Patch_4 data =')
print(euclidean(median_ppc_InsP4, zone4_cases_inflated))


#varnames=[r"$\beta_{1}$", r"$\beta_{2}$" , r"$\beta_{3}$", r"$\beta_{4}$", r"$\kappa_{1}$", r"$\kappa_{2}$" , r"$\kappa_{3}$", r"$\kappa_{4}$",
         # r"$\gamma_{1}$", r"$\gamma_{2}$" , r"$\gamma_{3}$", r"$\gamma_{4}$", r"$E01$",r"$I01$",r"$Y01$", r"$E02$",r"$I02$",r"$Y02$", 
          #r"$E03$",r"$I03$",r"$Y03$", r"$E04$",r"$I04$",r"$Y04$",r"$\phi_{1}$",r"$\phi_{2}$", r"$\phi_{3}$",r"$\phi_{4}$"]

varnames1=[r"$\beta_{1}$", r"$\beta_{2}$" , r"$\beta_{3}$", r"$\beta_{4}$", r"$\kappa_{1}$", r"$\kappa_{2}$" , r"$\kappa_{3}$", r"$\kappa_{4}$",
          r"$\gamma_{1}$", r"$\gamma_{2}$" , r"$\gamma_{3}$", r"$\gamma_{4}$"]

varnames2=[r"E01", r"I01", r"E02",r"I02", r"E03", r"I03", r"E04",r"I04", r"$\nu_{1}$", r"$\nu_{2}$", r"$\nu_{3}$", r"$\nu_{4}$"]



samples1 = SEIRS.Output[burnin:,0:12]

samples2 = SEIRS.Output[burnin:,12:24]

sampleT1 = pd.DataFrame(samples1, columns=["beta_1", "beta_2", "beta_3", "beta_4", "kappa_1", "kappa_2", "kappa_3", "kappa_4",
                                         "gamma_1", "gamma_2", "gamma_3", "gamma_4"])

sampleT2 = pd.DataFrame(samples2, columns=["E01", "I01", "E02", "I02", "E03", "I03","E04", "I04", "nu_1", "nu_2", "nu_3", "nu_4"])

map1 = qq[0,0:12]

map2 = qq[0,12:24]

range1 = np.array([(0.95*x,1.05*x) for x in map1])

range2 = np.array([(0.95*x,1.05*x) for x in map2])

#corner.corner(samples,show_titles=True,labels=varnames,
#                  quantiles=[0.025, 0.5, 0.975],
#                  truths=map,range=range)
corner.corner(sampleT1,show_titles=True,labels=varnames1,truths=map1,range=range1,
                plot_datapoints=False,quantiles=[0.025, 0.5, 0.975],
                title_fmt='.4f')
#plt.savefig(save_results_to + 'corner.pdf')
plt.savefig('/Volumes/F/Hermosillo_four_regions data/figures_twalk/corner_1_real_data.png', dpi = 300 )


corner.corner(sampleT2,show_titles=True,labels=varnames2,truths=map2,range=range2,
                plot_datapoints=False,quantiles=[0.025, 0.5, 0.975],
                title_fmt='.4f')
#plt.savefig(save_results_to + 'corner.pdf')
plt.savefig('/Volumes/F/Hermosillo_four_regions data/figures_twalk/corner_2_real_data.png', dpi = 300 )






"""Show the importance of thinning by using large thinning value so as to have more valid thinning value"""

tau=1500
indexes=np.arange(start=0,stop=TotalNumIter-burnin, step=tau)

beta1 = toutput2[:,0]
beta2 = toutput2[:,1]
beta3 = toutput2[:,2]
beta4 = toutput2[:,3]

kappa1 = toutput2[:,4]
kappa2 = toutput2[:,5]
kappa3 = toutput2[:,6]
kappa4 = toutput2[:,7]

gamma1 = toutput2[:,8]
gamma2 = toutput2[:,9]
gamma3 = toutput2[:,10]
gamma4 = toutput2[:,11]

E01 = toutput2[:,12]
I01 = toutput2[:,13]


E02 = toutput2[:,14]
I02 = toutput2[:,15]

E03 = toutput2[:,16]
I03 = toutput2[:,17]


E04 = toutput2[:,18]
I04 = toutput2[:,19]


nu1 = toutput2[:,20]
nu2 = toutput2[:,21]
nu3 = toutput2[:,22]
nu4 = toutput2[:,23]



tbeta1=beta1[indexes]
tbeta2=beta2[indexes]
tbeta3=beta3[indexes]
tbeta4=beta4[indexes]
tkappa1=kappa1[indexes]
tkappa2=kappa2[indexes]
tkappa3=kappa3[indexes]
tkappa4=kappa4[indexes]
tgamma1=gamma1[indexes]
tgamma2=gamma2[indexes]
tgamma3=gamma3[indexes]
tgamma4=gamma4[indexes]
tE01=E01[indexes]
tI01=I01[indexes]
tE02=E02[indexes]
tI02=I02[indexes]
tE03=E03[indexes]
tI03=I03[indexes]
tE04=E04[indexes]
tI04=I04[indexes]
tnu1=nu1[indexes]
tnu2=nu2[indexes]
tnu3=nu3[indexes]
tnu4=nu4[indexes]

#summary of the thinned chains 
stbeta1 = Analysis(tbeta1)
stbeta2 = Analysis(tbeta2) 
stbeta3 = Analysis(tbeta3)   
stbeta4 = Analysis(tbeta4) 

stkappa1 = Analysis(tkappa1)
stkappa2 = Analysis(tkappa2) 
stkappa3 = Analysis(tkappa3)   
stkappa4 = Analysis(tkappa4)        
      
stgamma1 = Analysis(tgamma1)
stgamma2 = Analysis(tgamma2) 
stgamma3 = Analysis(tgamma3)   
stgamma4 = Analysis(tgamma4)  

stE01 = Analysis(tE01) 
stI01 = Analysis(tI01)   

stE02 = Analysis(tE02) 
stI02 = Analysis(tI02)   

stE03 = Analysis(tE03) 
stI03 = Analysis(tI03)   

stE04 = Analysis(tE04) 
stI04 = Analysis(tI04)   

stnu1 = Analysis(tnu1)
stnu2 = Analysis(tnu2) 
stnu3 = Analysis(tnu3)   
stnu4 = Analysis(tnu4)   




###########plotting the ACF for thinned chains ################
fig, ax = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)

sm.graphics.tsa.plot_acf(tbeta1, ax=ax[0][0])
#ax[0][0].set_xlabel('Lags') 
ax[0][0].set_ylabel('ACF') 
ax[0][0].set_title(r"$\beta_{1}$") 

sm.graphics.tsa.plot_acf(tbeta2, ax=ax[0][1])
#ax[0][1].set_xlabel('Lags') 
#ax[0][1].set_ylabel('ACF') 
ax[0][1].set_title(r"$\beta_{2}$") 

sm.graphics.tsa.plot_acf(tbeta3, ax=ax[0][2])
#ax[0][2].set_xlabel('Lags') 
#ax[0][2].set_ylabel('ACF') 
ax[0][2].set_title(r"$\beta_{3}$") 

sm.graphics.tsa.plot_acf(tbeta4, ax=ax[0][3])
#ax[0][3].set_xlabel('Lags') 
#ax[0][3].set_ylabel('ACF') 
ax[0][3].set_title(r"$\beta_{4}$") 

sm.graphics.tsa.plot_acf(tkappa1, ax=ax[1][0])
#ax[0][5].set_xlabel('Lags') 
ax[1][0].set_ylabel('ACF') 
ax[1][0].set_title(r"$\kappa_{1}$") 

sm.graphics.tsa.plot_acf(tkappa2, ax=ax[1][1])
#ax[0][5].set_xlabel('Lags') 
#ax[0][5].set_ylabel('ACF') 
ax[1][1].set_title(r"$\kappa_{2}$") 

sm.graphics.tsa.plot_acf(tkappa3, ax=ax[1][2])
#ax[0][6].set_xlabel('Lags') 
#ax[0][6].set_ylabel('ACF') 
ax[1][2].set_title(r"$\kappa_{3}$") 

sm.graphics.tsa.plot_acf(tkappa4, ax=ax[1][3])
#ax[1][0].set_xlabel('Lags') 
ax[1][3].set_ylabel('ACF') 
ax[1][3].set_title(r"$\kappa_{4}$") 

sm.graphics.tsa.plot_acf(tgamma1, ax=ax[2][0])
ax[2][0].set_xlabel('Lags') 
ax[2][0].set_ylabel('ACF') 
ax[2][0].set_title(r"$\gamma_{1}$") 

sm.graphics.tsa.plot_acf(tgamma2, ax=ax[2][1])
ax[2][1].set_xlabel('Lags') 
#ax[1][2].set_ylabel('ACF') 
ax[2][1].set_title(r"$\gamma_{2}$") 

sm.graphics.tsa.plot_acf(tgamma3, ax=ax[2][2])
ax[2][2].set_xlabel('Lags') 
#ax[1][3].set_ylabel('ACF') 
ax[2][2].set_title(r"$\gamma_{3}$") 

sm.graphics.tsa.plot_acf(tgamma4, ax=ax[2][3])
ax[2][3].set_xlabel('Lags') 
#ax[1][4].set_ylabel('ACF') 
ax[2][3].set_title(r"$\gamma_{4}$") 
plt.savefig('/Volumes/F/Hermosillo_four_regions data/figures_twalk/ACF_thinned_beta_kappa_gamma_real_data.png', dpi = 300 )



fig, ax = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)

sm.graphics.tsa.plot_acf(tE01, ax=ax[0][0])
#ax[0][0].set_ylabel('Lags') 
#ax[1][5].set_xlabel('Lags') 
ax[0][0].set_ylabel('ACF') 
ax[0][0].set_title(r"E01") 

sm.graphics.tsa.plot_acf(tE02, ax=ax[0][1])
#ax[2][1].set_xlabel('Lags') 
#ax[2][1].set_ylabel('ACF') 
ax[0][1].set_title(r"E02") 

sm.graphics.tsa.plot_acf(tE03, ax=ax[0][2])
#ax[2][4].set_xlabel('Lags') 
#ax[0][2].set_ylabel('ACF') 
ax[0][2].set_title(r"E03") 

sm.graphics.tsa.plot_acf(tE04, ax=ax[0][3])
#ax[1][2].set_xlabel('Lags') 
#ax[1][2].set_ylabel('ACF') 
ax[0][3].set_title(r"E04") 

sm.graphics.tsa.plot_acf(tI01, ax=ax[1][0])
#ax[1][6].set_xlabel('Lags') 
ax[1][0].set_ylabel('ACF') 
ax[1][0].set_title(r"I01") 

sm.graphics.tsa.plot_acf(tI02, ax=ax[1][1])
#ax[2][2].set_xlabel('Lags') 
#ax[2][2].set_ylabel('ACF') 
ax[1][1].set_title(r"I02") 


sm.graphics.tsa.plot_acf(tI03, ax=ax[1][2])
#ax[1][1].set_xlabel('Lags') 
#ax[2][5].set_ylabel('ACF') 
ax[1][2].set_title(r"I03") 

sm.graphics.tsa.plot_acf(tI04, ax=ax[1][3])
#ax[1][3].set_xlabel('Lags') 
#ax[3][1].set_ylabel('ACF') 
ax[1][3].set_title(r"I04") 


sm.graphics.tsa.plot_acf(tnu1, ax=ax[2][0])
ax[2][0].set_xlabel('Lags') 
ax[2][0].set_ylabel('ACF') 
ax[2][0].set_title(r"$\nu_{1}$") 

sm.graphics.tsa.plot_acf(tnu2, ax=ax[2][1])
ax[2][1].set_xlabel('Lags') 
#ax[3][4].set_ylabel('ACF') 
ax[2][1].set_title(r"$\nu_{2}$") 

sm.graphics.tsa.plot_acf(tnu3, ax=ax[2][2])
ax[2][2].set_xlabel('Lags') 
#ax[3][5].set_ylabel('ACF') 
ax[2][2].set_title(r"$\nu_{3}$") 

sm.graphics.tsa.plot_acf(tnu4, ax=ax[2][3])
ax[2][3].set_xlabel('Lags') 
#ax[3][6].set_ylabel('ACF') 
ax[2][3].set_title(r"$\nu_{4}$") 
plt.savefig('/Volumes/F/Hermosillo_four_regions data/figures_twalk/ACF_thinned_E0_I0_nu_real_data.png', dpi = 300 )




#Trace plots after thinning
fig, ax = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
#fig.tight_layout()
ax[0][0].plot(tbeta1)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[0][0].set_ylabel( r'$\beta_{1}$' )
ax[0][0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[0][0].axhline(beta1.mean(), color='k', linestyle='dashed', linewidth=2)
#plt.title( r'$\beta$'+' Traceplot' )
#plt.savefig('SEIRbeta-tp.png', dpi=450)
#plt.show()

ax[0][1].plot(tbeta2)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[0][1].set_ylabel( r'$\beta_{2}$' )
ax[0][1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[0][1].axhline(beta2.mean(), color='k', linestyle='dashed', linewidth=2)

ax[0][2].plot(tbeta3)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[0][2].set_ylabel( r'$\beta_{3}$' )
ax[0][2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[0][2].axhline(beta3.mean(), color='k', linestyle='dashed', linewidth=2)

ax[0][3].plot(tbeta4)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[0][3].set_ylabel( r'$\beta_{4}$' )
ax[0][3].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[0][3].axhline(beta4.mean(), color='k', linestyle='dashed', linewidth=2)

ax[1][0].plot(tkappa1)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[1][0].set_ylabel( r'$\kappa_{1}$' )
ax[1][0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[1][0].axhline(kappa1.mean(), color='k', linestyle='dashed', linewidth=2)

ax[1][1].plot(tkappa2)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[1][1].set_ylabel( r'$\kappa_{2}$' )
ax[1][1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[1][1].axhline(kappa2.mean(), color='k', linestyle='dashed', linewidth=2)

ax[1][2].plot(tkappa3)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[1][2].set_ylabel( r'$\kappa_{3}$' )
ax[1][2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[1][2].axhline(kappa3.mean(), color='k', linestyle='dashed', linewidth=2)

ax[1][3].plot(tkappa4)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[1][3].set_ylabel( r'$\kappa_{4}$' )
ax[1][3].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[1][3].axhline(kappa4.mean(), color='k', linestyle='dashed', linewidth=2)

ax[2][0].plot(tgamma1)
ax[2][0].set_xlabel( r'$Iteration$' )
ax[2][0].set_ylabel( r'$\gamma_{1}$' )
ax[2][0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[2][0].axhline(gamma1.mean(), color='k', linestyle='dashed', linewidth=2)

ax[2][1].plot(tgamma2)
ax[2][1].set_xlabel( r'$Iteration$' )
ax[2][1].set_ylabel( r'$\gamma_{2}$' )
ax[2][1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[2][1].axhline(gamma2.mean(), color='k', linestyle='dashed', linewidth=2)

ax[2][2].plot(tgamma3)
ax[2][2].set_xlabel( r'$Iteration$' )
ax[2][2].set_ylabel( r'$\gamma_{3}$' )
ax[2][2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[2][2].axhline(gamma3.mean(), color='k', linestyle='dashed', linewidth=2)

ax[2][3].plot(tgamma4)
ax[2][3].set_xlabel( r'$Iteration$' )
ax[2][3].set_ylabel( r'$\gamma_{4}$' )
ax[2][3].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[2][3].axhline(gamma4.mean(), color='k', linestyle='dashed', linewidth=2)
plt.savefig('/Volumes/F/Hermosillo_four_regions data/figures_twalk/traceplots_thinned_beta_kappa_gamma_real_data.png', dpi = 300 )


fig, ax = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)

ax[0][0].plot(tE01)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[0][0].set_ylabel( r'E01' )
ax[0][0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[0][0].axhline(E01.mean(), color='k', linestyle='dashed', linewidth=2)

ax[0][1].plot(tE02)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[0][1].set_ylabel( r'E02' )
ax[0][1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[0][1].axhline(E02.mean(), color='k', linestyle='dashed', linewidth=2)

ax[0][2].plot(tE03)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[0][2].set_ylabel( r'E03' )
ax[0][2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[0][2].axhline(E03.mean(), color='k', linestyle='dashed', linewidth=2)

ax[0][3].plot(tE04)
#ax[1][2].set_xlabel( r'$Iteration$' )
ax[0][3].set_ylabel( r'E04' )
ax[0][3].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[0][3].axhline(E04.mean(), color='k', linestyle='dashed', linewidth=2)


ax[1][0].plot(tI01)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[1][0].set_ylabel( r'I01' )
ax[1][0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[1][0].axhline(I01.mean(), color='k', linestyle='dashed', linewidth=2)


ax[1][1].plot(tI02)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[1][1].set_ylabel( r'I02' )
ax[1][1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[1][1].axhline(I02.mean(), color='k', linestyle='dashed', linewidth=2)


ax[1][2].plot(tI03)
#ax[0][0].set_xlabel( r'$Iteration$' )
ax[1][2].set_ylabel( r'I03' )
ax[1][2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[1][2].axhline(I03.mean(), color='k', linestyle='dashed', linewidth=2)


ax[1][3].plot(tI04)
#ax[1][3].set_xlabel( r'$Iteration$' )
ax[1][3].set_ylabel( r'I04' )
ax[1][3].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[1][3].axhline(I04.mean(), color='k', linestyle='dashed', linewidth=2)

ax[2][0].plot(tnu1)
ax[2][0].set_xlabel( r'$Iteration$' )
ax[2][0].set_ylabel( r'$\nu_{1}$' )
ax[2][0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[2][0].axhline(nu1.mean(), color='k', linestyle='dashed', linewidth=2)

ax[2][1].plot(tnu2)
ax[2][1].set_xlabel( r'$Iteration$' )
ax[2][1].set_ylabel( r'$\nu_{2}$' )
ax[2][1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[2][1].axhline(nu2.mean(), color='k', linestyle='dashed', linewidth=2)

ax[2][2].plot(tnu3)
ax[2][2].set_xlabel( r'$Iteration$' )
ax[2][2].set_ylabel( r'$\nu_{3}$' )
ax[2][2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[2][2].axhline(nu3.mean(), color='k', linestyle='dashed', linewidth=2)

ax[2][3].plot(tnu4)
ax[2][3].set_xlabel( r'$Iteration$' )
ax[2][3].set_ylabel( r'$\nu_{4}$' )
ax[2][3].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[2][3].axhline(nu4.mean(), color='k', linestyle='dashed', linewidth=2)
plt.savefig('/Volumes/F/Hermosillo_four_regions data/figures_twalk/traceplots_thinned_E0_I0_nu_real_data.png', dpi = 300 )
