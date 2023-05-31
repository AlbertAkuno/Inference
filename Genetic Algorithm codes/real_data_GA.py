#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 12:43:56 2023

@author: albertakuno
"""

# Librerias necesarias para resolver numéricamente el sistema de ecuaciones

from scipy.integrate import odeint
#from scipy.interpolate import interp1d  # Para suavizar las curvas de las soluciones numericas

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Librerías necesarias para el algoritmo genético
#from scipy.stats import truncnorm

from deap import base
from deap import creator
from deap import tools
import pandas as pd
import matplotlib.dates as mdates

#import openpyxl
#from deap import algorithms

import random

#import pickle

# Módulo para implementar elitismo

import elitism

# problem constants:
DIMENSIONS = 20  # number of dimensions
#BOUND_LOW, BOUND_UP = 0.0, 7.0  # boundaries for all dimensions
BOUND_LOW, BOUND_UP = [0.0]*8 + [0.0]*4 + [0.0]*4 + [0.0]*4, [100.0]*8 + [1.68]*4 + [0.3]*4 + [1.0]*4 # boundaries for all dimensions

# Genetic Algorithm constants:
POPULATION_SIZE = 300
#LAMBDA = 400
P_CROSSOVER = 0.75  # probability for crossover
P_MUTATION = 0.55 # (try also 0.5) probability for mutating an individual
#MU = 60
MAX_GENERATIONS = 7000
HALL_OF_FAME_SIZE = 1
CROWDING_FACTOR = 1.0  # crowding factor for crossover and mutation

# set the random seed:
#RANDOM_SEED = 42
#random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# define a multi-objective, minimizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMin)

# helper function for creating random real numbers uniformly distributed within a given range [low, up]
# it assumes that the range is the same for every dimension

#def randomFloat(low, up):
#    return [random.uniform(l, u) for l, u in zip([low] * DIMENSIONS, [up] * DIMENSIONS)]
def randomFloat(lowv, upv):
    return [random.uniform(l, u) for l, u in zip(lowv, upv)]

# create an operator that randomly returns a float in the desired range and dimension:
toolbox.register("attrFloat", randomFloat, BOUND_LOW, BOUND_UP)


# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.attrFloat)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

dat_one_alpha_FF = pd.read_csv("/Volumes/F/Hermosillo_AGEBs_data/one_alpha_POBTOT/alphas_FF.csv") 
df_4zones2 = pd.read_excel(r"/Volumes/F/Hermosillo_four_regions data/Ageb_zonas_updated.xlsx")

frame = [dat_one_alpha_FF,df_4zones2]

#dat_one_alpha_FF2 = dat_one_alpha_FF
#dat_one_alpha_FF2["Zona"] = df_4zones2["Zona"]

#POBTOT_zone2 = dat_one_alpha_FF2[["POBTOT", "Zona"]]
#Nbar_data2 = POBTOT_zone2.groupby('Zona').sum().reset_index()
#Nbar2 = np.array(Nbar_data2["POBTOT"])
#N1 = Nbar2[0]
#N2 = Nbar2[1]
#N3 = Nbar2[2]
#N4 = Nbar2[3]

new_df = pd.concat(frame, axis=1)

POBTOT_zone = new_df[["POBTOT", "Zona"]]

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
# Sistema de ecuaciones del modelo SIR.


def deriv(Y, t, b1,b2,b3,b4, k1, k2, k3, k4, g1, g2, g3, g4):
    S1, S2, S3, S4, E1, E2, E3, E4, I1, I2, I3, I4, Y1, Y2, Y3, Y4, R1, R2, R3, R4, N1, N2, N3, N4 = Y
    S = np.array([S1, S2, S3, S4])
    E = np.array([E1, E2, E3, E4])
    I = np.array([I1, I2, I3, I4])
    R = np.array([R1, R2, R3, R4])
    N = np.array([N1, N2, N3, N4])
    beta = np.array([b1,b2,b3,b4])
    kappa = np.array([k1,k2,k3,k4])
    gamma = np.array([g1,g2,g3,g4])
    Ntilde=np.transpose(pstar)@ N
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

    return np.array([dS1dt, dS2dt, dS3dt, dS4dt, dE1dt, dE2dt, dE3dt, dE4dt, dI1dt, dI2dt, dI3dt, dI4dt, dY1dt, dY2dt, dY3dt, dY4dt, dR1dt, dR2dt, dR3dt, dR4dt, dN1dt, dN2dt, dN3dt,dN4dt])

N01 = Nbar[0]
N02 = Nbar[1]
N03 = Nbar[2]
N04 = Nbar[3]

R01 = 0; R02 = 0;R03 = 0;R04 = 0
e01 =10; e02 =20; e03 = 5; e04 = 5;
i01 =1; i02 = 2 ; i03 = 0; i04 = 0;
#y01 =1; y02 = 2; y03 = 0; y04 =  0;

S01 = N01 - e01 - i01 - R01
S02 = N02 - e02 - i02 - R02
S03 = N03 - e03 - i03 - R03
S04 = N04 - e04 - i04 - R04

Y01 = e01 + i01 + R01
Y02 = e02 + i02 + R02
Y03 = e03 + i03 + R03
Y04 = e04 + i04 + R04


tt = np.linspace(0, len(zone1_cases)+1, len(zone1_cases)+1)
#tt_sim = np.linspace(0, len(zone1_cases)+1, len(zone1_cases)+1)

#y0 = S01, e01, i01, y01, R01, S02, e02, i02, y02, R02,  S03, e03, i03,  y03, R03, S04, e04, i04, y04, R04
y0 = S01, S02, S03, S04, e01, e02, e03, e04, i01, i02,  i03, i04, Y01,  Y02, Y03, Y04, R01, R02, R03, R04, N01, N02, N03, N04

#b1 = 0.80; b2 = 0.90; b3 = 0.95; b4 = 0.8; k1 = 1/8; k2 = 1/7; k3=1/9; k4=1/15; g1 = 1/10; g2 =  1/14; g3 = 1/11; g4 = 1/20

b1 = 1.4869; b2 = 1.1999; b3 = 0.5999; b4 = 0.8840; 

k1 = 0.0099; k2 = 0.0099; k3=0.0099; k4=0.1999; 

g1 = 0.1989; g2 =  0.5286; g3 = 0.1999; g4 = 0.2995

ret = odeint(deriv, y0, tt, args=(b1,b2,b3,b4,k1,k2,k3,k4,g1,g2,g3,g4))


t =  np.linspace(0, len(zone1_cases), len(zone1_cases))

fig, ax = plt.subplots(2, 2, figsize=(14, 14))


ax[0][0].plot(t,np.diff(ret[:,12]) , 'ro', lw=2, label = "Best fit incidence")  # all susceptible curves
ax[0][0].plot(zone1_cases_inflated,'bo', alpha=0.5, lw=2, label = "Observed data")
ax[0][0].set_title("Fitted zone 1 data", weight = "bold")
ax[0][0].set_ylabel(r"count", fontsize=12)


ax[0][1].plot(t, np.diff(ret[:,13]) , 'ro', lw=2, label = "Best fit incidence")  # all susceptible curves
ax[0][1].plot(zone2_cases_inflated,'bo', alpha=0.5, lw=2, label = "Observed data") 
ax[0][1].set_title("Fitted zone 2 data", weight = "bold")


ax[1][0].plot(t,np.diff(ret[:,14]) , 'ro', lw=2, label = "Best fit incidence")  # all susceptible curves
ax[1][0].plot(zone3_cases_inflated,'bo', alpha=0.5, lw=2, label = "Observed data") 
ax[1][0].set_title("Fitted zone 3 data", weight = "bold")
ax[1][0].set_xlabel(r"time", fontsize=12)
ax[1][0].set_ylabel(r"count", fontsize=12)


ax[1][1].plot(t,np.diff(ret[:,15]) , 'ro', lw=2, label = "Best fit incidence")  # all susceptible curves
ax[1][1].plot(zone4_cases_inflated,'bo', alpha=0.5, lw=2, label = "Observed data") 
ax[1][1].set_title("Fitted zone 4 data", weight = "bold")
ax[1][1].set_xlabel(r"time", fontsize=12)
ax[0][0].grid()
ax[0][0].legend()
ax[0][1].grid()
ax[0][1].legend()
ax[1][0].grid()
ax[1][0].legend()
ax[1][1].grid()
ax[1][1].legend()


# Esta funcion calcula la distancia Manhattan entre los datos numericos y los reales.

def objective(individual):
    e01 = individual[0]
    e02 = individual[1]
    e03 = individual[2]
    e04 = individual[3]
    
    i01 = individual[4]
    i02 = individual[5]
    i03 = individual[6]
    i04 = individual[7]
    w1 = individual[8]
    w2 = individual[9]
    w3 = individual[10]
    w4 = individual[11]
    x1 = individual[12]
    x2 = individual[13]
    x3 = individual[14]
    x4 = individual[15]
    y1 = individual[16]
    y2 = individual[17]
    y3 = individual[18]
    y4 = individual[19]

    N01 = Nbar[0]
    N02 = Nbar[1]
    N03 = Nbar[2]
    N04 = Nbar[3]

    R01 = 0
    R02 = 0
    R03 = 0
    R04 = 0
    
    S01 = N01 - e01 - i01 - R01
    S02 = N02 - e02 - i02 - R02
    S03 = N03 - e03 - i03 - R03
    S04 = N04 - e04 - i04 - R04
    
    Y01 = e01 + i01 + R01
    Y02 = e02 + i02 + R02
    Y03 = e03 + i03 + R03
    Y04 = e04 + i04 + R04
    
    
    
    t = np.linspace(0, len(zone1_cases)+1, len(zone1_cases)+1)
    
    y0 = S01, S02, S03, S04, e01, e02, e03, e04, i01, i02, i03, i04, Y01,  Y02, Y03, Y04, R01, R02, R03, R04, N01, N02, N03, N04

    ret = odeint(deriv, y0, t, args=(w1,w2,w3,w4,x1,x2,x3,x4,y1,y2,y3,y4))
    

    delta1 = (( np.diff(ret[:,12]) - zone1_cases_inflated )**2)/(np.diff(ret[:,12])+1)
    delta2 = (( np.diff(ret[:,13]) - zone2_cases_inflated )**2)/(np.diff(ret[:,13])+1)
    delta3 = (( np.diff(ret[:,14]) - zone3_cases_inflated )**2)/(np.diff(ret[:,14])+1)
    delta4 = (( np.diff(ret[:,15]) - zone4_cases_inflated )**2)/(np.diff(ret[:,15])+1)
    
    f = np.sum(delta1)+ np.sum(delta2) + np.sum(delta3) + np.sum(delta4)
    
   
    return f,



#def fitness(individual):
#    return f1(individual), f2(individual), f3(individual), f4(individual),




toolbox.register("evaluate", objective)

#toolbox.register("evaluate", fitness)

# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=60)
#toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR,
                 indpb=1.0/DIMENSIONS)


# Genetic Algorithm flow:
def main():
    
    random.seed(64)
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    #hof = tools.ParetoFront()
    #hof = tools.ParetoFront(similar=operator.__eq__)   
    #hof.update(population)

    # perform the Genetic Algorithm flow with elitism:
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)
    
    #population, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=POPULATION_SIZE, lambda_=LAMBDA, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              #ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # print info for best solution found:
    best = hof.items[0]
    print("-- Best Individual = ", best)
    print("-- Best Fitness = ", best.fitness.values[0])
    
    # extract statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
#    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min Fitness')
    plt.savefig('/Volumes/F/Hermosillo_four_regions data/figures/GA_Evolution_1.png',format = "png", dpi=300)
    #plt.title('Min and Average fitness over Generations')

    plt.show()


if __name__ == "__main__": 
    main()



def funcion_objetivo(e01, e02, e03, e04, i01, i02, i03, i04, w1, w2, w3, w4, x1, x2, x3, x4, y1, y2, y3, y4):
    
    N01 = Nbar[0]
    N02 = Nbar[1]
    N03 = Nbar[2]
    N04 = Nbar[3]

    R01 = 0
    R02 = 0
    R03 = 0
    R04 = 0
    
    S01 = N01 - e01 - i01 - R01
    S02 = N02 - e02 - i02 - R02
    S03 = N03 - e03 - i03 - R03
    S04 = N04 - e04 - i04 - R04
    
    Y01 = e01 + i01 + R01
    Y02 = e02 + i02 + R02
    Y03 = e03 + i03 + R03
    Y04 = e04 + i04 + R04
    
    #t = np.linspace(0, len(zone1_cases)+1, len(zone1_cases)+1)
    t = np.linspace(0, len(zone1_cases)+1, len(zone1_cases)+1)
    
    #y0 = S01, e01, i01, y01, R01, S02, e02, i02, y02, R02,  S03, e03, i03,  y03, R03, S04, e04, i04, y04, R04
    y0 = S01, S02, S03, S04, e01, e02, e03, e04, i01, i02,  i03, i04, Y01,  Y02, Y03, Y04, R01, R02, R03, R04, N01, N02, N03, N04
    
    def deriv(Y, t, b1, b2, b3, b4, k1, k2, k3, k4, g1, g2, g3, g4):
        S1, S2, S3, S4, E1, E2, E3, E4, I1, I2, I3, I4, Y1, Y2, Y3, Y4, R1, R2, R3, R4, N1, N2, N3, N4 = Y
        S = np.array([S1, S2, S3, S4])
        E = np.array([E1, E2, E3, E4])
        I = np.array([I1, I2, I3, I4])
        R = np.array([R1, R2, R3, R4])
        N = np.array([N1, N2, N3, N4])
        beta = np.array([b1,b2,b3,b4])
        kappa = np.array([k1,k2,k3,k4])
        gamma = np.array([g1,g2,g3,g4])
        Ntilde=np.transpose(pstar)@ Nbar
        M1 = np.diag(Lambda)@N -np.diag(S) @ pstar@ np.diag(beta) @ np.linalg.inv(np.diag(Ntilde)) @ np.transpose(pstar)@I\
         - np.diag(mu)@S + np.diag(tau)@R
        M2 = np.diag(S) @ pstar@ np.diag(beta) @ np.linalg.inv(np.diag(Ntilde)) @ np.transpose(pstar)@I \
        -np.diag(kappa+mu)@E 
        M3 = np.diag(kappa)@E - np.diag(gamma + phi + mu)@I
        M4 = np.diag(kappa)@E
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


    ret = odeint(deriv, y0, t, args=(w1,w2,w3,w4,x1,x2,x3,x4,y1,y2,y3,y4))
    

    start_date = "2020-02-26"
    end_date = "2020-09-06"
    
    working_covidcases_zone1 = pd.DataFrame(pd.date_range(start_date, end_date), columns=['FECINISI'])
    tt = np.array(working_covidcases_zone1['FECINISI'][6:194])
    #t =  np.linspace(0, len(zone1_cases), len(zone1_cases))
    
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    ax[0][0].plot(tt,np.diff(ret[:,12]) , 'ro', lw=2, label = "Fitted model incidence")  # all susceptible curves
    ax[0][0].plot(tt, zone1_cases_inflated,'bo', alpha=0.5, lw=2, label = "Observed incidence")
    ax[0][0].set_title("Zone 1", fontsize=20, weight = "bold")
    ax[0][0].set_ylabel('Incidence', fontsize=15)
    ax[0][0].set_xlabel(r"Time", fontsize=15)  # Add an x-label to the axes.
    
    
    ax[0][1].plot(tt, np.diff(ret[:,13]) , 'ro', lw=2, label = "Fitted model incidence")  # all susceptible curves
    ax[0][1].plot(tt, zone2_cases_inflated,'bo', alpha=0.5, lw=2, label = "Observed incidence") 
    ax[0][1].set_title(" Zone 2", fontsize=20, weight = "bold")
    ax[0][1].set_ylabel('Incidence', fontsize=15)
    ax[0][1].set_xlabel(r"Time", fontsize=15)  # Add an x-label to the axes.
    
    ax[1][0].plot(tt,np.diff(ret[:,14]) , 'ro', lw=2, label = "Fitted model incidence")  # all susceptible curves 
    ax[1][0].plot(tt, zone3_cases_inflated,'bo', alpha=0.5, lw=2, label = "Observed incidence") 
    ax[1][0].set_title("Zone 3", fontsize=20, weight = "bold")
    ax[1][0].set_xlabel(r"Time", fontsize=15)
    ax[1][0].set_ylabel(r"Incidence", fontsize=15)
    
    ax[1][1].plot(tt,np.diff(ret[:,15]) , 'ro', lw=2, label = "Fitted model incidence")  # all susceptible curves
    ax[1][1].plot(tt, zone4_cases_inflated,'bo', alpha=0.5, lw=2, label = "Observed incidence") 
    ax[1][1].set_title(" Zone 4", fontsize=20, weight = "bold")
    ax[1][1].set_xlabel(r"Time", fontsize=15)
    ax[1][1].set_ylabel(r"Incidence", fontsize=15)
    ax[1][1].xaxis.set_major_locator(mdates.MonthLocator())
    ax[0][0].grid(True)
    ax[0][0].legend(loc="best", fontsize=15, edgecolor="k", shadow=True, fancybox=True)
    ax[0][1].grid(True)
    ax[0][1].legend(loc="best", fontsize=15, edgecolor="k", shadow=True, fancybox=True)
    ax[1][0].grid(True)
    ax[1][0].legend(loc="best", fontsize=15, edgecolor="k", shadow=True, fancybox=True)
    ax[1][1].grid(True)
    ax[1][1].legend(loc="best", fontsize=15, edgecolor="k", shadow=True, fancybox=True)
    plt.savefig("/Volumes/F/Hermosillo_four_regions data/figures/zones_fitted_real_data.png", format = "png", dpi=300)
    plt.show()
    
    delta1 = (np.absolute( np.diff(ret[:,12]) - zone1_cases_inflated )**2)/(np.diff(ret[:,12])+1)
    delta2 = (np.absolute( np.diff(ret[:,13]) - zone2_cases_inflated )**2)/(np.diff(ret[:,13])+1)
    delta3 = (np.absolute( np.diff(ret[:,14]) - zone3_cases_inflated )**2)/(np.diff(ret[:,14])+1)
    delta4 = (np.absolute( np.diff(ret[:,15]) - zone4_cases_inflated )**2)/(np.diff(ret[:,15])+1)
    
    f1 = np.sum(delta1)
    f2 = np.sum(delta2) 
    f3 = np.sum(delta3)
    f4 = np.sum(delta4)
    f_sum = f1 + f2 + f3 + f4
    return f1, f2, f3, f4, f_sum

funcion_objetivo(19.99999942369515, 19.999998471849075, 19.99997856949357, 2.6164477744447652e-05, 19.999997736977743, 19.999999320144887, 19.999992508440183, 12.245579451746421, 1.0088267859526594, 0.9833660221694541, 0.3886507991259145, 1.3970717581715928, 0.2999999858118283, 0.29999999965724844, 0.2999999999710161, 0.2999999956316358, 0.8837660125905965, 0.9911434917466939, 0.9999999893020635, 0.7033699469500899)
funcion_objetivo(19.999999999999307, 19.999999999945516, 19.99999999999962, 0.11840007016497191, 19.999999999977522, 19.99999999999637, 19.999999999995627, 14.420859322242805, 1.1053079378317514, 1.0122602546838357, 0.4408223155824959, 1.3428742286523965, 0.29999999999995136, 0.3, 0.29999999999997096, 0.3, 0.9999999999999457, 0.9999999999999627, 0.9999999999999983, 0.6839583504754282)

funcion_objetivo(58.99250025793385, 78.1313399114446, 22.940428635889372, 4.4192287513890935, 7.544404607688672e-10, 2.7515297960448733e-11, 4.671863140428291e-10, 2.6108555899579343e-10, 1.0949862078441988, 1.0473418647878072, 0.49042379724841284, 1.4951421562761436, 0.3, 0.2999999999999995, 0.3, 0.3, 1.0, 1.0, 1.0, 0.7852693136629093)
