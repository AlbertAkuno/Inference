#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 20:19:01 2023

@author: albertakuno
"""

import json
#import logging
#import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

tqdm.pandas()


def remove_non_travellers(zone_residence_matrix,alpha):
    n = zone_residence_matrix.shape[0]
    zone_avg_residence_matrix_star = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if i==j:
                zone_avg_residence_matrix_star[i,i] = (zone_residence_matrix[i,i] -1 + alpha[i])/(alpha[i])
            else:
                zone_avg_residence_matrix_star[i,j] = zone_residence_matrix[i,j]/alpha[i]
    return zone_avg_residence_matrix_star

##calculate zone residence matrices FF 

df_4zones = pd.read_csv("/Volumes/F/Hermosillo_four_regions data/SISVER_Agebs_Zonas.csv")
df_4zones2 = pd.read_excel(r"/Volumes/F/Hermosillo_four_regions data/Ageb_zonas_updated.xlsx")

df_res_FF = pd.read_csv(
    "/Volumes/F/Hermosillo_AGEBs_data/final_time_periods/combined/FirstPeriod_FirstPart_comb.csv",
    sep=";",
    header=0,
    names=["id", "ageb_crit2", "ageb_crit1", "loose"],
)
df_res_FF["zone"] = df_res_FF["loose"].apply(lambda x: df_4zones2.iloc[x]["Zona"] if x != -1 else -1)

matrices_dict_FF = json.load(
    open(
        "/Volumes/F/Hermosillo_four_regions data/zone_final_residence_mat/zones_final_residence_matrices_First_First.json",
        "r",
    )
)


zone_residence_matrix_FF = np.zeros((4, 4))
for idx, group in tqdm(df_res_FF.groupby("zone")):
    if idx == -1:
        #print(group.shape)
        continue
    for ids in tqdm(group["id"], leave=False):
        #print(idx)
        temp = np.array(matrices_dict_FF[ids])
        temp = temp / temp.sum() if temp.sum() != 0 else temp
        zone_residence_matrix_FF[idx-1, :] += temp
    zone_residence_matrix_FF[idx-1, :] /= group.shape[0]

zone_alpha_data_FF = pd.read_csv("/Volumes/F/Hermosillo_four_regions data/zone_alpha_values_data/alphas_FF.csv")
zone_alpha_FF = zone_alpha_data_FF["proporcion"]
mod_zone_alpha_data_FF = zone_alpha_data_FF.iloc[1: , :]
zone_alpha_FF =np.array([1,1,1,1])-np.array(mod_zone_alpha_data_FF["proporcion"])


zone_avg_residence_matrix_star_FF = remove_non_travellers(zone_residence_matrix_FF,zone_alpha_FF)

zone_avg_residence_matrix_star_FF
zone_avg_residence_matrix_star_FF.sum(axis=1)

np.save("/Volumes/F/Hermosillo_four_regions data/zone_avg_residence_matrices/zone_avg_res_mat_First_First",zone_avg_residence_matrix_star_FF)

#zone_avg_res_mat_First_First = np.load("/Volumes/F/Hermosillo_four_regions data/zone_avg_residence_matrices/zone_avg_res_mat_First_First.npy")


##calculate zone residence matrices FS 

df_res_FS = pd.read_csv(
    "/Volumes/F/Hermosillo_AGEBs_data/final_time_periods/combined/FirstPeriod_SecondPart_comb.csv",
    sep=";",
    header=0,
    names=["id", "ageb_crit2", "ageb_crit1", "loose"],
)
df_res_FS["zone"] = df_res_FS["loose"].apply(lambda x: df_4zones2.iloc[x]["Zona"] if x != -1 else -1)

matrices_dict_FS = json.load(
    open(
        "/Volumes/F/Hermosillo_four_regions data/zone_final_residence_mat/zones_final_residence_matrices_First_Second.json",
        "r",
    )
)


zone_residence_matrix_FS = np.zeros((4, 4))
for idx, group in tqdm(df_res_FS.groupby("zone")):
    if idx == -1:
        #print(group.shape)
        continue
    for ids in tqdm(group["id"], leave=False):
        #print(idx)
        temp = np.array(matrices_dict_FS[ids])
        temp = temp / temp.sum() if temp.sum() != 0 else temp
        zone_residence_matrix_FS[idx-1, :] += temp
    zone_residence_matrix_FS[idx-1, :] /= group.shape[0]

zone_alpha_data_FS = pd.read_csv("/Volumes/F/Hermosillo_four_regions data/zone_alpha_values_data/alphas_FS.csv")
zone_alpha_FS = zone_alpha_data_FS["proporcion"]
mod_zone_alpha_data_FS = zone_alpha_data_FS.iloc[1: , :]
zone_alpha_FS =np.array([1,1,1,1])-np.array(mod_zone_alpha_data_FS["proporcion"])


zone_avg_residence_matrix_star_FS = remove_non_travellers(zone_residence_matrix_FS,zone_alpha_FS)

zone_avg_residence_matrix_star_FS
zone_avg_residence_matrix_star_FS.sum(axis=1)

np.save("/Volumes/F/Hermosillo_four_regions data/zone_avg_residence_matrices/zone_avg_res_mat_First_Second",zone_avg_residence_matrix_star_FS)

#zone_avg_res_mat_First_Second = np.load("/Volumes/F/Hermosillo_four_regions data/zone_avg_residence_matrices/zone_avg_res_mat_First_Second.npy")


##calculate zone residence matrices SF 

df_res_SF = pd.read_csv(
    "/Volumes/F/Hermosillo_AGEBs_data/final_time_periods/combined/SecondPeriod_FirstPart_comb.csv",
    sep=";",
    header=0,
    names=["id", "ageb_crit2", "ageb_crit1", "loose"],
)
df_res_SF["zone"] = df_res_SF["loose"].apply(lambda x: df_4zones2.iloc[x]["Zona"] if x != -1 else -1)

matrices_dict_SF = json.load(
    open(
        "/Volumes/F/Hermosillo_four_regions data/zone_final_residence_mat/zones_final_residence_matrices_Second_First.json",
        "r",
    )
)


zone_residence_matrix_SF = np.zeros((4, 4))
for idx, group in tqdm(df_res_SF.groupby("zone")):
    if idx == -1:
        #print(group.shape)
        continue
    for ids in tqdm(group["id"], leave=False):
        #print(idx)
        temp = np.array(matrices_dict_SF[ids])
        temp = temp / temp.sum() if temp.sum() != 0 else temp
        zone_residence_matrix_SF[idx-1, :] += temp
    zone_residence_matrix_SF[idx-1, :] /= group.shape[0]

zone_alpha_data_SF = pd.read_csv("/Volumes/F/Hermosillo_four_regions data/zone_alpha_values_data/alphas_SF.csv")
zone_alpha_SF = zone_alpha_data_SF["proporcion"]
mod_zone_alpha_data_SF = zone_alpha_data_SF.iloc[1: , :]
zone_alpha_SF =np.array([1,1,1,1])-np.array(mod_zone_alpha_data_SF["proporcion"])


zone_avg_residence_matrix_star_SF = remove_non_travellers(zone_residence_matrix_SF,zone_alpha_SF)

zone_avg_residence_matrix_star_SF
zone_avg_residence_matrix_star_SF.sum(axis=1)

np.save("/Volumes/F/Hermosillo_four_regions data/zone_avg_residence_matrices/zone_avg_res_mat_Second_First",zone_avg_residence_matrix_star_SF)

#zone_avg_res_mat_Second_First = np.load("/Volumes/F/Hermosillo_four_regions data/zone_avg_residence_matrices/zone_avg_res_mat_Second_First.npy")


##calculate zone residence matrices SS 

df_res_SS = pd.read_csv(
    "/Volumes/F/Hermosillo_AGEBs_data/final_time_periods/combined/SecondPeriod_SecondPart_comb.csv",
    sep=";",
    header=0,
    names=["id", "ageb_crit2", "ageb_crit1", "loose"],
)
df_res_SS["zone"] = df_res_SS["loose"].apply(lambda x: df_4zones2.iloc[x]["Zona"] if x != -1 else -1)

matrices_dict_SS = json.load(
    open(
        "/Volumes/F/Hermosillo_four_regions data/zone_final_residence_mat/zones_final_residence_matrices_Second_Second.json",
        "r",
    )
)


zone_residence_matrix_SS = np.zeros((4, 4))
for idx, group in tqdm(df_res_SS.groupby("zone")):
    if idx == -1:
        #print(group.shape)
        continue
    for ids in tqdm(group["id"], leave=False):
        #print(idx)
        temp = np.array(matrices_dict_SS[ids])
        temp = temp / temp.sum() if temp.sum() != 0 else temp
        zone_residence_matrix_SS[idx-1, :] += temp
    zone_residence_matrix_SS[idx-1, :] /= group.shape[0]

zone_alpha_data_SS = pd.read_csv("/Volumes/F/Hermosillo_four_regions data/zone_alpha_values_data/alphas_SS.csv")
zone_alpha_SS = zone_alpha_data_SS["proporcion"]
mod_zone_alpha_data_SS = zone_alpha_data_SS.iloc[1: , :]
zone_alpha_SS =np.array([1,1,1,1])-np.array(mod_zone_alpha_data_SS["proporcion"])


zone_avg_residence_matrix_star_SS = remove_non_travellers(zone_residence_matrix_SS,zone_alpha_SS)

zone_avg_residence_matrix_star_SS
zone_avg_residence_matrix_star_SS.sum(axis=1)

np.save("/Volumes/F/Hermosillo_four_regions data/zone_avg_residence_matrices/zone_avg_res_mat_Second_Second",zone_avg_residence_matrix_star_SS)

#zone_avg_res_mat_Second_Second = np.load("/Volumes/F/Hermosillo_four_regions data/zone_avg_residence_matrices/zone_avg_res_mat_Second_Second.npy")


##calculate zone residence matrices TF 

df_res_TF = pd.read_csv(
    "/Volumes/F/Hermosillo_AGEBs_data/final_time_periods/combined/ThirdPeriod_FirstPart_comb.csv",
    sep=";",
    header=0,
    names=["id", "ageb_crit2", "ageb_crit1", "loose"],
)
df_res_TF["zone"] = df_res_TF["loose"].apply(lambda x: df_4zones2.iloc[x]["Zona"] if x != -1 else -1)

matrices_dict_TF = json.load(
    open(
        "/Volumes/F/Hermosillo_four_regions data/zone_final_residence_mat/zones_final_residence_matrices_Third_First.json",
        "r",
    )
)


zone_residence_matrix_TF = np.zeros((4, 4))
for idx, group in tqdm(df_res_TF.groupby("zone")):
    if idx == -1:
        #print(group.shape)
        continue
    for ids in tqdm(group["id"], leave=False):
        #print(idx)
        temp = np.array(matrices_dict_TF[ids])
        temp = temp / temp.sum() if temp.sum() != 0 else temp
        zone_residence_matrix_TF[idx-1, :] += temp
    zone_residence_matrix_TF[idx-1, :] /= group.shape[0]

zone_alpha_data_TF = pd.read_csv("/Volumes/F/Hermosillo_four_regions data/zone_alpha_values_data/alphas_TF.csv")
zone_alpha_TF = zone_alpha_data_TF["proporcion"]
mod_zone_alpha_data_TF = zone_alpha_data_TF.iloc[1: , :]
zone_alpha_TF =np.array([1,1,1,1])-np.array(mod_zone_alpha_data_TF["proporcion"])


zone_avg_residence_matrix_star_TF = remove_non_travellers(zone_residence_matrix_TF,zone_alpha_TF)

zone_avg_residence_matrix_star_TF
zone_avg_residence_matrix_star_TF.sum(axis=1)

np.save("/Volumes/F/Hermosillo_four_regions data/zone_avg_residence_matrices/zone_avg_res_mat_Third_First",zone_avg_residence_matrix_star_TF)

#zone_avg_res_mat_Third_First = np.load("/Volumes/F/Hermosillo_four_regions data/zone_avg_residence_matrices/zone_avg_res_mat_TF_First.npy")


##calculate zone residence matrices TS 

df_res_TS = pd.read_csv(
    "/Volumes/F/Hermosillo_AGEBs_data/final_time_periods/combined/ThirdPeriod_SecondPart_comb.csv",
    sep=";",
    header=0,
    names=["id", "ageb_crit2", "ageb_crit1", "loose"],
)
df_res_TS["zone"] = df_res_TS["loose"].apply(lambda x: df_4zones2.iloc[x]["Zona"] if x != -1 else -1)

matrices_dict_TS = json.load(
    open(
        "/Volumes/F/Hermosillo_four_regions data/zone_final_residence_mat/zones_final_residence_matrices_Third_Second.json",
        "r",
    )
)


zone_residence_matrix_TS = np.zeros((4, 4))
for idx, group in tqdm(df_res_TS.groupby("zone")):
    if idx == -1:
        #print(group.shape)
        continue
    for ids in tqdm(group["id"], leave=False):
        #print(idx)
        temp = np.array(matrices_dict_TS[ids])
        temp = temp / temp.sum() if temp.sum() != 0 else temp
        zone_residence_matrix_TS[idx-1, :] += temp
    zone_residence_matrix_TS[idx-1, :] /= group.shape[0]

zone_alpha_data_TS = pd.read_csv("/Volumes/F/Hermosillo_four_regions data/zone_alpha_values_data/alphas_TS.csv")
zone_alpha_TS = zone_alpha_data_TS["proporcion"]
mod_zone_alpha_data_TS = zone_alpha_data_TS.iloc[1: , :]
zone_alpha_TS =np.array([1,1,1,1])-np.array(mod_zone_alpha_data_TS["proporcion"])


zone_avg_residence_matrix_star_TS = remove_non_travellers(zone_residence_matrix_TS,zone_alpha_TS)

zone_avg_residence_matrix_star_TS
zone_avg_residence_matrix_star_TS.sum(axis=1)

np.save("/Volumes/F/Hermosillo_four_regions data/zone_avg_residence_matrices/zone_avg_res_mat_Third_Second",zone_avg_residence_matrix_star_TS)

#zone_avg_res_mat_Third_Second = np.load("/Volumes/F/Hermosillo_four_regions data/zone_avg_residence_matrices/zone_avg_res_mat_Third_Second.npy")



