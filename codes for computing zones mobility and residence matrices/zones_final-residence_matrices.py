#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:04:56 2023

@author: albertakuno
"""

import json
#import logging
#import argparse
#import numpy as np
import pandas as pd

from tqdm import tqdm

tqdm.pandas()

"""
The final residence matrices used in this file were computed in the paper 

Ramírez-Ramírez, L.L.; Montoya, J.A.; Espinoza, J.F.; Mehta, C.; Akuno, A.O.; Bui-Thanh, T. Use of
mobile phone sensing data to estimate residence and mobility times in urban patches during the COVID-19
epidemic: The case of the 2020 outbreak in Hermosillo, Mexico. Preprint in Research Square 2022. Available at
https://doi.org/10.21203/rs.3.rs-2219155/v1.

"""

df_4zones = pd.read_csv("/Volumes/F/Hermosillo_four_regions data/CVE_AGEB.csv") #load the CSV file containing the zones where each AGEB belong


"""Map, compute and save the zones final residence matrix for First Period First Part"""
final_residence_matrix_FF = json.load(open('/Volumes/F/Hermosillo_AGEBs_data/final_residence_matrices/final_residence_matrices_First_First.json', 'r'))
df_final_residence_matrix_FF = pd.DataFrame(final_residence_matrix_FF)
df_final_residence_matrix_FF.shape
df_final_residence_matrix_FF['Zona'] = df_4zones['ZONE']
df_final_residence_matrix_FF
df_final_residence_matrix_zone_FF = df_final_residence_matrix_FF.groupby('Zona').sum().reset_index()
df_final_residence_matrix_grouped_FF = df_final_residence_matrix_zone_FF.drop('Zona',axis=1)
df_final_residence_matrix_grouped_dict_FF = df_final_residence_matrix_grouped_FF.to_dict('list')
json.dump(
    df_final_residence_matrix_grouped_dict_FF,
    open("/Volumes/F/Hermosillo_four_regions data/zone_final_residence_mat/zones_final_residence_matrices_First_First.json", "w"),
    indent=4,
)


"""Map, compute and save the zones final residence matrix for First Period Second Part"""
final_residence_matrix_FS = json.load(open('/Volumes/F/Hermosillo_AGEBs_data/final_residence_matrices/final_residence_matrices_First_Second.json', 'r'))
df_final_residence_matrix_FS = pd.DataFrame(final_residence_matrix_FS)
df_final_residence_matrix_FS.shape
df_final_residence_matrix_FS['Zona'] = df_4zones['ZONE']
df_final_residence_matrix_FS
df_final_residence_matrix_zone_FS = df_final_residence_matrix_FS.groupby('Zona').sum().reset_index()
df_final_residence_matrix_grouped_FS = df_final_residence_matrix_zone_FS.drop('Zona',axis=1)
df_final_residence_matrix_grouped_dict_FS = df_final_residence_matrix_grouped_FS.to_dict('list')
json.dump(
    df_final_residence_matrix_grouped_dict_FS,
    open("/Volumes/F/Hermosillo_four_regions data/zone_final_residence_mat/zones_final_residence_matrices_First_Second.json", "w"),
    indent=4,
)

"""Map compute and save the zones final residence matrix for Second Period First Part"""
final_residence_matrix_SF = json.load(open('/Volumes/F/Hermosillo_AGEBs_data/final_residence_matrices/final_residence_matrices_Second_First.json', 'r'))
df_final_residence_matrix_SF = pd.DataFrame(final_residence_matrix_SF)
df_final_residence_matrix_SF.shape
df_final_residence_matrix_SF['Zona'] = df_4zones['ZONE']
df_final_residence_matrix_SF
df_final_residence_matrix_zone_SF = df_final_residence_matrix_SF.groupby('Zona').sum().reset_index()
df_final_residence_matrix_grouped_SF = df_final_residence_matrix_zone_SF.drop('Zona',axis=1)
df_final_residence_matrix_grouped_dict_SF = df_final_residence_matrix_grouped_SF.to_dict('list')
json.dump(
    df_final_residence_matrix_grouped_dict_SF,
    open("/Volumes/F/Hermosillo_four_regions data/zone_final_residence_mat/zones_final_residence_matrices_Second_First.json", "w"),
    indent=4,
)

"""Map, compute and save the zones final residence matrix for Second Period Second Part"""
final_residence_matrix_SS = json.load(open('/Volumes/F/Hermosillo_AGEBs_data/final_residence_matrices/final_residence_matrices_Second_Second.json', 'r'))
df_final_residence_matrix_SS = pd.DataFrame(final_residence_matrix_SS)
df_final_residence_matrix_SS.shape
df_final_residence_matrix_SS['Zona'] = df_4zones['ZONE']
df_final_residence_matrix_SS
df_final_residence_matrix_zone_SS = df_final_residence_matrix_SS.groupby('Zona').sum().reset_index()
df_final_residence_matrix_grouped_SS = df_final_residence_matrix_zone_SS.drop('Zona',axis=1)
df_final_residence_matrix_grouped_dict_SS = df_final_residence_matrix_grouped_SS.to_dict('list')
json.dump(
    df_final_residence_matrix_grouped_dict_SS,
    open("/Volumes/F/Hermosillo_four_regions data/zone_final_residence_mat/zones_final_residence_matrices_Second_Second.json", "w"),
    indent=4,
)

"""Map, compute and save the zones final residence matrix for Third Period First Part"""
final_residence_matrix_TF = json.load(open('/Volumes/F/Hermosillo_AGEBs_data/final_residence_matrices/final_residence_matrices_Third_First.json', 'r'))
df_final_residence_matrix_TF = pd.DataFrame(final_residence_matrix_TF)
df_final_residence_matrix_TF.shape
df_final_residence_matrix_TF['Zona'] = df_4zones['ZONE']
df_final_residence_matrix_TF
df_final_residence_matrix_zone_TF = df_final_residence_matrix_TF.groupby('Zona').sum().reset_index()
df_final_residence_matrix_grouped_TF = df_final_residence_matrix_zone_TF.drop('Zona',axis=1)
df_final_residence_matrix_grouped_dict_TF = df_final_residence_matrix_grouped_TF.to_dict('list')
json.dump(
    df_final_residence_matrix_grouped_dict_TF,
    open("/Volumes/F/Hermosillo_four_regions data/zone_final_residence_mat/zones_final_residence_matrices_Third_First.json", "w"),
    indent=4,
)

"""Map, compute and save the zones final residence matrix for Third Period Second Part"""
final_residence_matrix_TS = json.load(open('/Volumes/F/Hermosillo_AGEBs_data/final_residence_matrices/final_residence_matrices_Third_Second.json', 'r'))
df_final_residence_matrix_TS = pd.DataFrame(final_residence_matrix_TS)
df_final_residence_matrix_TS.shape
df_final_residence_matrix_TS['Zona'] = df_4zones['ZONE']
df_final_residence_matrix_TS
df_final_residence_matrix_zone_TS = df_final_residence_matrix_TS.groupby('Zona').sum().reset_index()
df_final_residence_matrix_grouped_TS = df_final_residence_matrix_zone_TS.drop('Zona',axis=1)
df_final_residence_matrix_grouped_dict_TS = df_final_residence_matrix_grouped_TS.to_dict('list')
json.dump(
    df_final_residence_matrix_grouped_dict_TS,
    open("/Volumes/F/Hermosillo_four_regions data/zone_final_residence_mat/zones_final_residence_matrices_Third_Second.json", "w"),
    indent=4,
)