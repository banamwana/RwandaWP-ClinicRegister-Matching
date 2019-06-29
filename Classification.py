#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:54:26 2019

@author: graemepm
"""
import recordlinkage
import pandas as pd
import os
import numpy as np
import re
from recordlinkage.base import BaseCompareFeature
from math import sqrt, exp
from jellyfish import damerau_levenshtein_distance as damLev
import inspect
import json
import time
import xlsxwriter

# =============================================================================
# Import DATA
# =============================================================================

if os.name == 'posix':
    path = '/Users/graemepm/Box Sync/EGHI Record Linkage/FRIL Matching Iterations/'
    desk = '/Users/graemepm/Desktop/'
    
else:
    path = 'C:/Users/gprenti/Box Sync/EGHI Record Linkage/FRIL Matching Iterations/'
    desk = 'C:/Users/gprenti/Desktop/'

readChildDict = {'childid': 'O', 'pkinyaname1': 'O', 'pkinyaname2': 'O',
                 'pkinyaname3': 'O', 'pothername1': 'O', 'pothername2': 'O',
                 'pothername3': 'O', 'pothername4': 'O', 'pnickname': 'O',
                 'ckinyaname1': 'O', 'ckinyaname2': 'O', 'ckinyaname3': 'O',
                 'cothername1': 'O', 'cothername2': 'O', 'cothername3': 'O',
                 'cnickname1': 'O', 'cnickname2': 'O', 'cnickname3': 'O',
                 'mutnum1': 'O', 'mutnum2': 'O', 'mutnum3': 'O', 'mutnum4': 'O',
                 'm_dist': 'O', 'm_hc': 'O', 'm_hh': 'O', 'm_indv': 'O',
                 'min_dob': 'O', 'max_dob': 'O', 'cgender': 'int64', 'vill ID': 'int64',
                 'district': 'O', 'sector': 'O', 'cell': 'O', 'village': 'O',
                 'pkinyaname1_sx': 'O', 'ckinyaname1_sx': 'O', 'cgender_chr': 'O', 'sectLat': 'float64',
                 'sectLong': 'float64'}

childFile = path + '0625/child_key0626.txt'
clinicFile = path + '0625/clinic0626.txt'

child = pd.read_csv(childFile, sep = '\t', parse_dates=['min_dob', 'max_dob'], dtype = readChildDict)
clinic = pd.read_csv(clinicFile, sep = '\t', parse_dates=['patientvisitdate', 'dob_est'])
clinic.drop('index', axis=1, inplace=True)

## Subset datasets to run test quickly
#child = child.sample(50)
#clinic = clinic.sample(50)

## Read in file
featCompFile = path + '0625/featComp0626.txt'
featComp = pd.read_csv(featCompFile, sep = '\t')
featComp.fillna(float(0), inplace=True)

# =============================================================================
# Classification Functions
# =============================================================================

renameClinicDict = {'level_0': 'index', 'recordid': 'RECORDID_', 'patientname': 'PATIENTNAME_',
                    'parentname': 'PARENTNAME_', 'district_clean': 'DISTRICT_',
                    'sector_clean': 'SECTOR_', 'cell_clean': 'CELL_',
                    'village_clean': 'VILLAGE_', 'mdist': 'MDIST_', 
                    'mhf': 'MHF_', 'mhh': 'MHH_', 'mindv': 'MINDV_',
                    'mutuelle_number': 'MUTUELLE_', 'sexe': 'GENDER_', 'dob_est': 'DOB_'}

renameChildDict = {'index': 'level_1', 'childid': '_childid', 'pkinyaname1': '_parentKinyaname',
                   'pothername1': '_parentOthername', 'ckinyaname1': '_childKinyaname',
                   'cothername1': '_childOthername', 'mutnum1': '_mutuelle',
                   'm_dist': '_mdist', 'm_hc': '_mhf', 'm_hh': '_mhh', 'm_indv' : '_mindv',
                   'min_dob': '_dob', 'cgender_chr': '_gender',  'district': '_district',
                   'sector': '_sector', 'cell': '_cell', 'village': '_village'}

tokeep =    ['level_0', 'level_1', 'RECORDID_', '_childid', 
             'PATIENTNAME_', '_childKinyaname', '_childOthername',
             'PARENTNAME_', '_parentKinyaname', '_parentOthername',
             'DISTRICT_', '_district', 'SECTOR_', '_sector',
             'CELL_', '_cell', 'VILLAGE_', '_village',
             'MUTUELLE_', '_mutuelle',
             'MDIST_', '_mdist', 'MHF_', '_mhf',
             'MHH_', '_mhh', 'MINDV_', '_mindv',
             'GENDER_', '_gender',  'DOB_', '_dob']

renameBottomDict = {'level_1': 'index', 'childid': 'RECORDID_', 'pkinyaname1': '_parentKinyaname',
                    'pothername1': '_parentOthername', 'ckinyaname1': '_childKinyaname',
                    'cothername1': '_childOthername', 'mutnum1': 'MUTUELLE_', 'm_dist': 'MDIST_',
                    'm_hc': 'MHF_', 'm_hh': 'MHH_', 'm_indv': 'MINDV_', 'min_dob': 'DOB_',
                    'cgender_chr': 'GENDER_', 'district': 'DISTRICT_', 'sector': 'SECTOR_',
                    'cell': 'CELL_', 'village': 'VILLAGE_'}


toKeepTop = ['index', 'RECORDID_', 'PATIENTNAME_', 'PARENTNAME_', 
             'DISTRICT_', 'SECTOR_', 'CELL_', 'VILLAGE_', 'MUTUELLE_',
             'MDIST_', 'MHF_', 'MHH_', 'MINDV_', 'GENDER_', 'DOB_']

def confSum(series, wDict):
    
    return sum([wDict[col] * value for col, value in series.items()])


def Fuze(df1, df2, matchKey, fuzePath):
    
    ## Match key
    index1 = matchKey.level_0
    index2 = matchKey.level_1
    confScore = matchKey.confScore
         
    ## Subset data using index based on match keys, then reset index for merge
    sub1 = df1.reindex(index = index1)\
        .reset_index().rename(columns=renameClinicDict)
    sub2 = df2.reindex(index = index2)\
        .reset_index().rename(columns=renameBottomDict)
        
    childname = sub2['_childKinyaname'].astype(str) + ' ' + sub2['_childOthername']
    parentname = sub2['_parentKinyaname'].astype(str) + ' ' + sub2['_parentOthername']
    sub2 = sub2.assign(PATIENTNAME_ = childname)
    sub2 = sub2.assign(PARENTNAME_ = parentname)
    sub2 = sub2.rename(columns=renameBottomDict)[toKeepTop]
    sub1 = sub1[toKeepTop]
    
    merged = pd.DataFrame()
    for (i1, row1), (i2, row2) in zip(sub1.iterrows(), sub2.iterrows()):
    
        t1 = row1.to_frame().transpose()
        t2 = row2.to_frame().transpose()
        
        merged = merged.append(t1, ignore_index=True)
        merged = merged.append(t2, ignore_index=True)
        
    
    confScoreDouble = []
    for score in confScore:
        confScoreDouble.append(int(score))
        confScoreDouble.append(int(score))
    
    merged = merged.assign(confScore = confScoreDouble)    
    merged = merged.fillna('')
    merged = merged.applymap(lambda s:s.upper() if type(s) == str else s)
    merged = merged.replace('NA','')
    merged = merged.assign(DOB_ = merged.DOB_.\
            apply(lambda x: x.strftime('%B, %Y') if x is not pd.NaT else x))
    
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(fuzePath, engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    merged.to_excel(writer, sheet_name='pairs')    
    
    # Get the xlsxwriter objects from the dataframe writer object.
    workbook  = writer.book
    worksheet = writer.sheets['pairs']
    
    worksheet.set_column('D:E', 30)
    worksheet.set_column('F:G', 13)
    worksheet.set_column('H:I', 16)
    worksheet.set_column('J:J', 17)
    worksheet.set_column('P:P', 18)
    
    greyTop = workbook.add_format({'fg_color': '#C0C0C0'}) #background color
    greyTop.set_top()
    greyBottom = workbook.add_format({'fg_color': '#C0C0C0'})
    greyBottom.set_bottom()
    dateFormat = workbook.add_format({'num_format': 'mmm yyyy'})
    
    for row in range(1, len(merged), 4):
        worksheet.set_row(row, cell_format=greyTop)
        worksheet.set_row(row + 1, cell_format=greyBottom)
    
    workbook.close()

# =============================================================================
# Confidence Score
# =============================================================================

## Assign weights
weights = {

## Parent and Child Names, built-in Python string comparison
'ckinyaname': 20,   'ckinyaname_cor': 0,    'ckinyaname_sx': 0,     'cothername': 15,
'pkinyaname': 10,   'pkinyaname_cor': 0,    'pkinyaname_sx': 0,     'pothername': 4,

## Parent and Child Names, FRIL style edit distance comparison
'ckinyaname_FR': 0,     'ckinyaname_corFR': 0,  'ckinyaname_sxFR': 0,   'cothername_FR': 0,
'pkinyaname_FR': 0,     'pkinyaname_corFR': 0,  'pkinyaname_sxFR': 0,   'pothername_FR': 0, 

## Mutuelle Number
'mdist': 2, 'mhf': 6, 'mhh': 13, 'mindv': 5,

## Location
'sector': 4, 'cell': 5, 'village': 8, 'geo': 2,

## Other
'gender': 2, 'dob': 4,

## Index
'level_0': 0, 'level_1': 0, 'index': 0}

### Calculate confidence scores
## 22 min (less than half old method)
featComp = featComp.assign(confScore = featComp.apply(lambda x: confSum(x, weights), axis=1))

# =============================================================================
# Acceptance level and Fusion
# =============================================================================

## Set the acceptance levels
accept = 'confScore >= 70'
manual = '50 <= confScore < 70'

matches = featComp.query(accept)[['level_0','level_1', 'confScore']]
possible_matches = featComp.query(manual)[['level_0','level_1', 'confScore']]

## Put data together
acceptFile = path + '0625/accept0629.xlsx'
manReviewFile = path + '0625/man_review0629.xlsx'
Fuze(clinic, child, matches, acceptFile)
Fuze(clinic, child, possible_matches, manReviewFile)










# =============================================================================
# =============================================================================
# Old Code
# =============================================================================
# =============================================================================

## Takes 50min
#confScores = [sum([weights[colname] * value for colname, value in row.iteritems()]) 
#                    for i, row in featComp.drop(['level_0', 'level_1'], axis = 1).iterrows()]
#
#featComp = featComp.assign(confScore = confScores)



#confScore = {colname: pd.Series([weights[colname] * value for value in values]) for colname, values in featComp.iteritems()}
#confScore = pd.DataFrame(confScore)
#confScoreSum = confScore.sum(axis=1)


# =============================================================================
# Data Fusion - Side by side
# =============================================================================

### Column renaming dictionaries
### CAPITAL_     = left/clinic
### _lowercase   = right/child
#renameClinicDict = {'index': 'level_0', 'recordid': 'RECORDID_', 'patientname': 'PATIENTNAME_',
#                    'parentname': 'PARENTNAME_', 'district_clean': 'DISTRICT_',
#                    'sector_clean': 'SECTOR_', 'cell_clean': 'CELL_',
#                    'village_clean': 'VILLAGE_', 'mdist': 'MDIST_', 
#                    'mhf': 'MHF_', 'mhh': 'MHH_', 'mindv': 'MINDV_',
#                    'mutuelle_number': 'MUTUELLE_', 'sexe': 'GENDER_', 'dob_est': 'DOB_'}
#
#renameChildDict = {'index': 'level_1', 'childid': '_childid', 'pkinyaname1': '_parentKinyaname',
#                   'pothername1': '_parentOthername', 'ckinyaname1': '_childKinyaname',
#                   'cothername1': '_childOthername', 'mutnum1': '_mutuelle',
#                   'm_dist': '_mdist', 'm_hc': '_mhf', 'm_hh': '_mhh', 'm_indv' : '_mindv',
#                   'min_dob': '_dob', 'cgender_chr': '_gender',  'district': '_district',
#                   'sector': '_sector', 'cell': '_cell', 'village': '_village'}
#
#
#tokeep =    ['level_0', 'level_1', 'RECORDID_', '_childid', 
#             'PATIENTNAME_', '_childKinyaname', '_childOthername',
#             'PARENTNAME_', '_parentKinyaname', '_parentOthername',
#             'DISTRICT_', '_district', 'SECTOR_', '_sector',
#             'CELL_', '_cell', 'VILLAGE_', '_village',
#             'MUTUELLE_', '_mutuelle',
#             'MDIST_', '_mdist', 'MHF_', '_mhf',
#             'MHH_', '_mhh', 'MINDV_', '_mindv',
#             'GENDER_', '_gender',  'DOB_', '_dob']
#
### Subset data using index based on match keys, then reset index for merge
#d1 = clinic.reindex(index = matches.level_0).reset_index().rename(columns=renameClinicDict)
#d2 = child.reindex(index = matches.level_1).reset_index().rename(columns=renameChildDict)
#merged = pd.concat([d1,d2], axis=1)[tokeep]
#merged = merged.assign(confScore = matches.confScore)
#
#p1 = clinic.reindex(index = possible_matches.level_0).reset_index().rename(columns=renameClinicDict)
#p2 = child.reindex(index = possible_matches.level_1).reset_index().rename(columns=renameChildDict)
#manReview = pd.concat([p1, p2], axis=1)[tokeep]
#manReview = manReview.assign(confScore = possible_matches.confScore)
#
#
##merged.to_csv(path + '0625/merged.txt', index=False, sep = '\t')
##manReview.to_csv(path + '0625/manReview.txt', index=False, sep = '\t')



