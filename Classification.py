#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for performing classification on dataframe of feature comparisons,
using deterministic threshold with weighted confidence score,
identifying accepted matches and potential matches in need of clerical review,
and through use of a rule-based classifier
"""
# =============================================================================
# Libraries
# =============================================================================
import recordlinkage
import pandas as pd
import os
import numpy as np
import re
import json
import time
import xlsxwriter
import seaborn as sns
from copy import deepcopy
import datetime
# =============================================================================
# Directories
# =============================================================================
if os.name == 'posix':
    home = '/Users/graemepm/'
else:
    home = 'C:/Users/gprenti/'
    
path = home + 'Box Sync/EGHI Record Linkage/FRIL Matching Iterations/'
path_code = home + 'Box Sync/EGHI Record Linkage/Code/'
desk = home + 'Desktop/'
    
childFile = path + 'Manual Cleaning/child_keyMCLEAN.txt'
clinicFile = path + '0820/clinic0806.txt'
featCompFile = path + '0820/featComp0820.txt'

# =============================================================================
# Source functions
# =============================================================================
exec(open(path_code + 'Clerical Review.py').read())
#exec(open(path_code + 'Rule Based Classifier.py').read())
# =============================================================================
# Import DATA
# =============================================================================
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

## Clinic and child data
child = pd.read_csv(childFile, sep = '\t', parse_dates=['dob'], dtype = readChildDict)
clinic = pd.read_csv(clinicFile, sep = '\t', parse_dates=['patientvisitdate', 'dob_est'])
clinic.drop('index', axis=1, inplace=True)

## Feature comparison data
#featComp = pd.read_csv(featCompFile, sep = '\t', nrows=1000)
#featComp = test50
featComp = pd.read_csv(featCompFile, sep = '\t')
featComp.fillna(float(0), inplace=True)

# =============================================================================
# Rename dictionaries
# =============================================================================
renameTopDict = {'level_0': 'index', 'recordid': 'RECORDID_', 'patientname': 'PATIENTNAME_',
                    'parentname': 'PARENTNAME_', 'district_clean': 'DISTRICT_',
                    'sector_clean': 'SECTOR_', 'cell_clean': 'CELL_',
                    'village_clean': 'VILLAGE_', 'mdist': 'MDIST_', 
                    'mhf': 'MHF_', 'mhh': 'MHH_', 'mindv': 'MINDV_',
                    'mutuelle_number': 'MUTUELLE_', 'sexe': 'GENDER_', 'dob_est': 'DOB_'}

renameBottomDict = {'level_1': 'index', 'childid': 'RECORDID_', 'pkinyaname1': '_parentKinyaname',
                    'pothername1': '_parentOthername', 'ckinyaname1': '_childKinyaname',
                    'cothername1': '_childOthername', 'mutnum1': 'MUTUELLE_', 'm_dist': 'MDIST_',
                    'm_hc': 'MHF_', 'm_hh': 'MHH_', 'm_indv': 'MINDV_', 'dob': 'DOB_',
                    'cgender_chr': 'GENDER_', 'district': 'DISTRICT_', 'sector': 'SECTOR_',
                    'cell': 'CELL_', 'village': 'VILLAGE_'}

toKeep = ['index', 'RECORDID_', 'PATIENTNAME_', 'PARENTNAME_', 
          'DISTRICT_', 'SECTOR_', 'CELL_', 'VILLAGE_', 'MUTUELLE_',
          'MDIST_', 'MHF_', 'MHH_', 'MINDV_', 'GENDER_', 'DOB_']

# =============================================================================
# Parent/HOH Name - take Kinyaname, Othername scores from highest Kinyaname score
# =============================================================================

# Sum pairs of Parent/HOH Kinyarwanda and Other name similarity scores
# Parent names are pair-dependent, i.e. kinyaname1 and othername1 considered jointly
featComp = featComp.assign(
        psum1 =     featComp[['pkinyaname1',    'pothername1']].sum(axis=1),
        psum_sx1 =  featComp[['pkinyaname1_sx', 'pothername1_sx']].sum(axis=1),
        psum2 =     featComp[['pkinyaname2',    'pothername2']].sum(axis=1),
        psum_sx2 =  featComp[['pkinyaname2_sx', 'pothername2_sx']].sum(axis=1),
        psum3 =     featComp[['pkinyaname3',    'pothername3']].sum(axis=1),
        psum_sx3 =  featComp[['pkinyaname3_sx', 'pothername3_sx']].sum(axis=1),
        psum4 =     featComp[['pkinyaname4',    'pothername4']].sum(axis=1),
        psum_sx4 =  featComp[['pkinyaname4_sx', 'pothername4_sx']].sum(axis=1),
        hsum1 =     featComp[['hkinyaname1',    'hothername1']].sum(axis=1),
        hsum_sx1 =  featComp[['hkinyaname1_sx', 'hothername1_sx']].sum(axis=1),
        hsum2 =     featComp[['hkinyaname2',    'hothername2']].sum(axis=1),
        hsum_sx2 =  featComp[['hkinyaname2_sx', 'hothername2_sx']].sum(axis=1),
        hsum3 =     featComp[['hkinyaname3',    'hothername3']].sum(axis=1),
        hsum_sx3 =  featComp[['hkinyaname3_sx', 'hothername3_sx']].sum(axis=1),
        hsum4 =     featComp[['hkinyaname4',    'hothername4']].sum(axis=1),
        hsum_sx4 =  featComp[['hkinyaname4_sx', 'hothername4_sx']].sum(axis=1)
        )

# Determine pair with highest sum of similarity scores
featComp = featComp.assign(
        PH_which = featComp.filter(regex='sum[1-4]').idxmax(axis=1),
        PH_sx_which = featComp.filter(regex='sum_sx').idxmax(axis=1)
        )

# Fill the name of the selected columns
featComp = featComp.assign(
        ParentKinyaCol = featComp.PH_which.apply(lambda x: x[:1] + 'kinyaname' + x[-1:]),
        ParentOtherCol = featComp.PH_which.apply(lambda x: x[:1] + 'othername' + x[-1:]),
        ParentKinyaSXCol = featComp.PH_sx_which.apply(lambda x: x[:1] + 'kinyaname' + x[-1:] + '_sx'),
        ParentOtherSXCol = featComp.PH_sx_which.apply(lambda x: x[:1] + 'othername' + x[-1:] + '_sx')
        )

# Make final columns with the scores for each selected comparison with highest score
featComp = featComp.assign(
        ParentKinya = featComp.apply(lambda x: x[x.ParentKinyaCol], axis=1),
        ParentOther = featComp.apply(lambda x: x[x.ParentOtherCol], axis=1),
        ParentKinyaSX = featComp.apply(lambda x: x[x.ParentKinyaSXCol], axis=1),
        ParentOtherSX = featComp.apply(lambda x: x[x.ParentOtherSXCol], axis=1)
        )

# =============================================================================
# Child Kinyaname - take max score
# =============================================================================
# Child is assumed to be same person, i.e kinyaname1 can mix with othername1,2,3
featComp = featComp.assign(
        ChildKinya = featComp.filter(regex='ckinyaname[1-3](?!_)').max(axis=1),
        ChildKinyaSX = featComp.filter(regex='ckinyaname[1-3]_sx').max(axis=1),
        ChildOther = featComp.filter(regex='cothername[1-3](?!_)').max(axis=1),
        ChildOtherSX = featComp.filter(regex='cothername[1-3]_sx').max(axis=1),
        ChildKinyaCol = featComp.filter(regex='ckinyaname[1-3](?!_)').idxmax(axis=1),
        ChildKinyaSXCol = featComp.filter(regex='ckinyaname[1-3]_sx').idxmax(axis=1),
        ChildOtherCol = featComp.filter(regex='cothername[1-3](?!_)').idxmax(axis=1),
        ChildOtherSXCol = featComp.filter(regex='cothername[1-3]_sx').idxmax(axis=1)
        )

# =============================================================================
# Confidence Score
# =============================================================================

## Assign weights
weights = {

## Parent and Child Names
'ChildKinya': 20, 'ChildOther': 15, 
'ParentKinya': 10, 'ParentOther': 4,

## Mutuelle Number
'mdist': 2, 'mhf': 6, 'mhh': 13, 'mindv': 5,

## Location
'district': 0, 'sector': 4, 'cell': 5, 'village': 8, 'geo': 2, 'adjacentVillage': 0,

## Other
'gender': 2, 'dob': 4}

weightsSX = {

## Parent and Child Names
'ChildKinyaSX': 20, 'ChildOtherSX': 15, 
'ParentKinyaSX': 10, 'ParentOtherSX': 4,

## Mutuelle Number
'mdist': 2, 'mhf': 6, 'mhh': 13, 'mindv': 5,

## Location
'district': 0, 'sector': 4, 'cell': 5, 'village': 8, 'geo': 2, 'adjacentVillage': 0,

## Other
'gender': 2, 'dob': 4}

### Calculate confidence scores
featComp = featComp.assign(confScore = featComp.apply(lambda x: confSum(x, weights), axis=1))
featComp = featComp.assign(confScoreSX = featComp.apply(lambda x: confSum(x, weightsSX), axis=1))

# =============================================================================
# Test fusion
# =============================================================================
#test = featComp[['confScore'] + choiceCols]
#Fuze(clinic, child, test, desk + 'test.xlsx')

# =============================================================================
# Clerical Review
# =============================================================================

## Set the acceptance levels
accLevel = 70
manLevel = 55

accept = 'confScore >= ' + str(accLevel)
manual = str(manLevel) + ' <= confScore < ' + str(accLevel)

acceptSX = 'confScoreSX >= ' + str(accLevel)
manualSX = str(manLevel) + ' <= confScoreSX < ' + str(accLevel)

choiceCols = ['level_0','level_1', 
              'ParentKinyaCol', 'ParentOtherCol', 'ParentKinyaSXCol', 'ParentOtherSXCol',
              'ChildKinyaCol', 'ChildKinyaSXCol', 'ChildOtherCol', 'ChildOtherSXCol']

matches = featComp.query(accept)[['confScore'] + choiceCols]
possible_matches = featComp.query(manual)[['confScore'] + choiceCols]

matchesSX = featComp.query(acceptSX)[['confScoreSX'] + choiceCols]
possible_matchesSX = featComp.query(manualSX)[['confScoreSX'] + choiceCols]

## Put data together
acceptFile = path + '0820/accept0820.xlsx'
manReviewFile = path + '0820/man_review0820.xlsx'
Fuze(clinic, child, matches, acceptFile, renameTopDict, renameBottomDict, toKeep)
Fuze(clinic, child, possible_matches, manReviewFile, renameTopDict, renameBottomDict, toKeep)

acceptFileSX = path + '0820/acceptSX0820.xlsx'
manReviewFileSX = path + '0820/man_reviewSX0820.xlsx'
Fuze(clinic, child, matches, acceptFileSX, renameTopDict, renameBottomDict, toKeep, sx=True)
Fuze(clinic, child, possible_matches, manReviewFileSX, renameTopDict, renameBottomDict, toKeep, sx=True)

## Plot confidence score distribution
confScoreHist = sns.distplot(featComp.confScore)
confScoreHist.axvline(accLevel, color="k", linestyle="--")
confScoreHist.axvline(manLevel, color="r", linestyle="--")
confScoreHist.annotate('Accept\nn=' + str(len(matches)), (accLevel + 1, 0.02))
confScoreHist.annotate('Manual\nReview\nn=' + str(len(possible_matches)), (manLevel + 1, 0.02))
confScoreHist.get_figure().savefig(path + '0820/confScoreHist0820.png', dpi=300)
confScoreHist.get_figure().clf()

confScoreHistSX = sns.distplot(featComp.confScoreSX)
confScoreHistSX.axvline(accLevel, color="k", linestyle="--")
confScoreHistSX.axvline(manLevel, color="r", linestyle="--")
confScoreHistSX.annotate('Accept\nn=' + str(len(matchesSX)), (accLevel + 1, 0.02))
confScoreHistSX.annotate('Manual\nReview\nn=' + str(len(possible_matchesSX)), (manLevel + 1, 0.02))
confScoreHistSX.get_figure().savefig(path + '0820/confScoreHistSX0820.png', dpi=300)
confScoreHistSX.get_figure().clf()

## Save weights
weightsDF = pd.concat(
        [pd.DataFrame.from_dict(weights, orient='index').reset_index(),
        pd.DataFrame.from_dict(weightsSX, orient='index').reset_index()],
        axis=1)

weightsDF.columns = ['confScore', 'confScore_Weights',
                     'confScoreSX', 'confScoreSX_Weights']

weightsDF.to_csv(path + '0820/weights0820.csv')

# =============================================================================
# Man Review 54-48
# =============================================================================
#m54 = featComp.query('confScore == 54')[['confScore'] + choiceCols]
#m53 = featComp.query('confScore == 53')[['confScore'] + choiceCols]
#m52 = featComp.query('confScore == 52')[['confScore'] + choiceCols]
#m51 = featComp.query('confScore == 51')[['confScore'] + choiceCols]
#m50 = featComp.query('confScore == 50')[['confScore'] + choiceCols]
#m49 = featComp.query('confScore == 49')[['confScore'] + choiceCols]
#m48 = featComp.query('confScore == 48')[['confScore'] + choiceCols]
#
#Fuze(clinic, child, m54, path + '0820/man_review0820_54.xlsx')
#Fuze(clinic, child, m53, path + '0820/man_review0820_53.xlsx')
#Fuze(clinic, child, m52, path + '0820/man_review0820_52.xlsx')
#Fuze(clinic, child, m51, path + '0820/man_review0820_51.xlsx')
#Fuze(clinic, child, m50, path + '0820/man_review0820_50.xlsx')
#Fuze(clinic, child, m49, path + '0820/man_review0820_49.xlsx')
#Fuze(clinic, child, m48, path + '0820/man_review0820_48.xlsx')

