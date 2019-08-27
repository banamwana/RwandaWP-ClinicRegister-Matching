#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for performing classification on dataframe of feature comparisons,
using deterministic threshold with weighted confidence score,
identifying accepted matches and potential matches in need of clerical review
"""
# =============================================================================
# Libraries
# =============================================================================
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
desk = home + 'Desktop/'
    
childFile = path + '0820/child_key0820.txt'
clinicFile = path + '0820/clinic0806.txt'
featCompFile = path + '0820/featComp0820.txt'
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
# Classification Functions
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

def confSum(series, wDict):
    
    return np.nansum([wDict[col] * value for col, value in series.items() if col in wDict.keys()])

def combineTopBottom(df1, df2):
    
    clinicID = df1.RECORDID_
    redCap = df1.redcap_event_name
    childID = df2.RECORDID_
    hhid = df2.hhid
    
    df1 = df1.drop(['RECORDID_', 'index'], axis=1)
    df2 = df2.drop(['RECORDID_', 'index'], axis=1)
    
    def strf_time(date):
        try:
            datestring = date.strftime('%B, %Y')
        except:
            datestring = date
        return datestring
        
    make_upper = lambda s: str(s).upper()
    
    df1 = df1.assign(DOB_ = df1.DOB_.apply(strf_time))
    df1 = df1.applymap(make_upper).replace('NA',' ').replace('NAN', ' ').replace('NAT', ' ')

    df2 = df2.assign(DOB_ = df2.DOB_.apply(strf_time))
    df2 = df2.applymap(make_upper).replace('NA',' ').replace('NAN', ' ').replace('NAT', ' ')
    
    merged = pd.DataFrame()
    for (i1, row1), (i2, row2) in zip(df1.iterrows(), df2.iterrows()):
        
        mergedRow = {}
        for (col1, value1), (col2, value2) in zip(row1.iteritems(), row2.iteritems()):
            
            value = '\n'.join([str(value1), str(value2)])
            
            mergedRow.update({col1: value})
            
        merged = merged.append(pd.DataFrame(mergedRow, index=[0]), ignore_index=True, sort=False)

    merged = merged.assign(clinicID = clinicID, redCap = redCap, 
                           childID = childID, hhID = hhid)
    merged = merged.sort_values(by='clinicID').reset_index(drop=True)
    
    
    return merged

def Fuze(df1, df2, matchKey, fuzePath, sx=False):
    
    # =============================================================================
    # Match keys and confidence score
    # =============================================================================
    index1 = matchKey.level_0
    index2 = matchKey.level_1
    confScore = [int(score) for score in matchKey.confScore]
    
    # =============================================================================
    # Clinic Data
    # =============================================================================
    
    ## Sync clinic data with matchkey using index based on match keys, reset index for merge
    sub1 = df1.reindex(index = index1).reset_index()
    dob1 = sub1.dob_est
    sub1 = sub1.drop('dob_est', axis=1).fillna(' ')
    
    ## Use either Parent or HOH name (based off the max sum of similarity scores)
    ## WhichParentName = wPN
    if sx:
        wPN1 = matchKey.ParentKinyaSXCol.apply(lambda x: x.rstrip('[1-4]_sx'))
    else:
        wPN1 = matchKey.ParentKinyaCol.apply(lambda x: x.rstrip('[1-4]'))
    
    wPN1 = ['hohname' if name == 'hkinyaname' else 'parentname' for name in wPN1]
    sub1 = sub1.assign(wPN1 = wPN1)
    sub1 = sub1.assign(parentname = sub1.apply(lambda x: x[x.wPN1], axis=1))
    
    ## Select and rename columns
    sub1 = sub1.assign(dob_est = dob1)
    sub1 = sub1.rename(columns=renameTopDict)[toKeep + ['redcap_event_name']]
    
    # =============================================================================
    # HH Survey Data
    # =============================================================================
    
    ## Sync hh survey data with matchkey using index based on match keys, reset index for merge        
    sub2 = df2.reindex(index = index2).reset_index().fillna(' ')
    
    ## Display child/parent name variant based off max sum of similarity scores
    ## Parent names are pair-dependent, i.e. kinyaname1 and othername1 considered jointly
    ## Child is assumed to be same person, i.e kinyaname1 can mix with othername1,2,3
    extractVariant = lambda x: re.sub('[^1-9]','', x)
    if sx:
        wCKN = ['ckinyaname' + extractVariant(x) for x in matchKey.ChildKinyaSXCol]
        wCON = ['cothername' + extractVariant(x) for x in matchKey.ChildOtherSXCol]
        wPKN2 = ['pkinyaname' + extractVariant(x) for x in matchKey.ParentKinyaSXCol]
        wPON2 = ['pothername' + extractVariant(x) for x in matchKey.ParentKinyaSXCol]

    else:
        wCKN = ['ckinyaname' + extractVariant(x) for x in matchKey.ChildKinyaCol]
        wCON = ['cothername' + extractVariant(x) for x in matchKey.ChildOtherCol]
        wPKN2 = ['pkinyaname' + extractVariant(x) for x in matchKey.ParentKinyaCol]
        wPON2 = ['pothername' + extractVariant(x) for x in matchKey.ParentKinyaCol]
        
    sub2 = sub2.assign(wCKN = wCKN, wCON = wCON, wPKN2 = wPKN2, wPON2 = wPON2)

    childname = sub2.apply(lambda x: str(x[x.wCKN]) + ' ' + str(x[x.wCON]), axis=1)
    parentname = sub2.apply(lambda x: str(x[x.wPKN2]) + ' ' + str(x[x.wPON2]), axis=1)
    sub2 = sub2.assign(PATIENTNAME_ = childname, PARENTNAME_ = parentname)
    
    ## Select child data columns and rename
    sub2 = sub2.rename(columns=renameBottomDict)[toKeep + ['hhid']]
    
    # =============================================================================
    # Combine both dataframes  
    # =============================================================================
    merged = combineTopBottom(sub1, sub2)
    
    ## Add confidence score
    merged = merged.assign(confScore = confScore)
    
    # =============================================================================
    # Write to Excel
    # =============================================================================

    ## Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(fuzePath, engine='xlsxwriter')

    ## Convert the dataframe to an XlsxWriter Excel object.
    merged.to_excel(writer, sheet_name='pairs') 
    
    ## Get the xlsxwriter objects from the dataframe writer object.
    workbook  = writer.book
    worksheet = writer.sheets['pairs']
    
    ## Set column widths
    worksheet.set_column('B:C', 35)
    worksheet.set_column('D:E', 13)
    worksheet.set_column('F:G', 16)
    worksheet.set_column('H:H', 22)
    worksheet.set_column('N:N', 18)
    worksheet.set_column('P:P', 20)
    
    ## Background color for every other row
    ## Bottom border at end of succession of same clinic IDs
    
    ## Count up how many times a clinic id appears in manual review
    clinicIDs = list(merged.clinicID) # already sorted
    boxDict = {}
    for i, idValue in enumerate(clinicIDs):
        
        if idValue in boxDict.keys():
            boxDict[idValue]['rows'] += 1
            
        else:
            boxDict.update({idValue: {'index': i, 'rows': 1}})
    
    ## Get indeces for writing top & bottom borders for clinic ids appearing more than once
    topBorderRows = [value['index']+1 for value in boxDict.values() if value['rows'] > 1]
    bottomBorderRows = [value['index'] + value['rows'] for value in boxDict.values() if value['rows'] > 1]
    
    ## Formats
    grey = workbook.add_format({'fg_color': '#C0C0C0', 'text_wrap': True})
    wrap = workbook.add_format({'text_wrap': True})
    grey_top = workbook.add_format({'fg_color': '#C0C0C0', 'text_wrap': True, 'top': 3, 'italic': True})
    wrap_top = workbook.add_format({'text_wrap': True, 'top': 3, 'italic': True})
    grey_bottom = workbook.add_format({'fg_color': '#C0C0C0', 'text_wrap': True, 'bottom': 3, 'italic': True})
    wrap_bottom = workbook.add_format({'text_wrap': True, 'bottom': 3, 'italic': True})
    
    ## Apply formats
    for row in range(1, len(merged) + 1, 2):
        
        if row in topBorderRows:
            worksheet.set_row(row, cell_format=grey_top)
        elif row in bottomBorderRows:
            worksheet.set_row(row, cell_format=grey_bottom)
        else:
            worksheet.set_row(row, cell_format=grey)
            
        if row+1 in topBorderRows:
            worksheet.set_row(row+1, cell_format=wrap_top)
        elif row+1 in bottomBorderRows:
            worksheet.set_row(row+1, cell_format=wrap_bottom)
        else:
            worksheet.set_row(row+1, cell_format=wrap)
        
    workbook.close()

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
# Acceptance level and Fusion
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
acceptFile = desk + '0820/accept0820.xlsx'
manReviewFile = desk + '0820/man_review0820.xlsx'
Fuze(clinic, child, matches, acceptFile)
Fuze(clinic, child, possible_matches, manReviewFile)

acceptFileSX = path + '0820/acceptSX0820.xlsx'
manReviewFileSX = path + '0820/man_reviewSX0820.xlsx'
Fuze(clinic, child, matches, acceptFileSX, sx=True)
Fuze(clinic, child, possible_matches, manReviewFileSX, sx=True)

# =============================================================================
# Plot confidence score distribution
# =============================================================================

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

# =============================================================================
# Save weights
# =============================================================================
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
# =============================================================================
# =============================================================================
# Old Code
# =============================================================================
# =============================================================================

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



#def Fuze(df1, df2, matchKey, fuzePath):
#    
#    ## Match key
#    index1 = matchKey.level_0
#    index2 = matchKey.level_1
#    confScore = matchKey.confScore
#         
#    ## Subset data using index based on match keys, then reset index for merge
#    sub1 = df1.reindex(index = index1)\
#        .reset_index().rename(columns=renameClinicDict)
#    sub2 = df2.reindex(index = index2)\
#        .reset_index().rename(columns=renameBottomDict)
#        
#    childname = sub2['_childKinyaname'].astype(str) + ' ' + sub2['_childOthername']
#    parentname = sub2['_parentKinyaname'].astype(str) + ' ' + sub2['_parentOthername']
#    sub2 = sub2.assign(PATIENTNAME_ = childname)
#    sub2 = sub2.assign(PARENTNAME_ = parentname)
#    sub2 = sub2.rename(columns=renameBottomDict)[toKeepTop]
#    sub1 = sub1[toKeepTop]
#    
#    merged = pd.DataFrame()
#    for (i1, row1), (i2, row2) in zip(sub1.iterrows(), sub2.iterrows()):
#    
#        t1 = row1.to_frame().transpose()
#        t2 = row2.to_frame().transpose()
#        
#        merged = merged.append(t1, ignore_index=True)
#        merged = merged.append(t2, ignore_index=True)
#        
#    
#    confScoreDouble = []
#    for score in confScore:
#        confScoreDouble.append(int(score))
#        confScoreDouble.append(int(score))
#    
#    merged = merged.assign(confScore = confScoreDouble)    
#    merged = merged.fillna('')
#    merged = merged.applymap(lambda s:s.upper() if type(s) == str else s)
#    merged = merged.replace('NA','')
#    merged = merged.assign(DOB_ = merged.DOB_.\
#            apply(lambda x: x.strftime('%B, %Y') if x is not pd.NaT else x))
#    
#    # Create a Pandas Excel writer using XlsxWriter as the engine.
#    writer = pd.ExcelWriter(fuzePath, engine='xlsxwriter')
#
#    # Convert the dataframe to an XlsxWriter Excel object.
#    merged.to_excel(writer, sheet_name='pairs')    
#    
#    # Get the xlsxwriter objects from the dataframe writer object.
#    workbook  = writer.book
#    worksheet = writer.sheets['pairs']
#    
#    worksheet.set_column('D:E', 30)
#    worksheet.set_column('F:G', 13)
#    worksheet.set_column('H:I', 16)
#    worksheet.set_column('J:J', 17)
#    worksheet.set_column('P:P', 18)
#    
#    greyTop = workbook.add_format({'fg_color': '#C0C0C0'}) #background color
#    greyTop.set_top()
#    greyBottom = workbook.add_format({'fg_color': '#C0C0C0'})
#    greyBottom.set_bottom()
#    dateFormat = workbook.add_format({'num_format': 'mmm yyyy'})
#    
#    for row in range(1, len(merged), 4):
#        worksheet.set_row(row, cell_format=greyTop)
#        worksheet.set_row(row + 1, cell_format=greyBottom)
#    
#    workbook.close()


# =============================================================================
# Old Weights
# =============================================================================
#weights = {
#
### Parent and Child Names, built-in Python string comparison
#'ckinyaname': 20,       'ckinyaname_cor': 0,    'ckinyaname_sx': 0,     
#'cothername': 15,       'cothername_sx': 0,
#'pkinyaname': 0,        'pkinyaname_cor': 0,    'pkinyaname_sx': 0,
#'pothername': 0,        'pothername_sx': 0,
#'PHkinyaname': 10,      'PHothername': 4,
#'PHkinyaname_sx': 0,    'PHothername_sx': 0,
#
### Parent and Child Names, FRIL style edit distance comparison
#'ckinyaname_FR': 0,     'ckinyaname_corFR': 0,  'ckinyaname_sx_FR': 0,
#'cothername_FR': 0,     'cothername_sx_FR': 0,
#'pkinyaname_FR': 0,     'pkinyaname_corFR': 0,  'pkinyaname_sx_FR': 0,
#'pothername_FR': 0,     'pothername_sx_FR': 0,
#
### Mutuelle Number
#'mdist': 2, 'mhf': 6, 'mhh': 13, 'mindv': 5,
#
### Location
#'district': 0, 'sector': 4, 'cell': 5, 'village': 8, 'geo': 2, 'adjacentVillage': 0,
#
### Other
#'gender': 2, 'dob': 4}
#
#weightsSX = {
#
### Parent and Child Names, built-in Python string comparison
#'ckinyaname': 0,        'ckinyaname_cor': 0,    'ckinyaname_sx': 20,
#'cothername': 0,        'cothername_sx': 15, 
#'pkinyaname': 0,        'pkinyaname_cor': 0,    'pkinyaname_sx': 0,
#'pothername': 0,        'pothername_sx': 0,
#'PHkinyaname': 0,       'PHothername': 0,
#'PHkinyaname_sx': 10,   'PHothername_sx': 4,
#
### Parent and Child Names, FRIL style edit distance comparison
#'ckinyaname_FR': 0,     'ckinyaname_corFR': 0,  'ckinyaname_sx_FR': 0,
#'cothername_FR': 0,     'cothername_sx_FR': 0,
#'pkinyaname_FR': 0,     'pkinyaname_corFR': 0,  'pkinyaname_sx_FR': 0,
#'pothername_FR': 0,     'pothername_sx_FR': 0,
#
### Mutuelle Number
#'mdist': 2, 'mhf': 6, 'mhh': 13, 'mindv': 5,
#
### Location
#'district':0, 'sector': 4, 'cell': 5, 'village': 8, 'geo': 2, 'adjacentVillage': 0,
#
### Other
#'gender': 2, 'dob': 4}