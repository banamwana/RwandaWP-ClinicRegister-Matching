# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:53:03 2019

@author: gprenti
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

# =============================================================================
# Import DATA
# =============================================================================

if os.name == 'posix':
    path = '/Users/graemepm/Box Sync/EGHI Record Linkage/FRIL Matching Iterations/'
    desk = '/Users/graemepm/Desktop/'
    
else:
    path = 'C:/Users/gprenti/Box Sync/EGHI Record Linkage/FRIL Matching Iterations/'
    desk = 'C:/Users/gprenti/Desktop/'

childFile = path + '0625/child_key0626.txt'
clinicFile = path + '0625/clinic0626.txt'

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

child = pd.read_csv(childFile, sep = '\t', parse_dates=['min_dob', 'max_dob'], dtype = readChildDict)
clinic = pd.read_csv(clinicFile, sep = '\t', parse_dates=['patientvisitdate', 'dob_est'])
clinic.drop('index', axis=1, inplace=True)

## Make tuple of sector coordinates
#child = child.assign(sectCentr = child[['sectLat', 'sectLong']].apply(tuple, axis=1))
#clinic = clinic.assign(sectCentr = clinic[['sectLat', 'sectLong']].apply(tuple, axis=1))

## Subset datasets to run test quickly
#child = child.sample(50)
#clinic = clinic.sample(50)

# =============================================================================
# Custom Algorithms
# =============================================================================
 
# Date comparison
class DateAppr(BaseCompareFeature):

    def _compute_vectorized(self, d1, d2, day_margin = 0):
        score = 0
        # Absolute time difference in days
        tdays = (d1.dt.day - d2.dt.day).abs()
        days_out = tdays - day_margin
        days_out = days_out.clip(0, 100)
        addScore = (100 - days_out)**2 / 100**2
        score += addScore
        #score = np.where(score > 0.9, 1, 0)
        if score is np.NaN or score is None:
            score = 0

        return pd.Series(score.astype(float))
    
# Euclidian distanace
# Not working because can't store tuples in txt file
class EuDist(BaseCompareFeature):

    def _compute_vectorized(self, c1, c2, distRange = 50):
        score = 0
        
        if np.NaN in c1 or np.NaN in c2:
            return pd.Series(float(0))
        
        dist = sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        distBeyond = pd.Series((dist - distRange)).clip(0, 500)
        addScore = (500 - distBeyond)**2 / 500**2
        score += addScore
        if score is np.NaN:
            score = 0
    
        return pd.Series(score.astype(float))
    
class EditDistFRIL(BaseCompareFeature):
    
    def _compute_vectorized(self, s1, s2, dLevel = 0.4, aLevel = 0.2):
        
        S1 = str(s1)
        S2 = str(s2)
                    
        eDist = damLev(S1, S2)
        maxLen = max([len(S1), len(S2)])
        
        if eDist > dLevel * maxLen:
            return pd.Series(float(0))
        
        if eDist < aLevel * maxLen:
            return pd.Series(float(1))
        
        else:
            score = (dLevel * maxLen - eDist) / ((dLevel - aLevel) * maxLen)
            return pd.Series(float(score))
        
#class EditDistFRIL2(BaseCompareFeature):
#    
#    def _compute_vectorized(self, s1, s2, dLevel = 0.4, aLevel = 0.2):
#        
#        conc = pandas.Series(list(zip(s1, s2)))
#        
#        def EditDistFRIL_apply(x):
#            
#            try:
#                S1 = x[0]
#                S2 = x[1]
#                    
#                eDist = damLev(S1, S2)
#                maxLen = max([len(S1), len(S2)])
#        
#                if eDist > dLevel * maxLen:
#                    return pd.Series(float(0))
#        
#                if eDist < aLevel * maxLen:
#                    return pd.Series(float(1))
#        
#                else:
#                    score = (dLevel * maxLen - eDist) / ((dLevel - aLevel) * maxLen)
#                    return pd.Series(float(score))
#            
#            except:
#                if pandas.isnull(x[0]) or pandas.isnull(x[1]):
#                    return np.nan
#                else:
#                    raise err
#                    
#        
#        return conc.apply(EditDistFRIL_apply)
        
## From Python Record Linkage source code 
#def jaro_similarity(s1, s2):
#
#    conc = pandas.Series(list(zip(s1, s2)))
#
#    def jaro_apply(x):
#
#        try:
#            return jellyfish.jaro_distance(x[0], x[1])
#        except Exception as err:
#            if pandas.isnull(x[0]) or pandas.isnull(x[1]):
#                return np.nan
#            else:
#                raise err
#
#    return conc.apply(jaro_apply)

        
# =============================================================================
# Source Code
# =============================================================================

#source_DF = inspect.getsource(recordlinkage.compare.Geographic)
#print(source_DF)    

# =============================================================================
# Indexer
# =============================================================================

indexer = recordlinkage.Index()

### No Blocking
indexer.full()

### Blocking
#indexer.block(left_on=('district_clean'), right_on=('district'))

candidate_links = indexer.index(clinic, child)

# =============================================================================
# Record Comparison
# =============================================================================

'''
    .add algorithms need to come after built-in algorithms
    or else the feature comparison index will default to 1
'''

## String comparison options
#    [‘jaro’, ‘jarowinkler’, ‘levenshtein’, ‘damerau_levenshtein’, 
#     ‘qgram’, ‘cosine’, ‘smith_waterman’, ‘lcs’]

compare = recordlinkage.Compare()

## Child Names
compare.string('ckinyaname', 'ckinyaname1',
                method='damerau_levenshtein', label = 'ckinyaname')
compare.string('ckinyaname_cor', 'ckinyaname1',
                method='damerau_levenshtein', label = 'ckinyaname_cor')
compare.string('ckinyaname_sx', 'ckinyaname1_sx', 
                method='damerau_levenshtein', label = 'ckinyaname_sx')
compare.string('cothername', 'cothername1',
                method='damerau_levenshtein', label = 'cothername')

## Parent Names
compare.string('pkinyaname', 'pkinyaname1',
                method='damerau_levenshtein', label = 'pkinyaname')
compare.string('pkinyaname_cor', 'pkinyaname1',
                method='damerau_levenshtein', label = 'pkinyaname_cor')
compare.string('pkinyaname_sx', 'pkinyaname1_sx', 
               method='damerau_levenshtein', label = 'pkinyaname_sx')
compare.string('pothername', 'pothername1',
                method='damerau_levenshtein', label = 'pothername')


## FRIL edit distance
## Child Names
compare.add(EditDistFRIL('ckinyaname', 'ckinyaname1', label = 'ckinyaname_FR'))
compare.add(EditDistFRIL('ckinyaname_cor', 'ckinyaname1', label = 'ckinyaname_corFR'))
compare.add(EditDistFRIL('ckinyaname_sx', 'ckinyaname1_sx', label = 'ckinyaname_sxFR'))
compare.add(EditDistFRIL('cothername', 'cothername1', label = 'cothername_FR'))

## Parent Names
compare.add(EditDistFRIL('pkinyaname', 'pkinyaname1', label = 'pkinyaname_FR'))
compare.add(EditDistFRIL('pkinyaname_cor', 'pkinyaname1', label = 'pkinyaname_corFR'))
compare.add(EditDistFRIL('pkinyaname_sx', 'pkinyaname1_sx', label = 'pkinyaname_sxFR'))
compare.add(EditDistFRIL('pothername', 'pothername1', label = 'pothername_FR'))

## Location
#compare.exact('district_clean', 'district', label = 'district')
compare.exact('sector_clean', 'sector', label = 'sector')
compare.exact('cell_clean', 'cell', label = 'cell')
compare.exact('village_clean', 'village', label = 'village')

## Sector centroids
#compare.add(EuDist('sectCentr', 'sectCentr', label = 'sectCentr'))
compare.geo(left_on_lat = 'sectLat', left_on_lng = 'sectLong',
            right_on_lat = 'sectLat', right_on_lng = 'sectLong',
            label = 'geo', method='linear')

## Mutuelle
#compare.exact('mdist', 'm_dist', label = 'mdist')
#compare.exact('mhf', 'm_hc', label = 'mhf')
#compare.exact('mhh', 'm_hh', label = 'mhh')
#compare.exact('mindv', 'm_indv', label = 'mindv')

compare.string('mdist', 'm_dist', method='damerau_levenshtein', label = 'mdist')
compare.string('mhf', 'm_hc', method='damerau_levenshtein', label = 'mhf')
compare.string('mhh', 'm_hh', method='damerau_levenshtein', label = 'mhh')
compare.string('mindv', 'm_indv', method='damerau_levenshtein', label = 'mindv')

## Gender
compare.exact('sexe', 'cgender_chr', label = 'gender')

## Date
compare.add(DateAppr('dob_est', 'min_dob', label = 'dob'))

## Comparison Features
startCompare = time.time()
featComp = compare.compute(candidate_links, x = clinic, x_link = child)
featComp.reset_index(inplace=True)
endCompare = time.time()
featComp.describe()

## Write to disk (won't write multi-index)
featCompFile = path + '0625/featComp0626.txt'
featComp.to_csv(featCompFile, index=False, sep = '\t')

