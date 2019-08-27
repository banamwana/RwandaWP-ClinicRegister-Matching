# -*- coding: utf-8 -*-
"""
Script to perform feature comparison of houshold survey data
and digitized health facility register and chw monthly report data
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
from collections import Counter
# =============================================================================
# Directories
# =============================================================================
if os.name == 'posix':
    home = '/Users/graemepm/'
else:
    home = 'C:/Users/gprenti/'
    
path = home + 'Box Sync/EGHI Record Linkage/FRIL Matching Iterations/'
path_data = home + 'Box Sync/EGHI Record Linkage/Data/'
path_NB = home + 'Box Sync/EGHI Record Linkage/Data/Name Banks/'
desk = home + 'Desktop/'
    
childFile = path + 'Manual Cleaning/child_keyMCLEAN.txt'
clinicFile = path + '0820/clinic0806.txt'
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

child = pd.read_csv(childFile, sep = '\t', parse_dates=['dob'], dtype = readChildDict)
clinicCols = list(pd.read_csv(clinicFile, sep = '\t', nrows =1))
clinicCols.remove('index')
clinic = pd.read_csv(clinicFile, sep = '\t', parse_dates=['patientvisitdate', 'dob_est'], usecols=clinicCols)
clinicIter = pd.read_csv(clinicFile, sep = '\t', parse_dates=['patientvisitdate', 'dob_est'], usecols=clinicCols, chunksize=2000)

## Import name bank and create counter
with open(path_NB + 'kinyanamesLarge.txt', 'r') as f:
    names = json.loads(f.read())
    
## Import dictionary of villages within 5km of health facilities
with open(path_data + 'Village Lists/WP_hf_adjacent_villages.txt', 'r') as f:
    adVillages = json.loads(f.read())

adVillages = {key: [tuple(vill) for vill in villages] 
              for key, villages in adVillages.items()}

kinyaCounter = Counter(names)
total = sum(kinyaCounter.values())
kinyaCounter = {key: (value / total) * 100000 for key, value in kinyaCounter.items()}
#kinyanameFreqs = keys = np.fromiter(kinyaCounter.values(), dtype=float)
#np.percentile(kinyanameFreqs, 95)

## Make tuple of sector coordinates
#child = child.assign(sectCentr = child[['sectLat', 'sectLong']].apply(tuple, axis=1))
#clinic = clinic.assign(sectCentr = clinic[['sectLat', 'sectLong']].apply(tuple, axis=1))

# =============================================================================
# Make tuple of District, Sector, Cell, Village
# Keep in comparison script because can't store tuples in csv/txt
# =============================================================================
childVillTuples = [(row['district'], row['sector'], row['cell'], row['village'])
                        for i, row in child.iterrows()]

clinicVillTuples = [(row['district_clean'], row['sector_clean'], 
                     row['cell_clean'], row['village_clean'])
                        for i, row in clinic.iterrows()]

child = child.assign(villTuple = childVillTuples)
clinic = clinic.assign(villTuple = clinicVillTuples)

clinicCoord = [(row['latitude'], row['longitude'])
                        for i, row in clinic.iterrows()]
childGeo = [(row['lat'], row['long'])
                        for i, row in child.iterrows()]

clinic = clinic.assign(Coord = clinicCoord)
child = child.assign(Geo = childGeo)

#CHW data doesn't have facility ids
clinic = clinic.assign(facilityid = clinic.facilityid.fillna(99.0).astype(str)\
                       .apply(lambda x: x.rstrip('0').rstrip('.')))

# =============================================================================
# Final Cleaning
# =============================================================================

#child = pd.concat([child.select_dtypes(exclude='object'),
#                   child.select_dtypes('object').fillna('')], axis=1)
#clinic = pd.concat([clinic.select_dtypes(exclude='object'),
#                    clinic.select_dtypes('object').fillna('')], axis=1)

## Subset datasets to run test quickly
child50 = child.sample(50)
clinic50 = clinic.sample(50)

# =============================================================================
# NA handling
# =============================================================================
#1 - damLev(' ', ' ') / np.max([len(' '), len(' ')]) # returns nan
#1 - damLev('', '') / np.max([len(''), len('')]) # returns 1.0
#1 - damLev(np.NaN, np.NaN) / np.max([len(np.NaN), len(np.NaN)]) # throws error
#clinic = clinic[clinic.pkinyaname.isna()] # for testing out NA handling in string comparison
# Built in damereau-levenshtein distanace returns ('','') = 0

# =============================================================================
# Custom Algorithms
# =============================================================================
 
class DateAppr(BaseCompareFeature):
    
    '''
    Compare dates and similarity score with supplied day margin
    '''

    def _compute_vectorized(self, d1, d2, day_margin = 30):
        
        conc = pd.Series(list(zip(d1, d2)))
        
        def compareDate(candidate):
            
            score = 0
            
            # Absolute time difference in days
            tdays = abs((candidate[0] - candidate[1]).days)
            days_out = np.clip(tdays - day_margin, 0, 100)
            addScore = (100 - days_out)**2 / 100**2
            score += addScore
            #score = np.where(score > 0.9, 1, 0)
            if score is np.NaN or score is None:
                score = 0

            return pd.Series(float(score))
        
        return conc.apply(compareDate).fillna(float(0))

class EuDist(BaseCompareFeature):

    '''
    Calculate euclidian distance between two coordinates
    and return score based on supplied distance range
    '''
    
    def _compute_vectorized(self, c1, c2, distRange = 50):
        
        conc = pd.Series(list(zip(c1, c2)))
        
        def applyEuDist(candidate):
            
            score = 0
            
            C1 = candidate[0]
            C2 = candidate[1]
            
            if np.NaN in C1 or np.NaN in C2:
                return pd.Series(float(0))
            
            dist = sqrt((C1[0] - C2[0])**2 + (C1[1] - C2[1])**2)
            distBeyond = pd.Series((dist - distRange)).clip(0, 500)
            addScore = (500 - distBeyond)**2 / 500**2
            score += addScore
            if score is np.NaN:
                score = 0
        
            return pd.Series(score.astype(float))
        
        return conc.apply(applyEuDist).fillna(float(0))
    
class EditDistFRIL(BaseCompareFeature):
    
    '''
    Implement FRIL methodology of Damereau-Levenshtein distance
    using supplied approve and disapprove levels
    '''
    
    def _compute_vectorized(self, s1, s2, dLevel = 0.4, aLevel = 0.2):
        
        s1 = s1.fillna('')
        s2 = s2.fillna('')
        
        conc = pd.Series(list(zip(s1, s2)))
        
        def eScoreFRIL(candidate):
            
            S1 = candidate[0].upper()
            S2 = candidate[1].upper()
            
            eDist = damLev(S1, S2)
            maxLen = max([len(S1), len(S2)])
            
            if maxLen == 0:
                return pd.Series(float(0))
        
            if eDist > (dLevel * maxLen):
                return pd.Series(float(0))
            
            elif eDist < (aLevel * maxLen):
                return pd.Series(float(1))
            
            else:
                score = (dLevel * maxLen - eDist) / ((dLevel - aLevel) * maxLen)
                return pd.Series(float(score))
            
        return conc.apply(eScoreFRIL).fillna(float(0))
        
class AdjacentVillage(BaseCompareFeature):
    
    '''
    Look up village in list of valid villages for input facility ID
    and return score of 1 for match and 0 for non-match
    '''
    
    def _compute_vectorized(self, l1, l2):
        
        conc = pd.Series(list(zip(l1, l2)))
        
        def lookupVillage(candidate):
            
            facilityID = candidate[0]
            village = tuple(candidate[1])
        
            if facilityID in adVillages.keys():
            
                validVillages = adVillages[facilityID]
            
                if village in validVillages:
                    return pd.Series(float(1))
                else:
                    return pd.Series(float(0))
            else:
                return pd.Series(float(0))
        
        return conc.apply(lookupVillage).fillna(float(0))
    
    
class DamLev(BaseCompareFeature):
    
    '''
    Replica of standard damereau-levenshtein distance in record linkage
    but more tolerant to invalid inputs
    '''
    
    def _compute_vectorized(self, s1, s2):
        
        conc = pd.Series(list(zip(s1, s2)))
        
        def damerau_levenshtein_apply(x):
    
            try:
                return 1 - damLev(x[0], x[1]) / np.max([len(x[0]), len(x[1])])
            except Exception as err:
                if pd.isnull(x[0]) or pd.isnull(x[1]):
                    return np.nan
                if type(x[0]) is not str or type(x[1]) is not str:
                    return np.nan
                else:
                    raise err
    
        return conc.apply(damerau_levenshtein_apply)

        
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
# Comparison Features
# =============================================================================

'''
    .add algorithms need to come after built-in algorithms
    or else the feature comparison index will default to 1
'''

## String comparison options
# ‘jaro’, ‘jarowinkler’, ‘levenshtein’, ‘damerau_levenshtein’, 
# ‘qgram’, ‘cosine’, ‘smith_waterman’, ‘lcs’

compare = recordlinkage.Compare()

## Location
compare.exact('district_clean', 'district', label = 'district')
compare.exact('sector_clean', 'sector', label = 'sector')
compare.exact('cell_clean', 'cell', label = 'cell')
compare.exact('village_clean', 'village', label = 'village')

## Built-in damereau-levenshtein distance
## Child Names
#compare.string('ckinyaname', 'ckinyaname1',
#                method='damerau_levenshtein', label = 'ckinyaname')
##compare.string('ckinyaname_cor', 'ckinyaname1',
##                method='damerau_levenshtein', label = 'ckinyaname_cor')
#compare.string('ckinyaname_sx', 'ckinyaname1_sx', 
#                method='damerau_levenshtein', label = 'ckinyaname_sx')
#compare.string('cothername', 'cothername1',
#                method='damerau_levenshtein', label = 'cothername')
#compare.string('cothername_sx', 'cothername1_sx',
#                method='damerau_levenshtein', label = 'cothername_sx')

## Parent Names
#compare.string('pkinyaname', 'pkinyaname1',
#                method='damerau_levenshtein', label = 'pkinyaname')
##compare.string('pkinyaname_cor', 'pkinyaname1',
##                method='damerau_levenshtein', label = 'pkinyaname_cor')
#compare.string('pkinyaname_sx', 'pkinyaname1_sx', 
#               method='damerau_levenshtein', label = 'pkinyaname_sx')
#compare.string('pothername', 'pothername1',
#                method='damerau_levenshtein', label = 'pothername')
#compare.string('pothername_sx', 'pothername1_sx',
#                method='damerau_levenshtein', label = 'pothername_sx')

## Modified damereau-levenshtein distance
## Consider iterating, then could build rename dictionary automatically

## Child Names 1
compare.add(DamLev('ckinyaname', 'ckinyaname1')) #4 - ckinyaname1
compare.add(DamLev('ckinyaname_sx', 'ckinyaname1_sx')) #5 - ckinyaname1_sx
compare.add(DamLev('cothername', 'cothername1')) #6 - cothername1
compare.add(DamLev('cothername_sx', 'cothername1_sx')) #7 - cothername1_sx

## Child Names 2
compare.add(DamLev('ckinyaname', 'ckinyaname2')) #8 - ckinyaname2
compare.add(DamLev('ckinyaname_sx', 'ckinyaname2_sx')) #9 - ckinyaname2_sx
compare.add(DamLev('cothername', 'cothername2')) #10 - cothername2
compare.add(DamLev('cothername_sx', 'cothername2_sx')) #11 - cothername2_sx

## Child Names 3
compare.add(DamLev('ckinyaname', 'ckinyaname3')) #12 - ckinyaname3
compare.add(DamLev('ckinyaname_sx', 'ckinyaname3_sx')) #13 - ckinyaname3_sx
compare.add(DamLev('cothername', 'cothername3')) #14 - cothername3
compare.add(DamLev('cothername_sx', 'cothername3_sx')) #15 - cothername3_sx

## Parent Names 1
compare.add(DamLev('pkinyaname', 'pkinyaname1')) #16 - pkinyaname1
compare.add(DamLev('pothername', 'pothername1')) #17 - pothername1
compare.add(DamLev('pkinyaname_sx', 'pkinyaname1_sx')) #18 - pkinyaname1_sx
compare.add(DamLev('pothername_sx', 'pothername1_sx')) #19 - pothername1_sx

## Parent Names 2
compare.add(DamLev('pkinyaname', 'pkinyaname2')) #20 - pkinyaname2
compare.add(DamLev('pothername', 'pothername2')) #21 - pothername2
compare.add(DamLev('pkinyaname_sx', 'pkinyaname2_sx')) #22 - pkinyaname2_sx
compare.add(DamLev('pothername_sx', 'pothername2_sx')) #23 - pothername2_sx

## Parent Names 3
compare.add(DamLev('pkinyaname', 'pkinyaname3')) #24 - pkinyaname3
compare.add(DamLev('pothername', 'pothername3')) #25 - pothername3
compare.add(DamLev('pkinyaname_sx', 'pkinyaname3_sx')) #26 - pkinyaname3_sx
compare.add(DamLev('pothername_sx', 'pothername3_sx')) #27 - pothername3_sx

## Parent Names 4
compare.add(DamLev('pkinyaname', 'pkinyaname4')) #28 - pkinyaname
compare.add(DamLev('pothername', 'pothername4')) #29 - pothername
compare.add(DamLev('pkinyaname_sx', 'pkinyaname4_sx')) #30 - pkinyaname_sx
compare.add(DamLev('pothername_sx', 'pothername4_sx')) #31 - pothername_sx

## HOH Names 1
compare.add(DamLev('hkinyaname', 'pkinyaname1')) #32 - hkinyaname
compare.add(DamLev('hkinyaname_sx', 'pkinyaname1_sx')) #33 - hkinyaname_sx
compare.add(DamLev('hothername', 'pothername1')) #34 - hothername
compare.add(DamLev('hothername_sx', 'pothername1_sx')) #35 - hothername_sx

## HOH Names 2
compare.add(DamLev('hkinyaname', 'pkinyaname2')) #36 - hkinyaname
compare.add(DamLev('hkinyaname_sx', 'pkinyaname2_sx')) #37 - hkinyaname_sx
compare.add(DamLev('hothername', 'pothername2')) #38 - hothername
compare.add(DamLev('hothername_sx', 'pothername2_sx')) #39 - hothername_sx

## HOH Names 3
compare.add(DamLev('hkinyaname', 'pkinyaname3')) #40 - hkinyaname
compare.add(DamLev('hkinyaname_sx', 'pkinyaname3_sx')) #41 - hkinyaname_sx
compare.add(DamLev('hothername', 'pothername3')) #42 - hothername
compare.add(DamLev('hothername_sx', 'pothername3_sx')) #43 - hothername_sx

## HOH Names 4
compare.add(DamLev('hkinyaname', 'pkinyaname4')) #44 - hkinyaname
compare.add(DamLev('hkinyaname_sx', 'pkinyaname4_sx')) #45 - hkinyaname_sx
compare.add(DamLev('hothername', 'pothername4')) #46 - hothername
compare.add(DamLev('hothername_sx', 'pothername4_sx')) #47 - hothername_sx

## Geolocation comparison
compare.geo(left_on_lat = 'latitude', left_on_lng = 'longitude',
            right_on_lat = 'lat', right_on_lng = 'long',
            label = 'geo', method = 'linear')

#compare.add(AdjacentVillage('facilityid', 'villTuple'))

## Mutuelle
compare.add(DamLev('mdist', 'm_dist')) #49 - mdist
compare.add(DamLev('mhf', 'm_hc')) #50 - mhf
compare.add(DamLev('mhh', 'm_hh')), #51 - mhh
compare.add(DamLev('mindv', 'm_indv')) #52 - mindv

## Gender
compare.exact('sexe', 'cgender_chr', label = 'gender')

## Date
compare.add(DateAppr('dob_est', 'dob')) #54 - dob

## Check labels
#leftMissing = [label for label in compare._get_labels_left if label not in clinic.columns]

# =============================================================================
# Indexer - In one go
# =============================================================================

#indexer = recordlinkage.Index()
#indexer.full() # No Blocking
##indexer.block(left_on=('district_clean'), right_on=('district'))
#candidate_links = indexer.index(clinic, child)

#indexer50 = recordlinkage.Index()
#indexer50.full()
#candidate_links50 = indexer50.index(clinic50, child50)

# =============================================================================
# Compute Comparison - In one go
# =============================================================================
#featComp = compare.compute(candidate_links, x = clinic, x_link = child)
#featComp.reset_index(inplace=True)
#featComp.describe()

# =============================================================================
# Function to iteratively implement feature comparison
# ~ 4.5 hours per chunk
# =============================================================================

def compareChunk(Achunk, Bdata):
    
    clinicCoord = [(row['latitude'], row['longitude'])
                        for i, row in Achunk.iterrows()]
    clinicVillTuples = [(row['district_clean'], row['sector_clean'], 
                         row['cell_clean'], row['village_clean'])
                        for i, row in Achunk.iterrows()]
    
    chunkDF = Achunk.assign(Coord = clinicCoord)
    chunkDF = chunkDF.assign(villTuple = clinicVillTuples)

    chunkDF = chunkDF.assign(facilityid = chunkDF.facilityid.fillna(99.0).astype(str)\
                       .apply(lambda x: x.rstrip('0').rstrip('.')))
    
    #Pre-Indexer
    preIndexer = recordlinkage.Index()
    preIndexer.full()
    full_candidate_links = preIndexer.index(chunkDF, Bdata)
    
    #Block on 5km radius adjacent village if Clinic or exact village match if CHW
    comparePreIndex = recordlinkage.Compare()
    comparePreIndex.add(AdjacentVillage('facilityid', 'villTuple'))
    comparePreIndex.exact('villTuple', 'villTuple')
    PreIndex = comparePreIndex.compute(full_candidate_links, x = chunkDF, x_link = Bdata)
    candidate_links = PreIndex[(PreIndex[0] == 1) | (PreIndex[1] == 1)].index
    
    #Feature comparison
    featCompChunk = compare.compute(candidate_links, x = chunkDF, x_link = Bdata)
    featCompChunk.reset_index(inplace=True)
    
    return featCompChunk

#test50 = compareChunk(clinic50, child50)

# =============================================================================
# Compute feature comparison in one go
# =============================================================================

#start = time.time()
#featComp = compareChunk(clinic, child)
#end = time.time()

# =============================================================================
# Compute feature comparison - takes aaawwwhhhile!!
# =============================================================================

featComp = pd.DataFrame()
errorDict = {}
timeDict = {}
for i, chunk in enumerate(clinicIter):
    
    start = time.time()
    
    try:
        featComp = featComp.append(compareChunk(chunk, child))
        
    except Exception as e:
        errorDict.update({'chunk' + str(i): [e, chunk]})
        
    end = time.time()
    timeDict.update({'chunk' + str(i): end - start})

errorMSGDict = {key: value[0] for key, value in errorDict.items()}

# =============================================================================
# Finalize Comparison Output
# =============================================================================

## Add name frequencies (frequency in UBD list)
ckinyaname_CLFreq = [kinyaCounter[clinic.ckinyaname[index]] 
        if clinic.ckinyaname[index] in kinyaCounter.keys() else np.NaN for index in featComp.level_0]
ckinyaname_HHFreq = [kinyaCounter[child.ckinyaname1[index]] 
        if child.ckinyaname1[index] in kinyaCounter.keys() else np.NaN for index in featComp.level_1]

featComp = featComp.assign(ckinyaname_CLFreq = ckinyaname_CLFreq)
featComp = featComp.assign(ckinyaname_HHFreq = ckinyaname_HHFreq)

## Add source (HF or CHW data)
source = clinic[['redcap_event_name']].reset_index()
featComp = pd.merge(featComp, source, how='left', left_on='level_0', right_on='index').drop('index', axis=1)

## Provide label names for comparisons with custom algorithms, dependent on order
#labelDict = {4: 'ckinyaname', 5: 'ckinyaname_sx', 6: 'cothername', 7: 'cothername_sx',
#             8: 'pkinyaname', 9: 'pkinyaname_sx', 10: 'pothername', 11: 'pothername_sx',
#             12: 'hkinyaname', 13: 'hkinyaname_sx', 14: 'hothername', 15: 'hothername_sx',
#             16: 'ckinyaname_FR', 17:  'ckinyaname_sx_FR', 18: 'cothername_FR', 19: 'cothername_sx_FR',
#             20: 'pkinyaname_FR', 21: 'pkinyaname_sx_FR', 22: 'pothername_FR', 23: 'pothername_sx_FR',
#             24: 'hkinyaname_FR', 25: 'hkinyaname_sx_FR', 26: 'hothername_FR', 27: 'hothername_sx_FR',
#             29: 'mdist', 30: 'mhf', 31: 'mhh', 32: 'mindv', 
#             34: 'dob'}

labelDict = { 4: 'ckinyaname1',  5: 'ckinyaname1_sx',  6: 'cothername1',  7: 'cothername1_sx',
              8: 'ckinyaname2',  9: 'ckinyaname2_sx', 10: 'cothername2', 11: 'cothername2_sx',
             12: 'ckinyaname3', 13: 'ckinyaname3_sx', 14: 'cothername3', 15: 'cothername3_sx',
             16: 'pkinyaname1', 17: 'pkinyaname1_sx', 18: 'pothername1', 19: 'pothername1_sx',
             20: 'pkinyaname2', 21: 'pkinyaname2_sx', 22: 'pothername2', 23: 'pothername2_sx',
             24: 'pkinyaname3', 25: 'pkinyaname3_sx', 26: 'pothername3', 27: 'pothername3_sx',
             28: 'pkinyaname4', 29: 'pkinyaname4_sx', 30: 'pothername4', 31: 'pothername4_sx',
             32: 'hkinyaname1', 33: 'hkinyaname1_sx', 34: 'hothername1', 35: 'hothername1_sx',
             36: 'hkinyaname2', 37: 'hkinyaname2_sx', 38: 'hothername2', 39: 'hothername2_sx',
             40: 'hkinyaname3', 41: 'hkinyaname3_sx', 42: 'hothername3', 43: 'hothername3_sx',
             44: 'hkinyaname4', 45: 'hkinyaname4_sx', 46: 'hothername4', 47: 'hothername4_sx',
             49: 'mdist', 50: 'mhf', 51: 'mhh', 52: 'mindv', 54: 'dob'}

#test50.rename(columns = labelDict, inplace=True)
featComp.rename(columns = labelDict, inplace=True)

# =============================================================================
# Codebook
# =============================================================================
codebook =  {'ckinyaname1': 'Clinic child kinyaname compared to 1st variant of hh survey child kinyaname',
             'ckinyaname1_sx': 'Clinic child kinyaname soundex compared to soundex of 1st variant of hh survey child kinyaname',
             'cothername1': 'Clinic child othername compared to 1st variant of hh survey child othername',
             'cothername1_sx': 'Clinic child othername soundex compared to soundex of 1st variant of hh survey child othername',
             'pkinyaname1': 'Clinic parent kinyaname compared to kinyaname of 1st variant name-set of hh survey parent',
             'pkinyaname1_sx': 'Clinic parent kinyaname soundex compared to kinyaname soundex of 1st variant name-set of hh survey parent',
             'pothername1': 'Clinic parent othername compared to othername of 1st variant name-set of hh survey parent',
             'pothername1_sx': 'Clinic parent othername soundex compared to othername soundex of 1st variant name-set of hh survey parent',
             'hkinyaname1': 'Clinic HOH kinyaname compared to kinyaname of 1st variant name-set of hh survey parent',
             'hkinyaname1_sx': 'Clinic HOH kinyaname soundex compared to kinyaname soundex of 1st variant name-set of hh survey parent',
             'hothername1': 'Clinic HOH othername compared to othername of 1st variant name-set of hh survey parent',
             'hothername1_sx': 'Clinic HOH othername soundex compared to othername soundex of 1st variant name-set of hh survey parent',
             'mdist': 'Comparison of district section of mutuelle number',
             'mhf': 'Comparison of health facility section of mutuelle number',
             'mhh': 'Comparison of household section of mutuelle number',
             'mindv': 'Comparison of individual section of mutuelle number',
             'dob': 'Comparison of date of birth',
             'gender': 'Comparison of gender',
             'geo': 'Comparison of hh survey gps median coordinates to health facility coordinates',
             'district': 'Comparison of districts',
             'sector': 'Comparison of sectors',
             'cell': 'Comparison of cells',
             'village': 'Comparison of villages'}

# =============================================================================
# Write to disk (won't write multi-index)
# =============================================================================

featCompFile = path + '0820/featComp0820.txt'
featComp.to_csv(featCompFile, index=False, sep = '\t')

with open(path + '0820/compareDict0820.txt', 'w') as f:
    f.write(json.dumps(timeDict))
    
with open(path + '0820/errorDict0820.txt', 'w') as f:
    f.write(json.dumps(errorMSGDict))
    
codebook_lines = ['{:20} {:105}\n'.format(key + ':', value) for key, value in codebook.items()]
file = open(path + '0820/featComp_codebook0820.txt', 'w')
file.writelines(codebook_lines)
file.close()

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

#featComp = pd.DataFrame()
#for chunk in clinicIter:
#    
#    clinicCoord = [(row['latitude'], row['longitude'])
#                        for i, row in chunk.iterrows()]
#    
#    chunkDF = chunk.assign(Coord = clinicCoord)
#
#    chunkDF = chunkDF.assign(facilityid = chunkDF.facilityid.fillna(99.0).astype(str)\
#                       .apply(lambda x: x.rstrip('0').rstrip('.')))
#    
#    #Indexer
#    indexer = recordlinkage.Index()
#    indexer.full()
#    candidate_links = indexer.index(chunkDF, child)
#    
#    #Feature comparison
#    featCompChunk = compare.compute(candidate_links, x = chunkDF, x_link = child)
#    featCompChunk.reset_index(inplace=True)
#    
#    #Append to featComp dataframe
#    featComp = featComp.append(featCompChunk, ignore_index=True)