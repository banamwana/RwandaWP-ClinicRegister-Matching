#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom comparison algorithms
"""
# =============================================================================
# Libraries
# =============================================================================
import recordlinkage
import pandas as pd
import numpy as np
import re
from recordlinkage.base import BaseCompareFeature
from math import sqrt, exp
from jellyfish import damerau_levenshtein_distance as damLev

# =============================================================================
# NA handling
# =============================================================================
#1 - damLev(' ', ' ') / np.max([len(' '), len(' ')]) # returns nan
#1 - damLev('', '') / np.max([len(''), len('')]) # returns 1.0
#1 - damLev(np.NaN, np.NaN) / np.max([len(np.NaN), len(np.NaN)]) # throws error
#clinic = clinic[clinic.pkinyaname.isna()] # for testing out NA handling in string comparison
# Built in damereau-levenshtein distanace returns ('','') = 0

 
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