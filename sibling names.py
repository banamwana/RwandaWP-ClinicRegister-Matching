# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:21:00 2019

@author: gprenti
"""

import pandas as pd
import os
import numpy as np
import re
#from jellyfish import damerau_levenshtein_distance as damLev #C Implementation
from jellyfish._jellyfish import damerau_levenshtein_distance as damLev #Pyton implementaion
import json

# =============================================================================
# Import DATA
# =============================================================================

if os.name == 'posix':
    path = '/Users/graemepm/Box Sync/EGHI Record Linkage/Data/Raw UBD/From Kyle/'
    desk = '/Users/graemepm/Desktop/'
    
else:
    path = 'C:/Users/gprenti/Box Sync/EGHI Record Linkage/Data/Raw UBD/From Kyle/'
    desk = 'C:/Users/gprenti/Desktop/'
    
files = [file for file in os.listdir(path) if re.search('xlsx', file)]

ubd = pd.DataFrame()
for file in files:
    
    data = pd.read_excel(path + file)
    
    ubd = ubd.append(data)
  
# =============================================================================
# 'DistrictCode', 'DistrictName', 'SectorID', 'SectorName', 'CellID',
# 'CellName', 'VillageCode', 'VillageName', 'VillageType', 'HeadID',
# 'HeadofHouse', 'FirstName', 'LastName', 'NID', 'Age', 'Gender',
# 'Ubudhehe2012', 'Ubudehe2010', 'LandOwner', 'NbrDependants',
# 'NbrDependants18', 'AbleToWork', 'Retired', 'Sick', 'Handicaped',
# 'Studies', 'OtheReasons'
# =============================================================================

'''
Each HeadID is unique
'''
#households = set([(r['SectorName'], r['HeadID']) for i, r in ubd.iterrows()])
#len(households)
#len(set(ubd.HeadID))

# =============================================================================
# Distance function
# =============================================================================

def dist(s1, s2):
    return 1 - damLev(str(s1), str(s2)) / max([len(str(s1)), len(str(s2))])

# =============================================================================
# Compare within household names
# =============================================================================

households = list(set(ubd.HeadID))

sameName = {}
for house in households:
    
    subset = ubd[ubd.HeadID == house]
    
    otherDict = {i: r['LastName'] for i, r in subset.iterrows()}

    indexList = list(subset.index)
    kinyaList = list(subset.FirstName)
    
    while len(kinyaList) > 1:
        
        ## Each list will remain same length with every subtraction
        takenIndex =    indexList.pop(0)
        takenName =     kinyaList.pop(0)
        
        ## Iterate over the reduced kinyaname list
        ## Compare takenName to each remaining name
        for i, name in enumerate(kinyaList):
            
            score = dist(takenName, name)
            
            if score > 0.8:
                
                ## Get original index of matched name
                matchedIndex = indexList[i]
                
                line = (house, takenIndex, takenName, otherDict[takenIndex],
                        matchedIndex, name, otherDict[matchedIndex])
                
                sameName.update({line: score})
                

sameName_df = pd.DataFrame(list(sameName.keys()))
sameName_df = sameName_df.assign(score = list(sameName.values()))
sameName_df.columns = ['hh_id', 'index1', 'kinyaname1', 'othername1', 
                       'index2', 'kinyaname2', 'othername2', 'score']

#sameName_df.to_csv(desk + 'sameName.txt', index=False, sep = '\t')
#sameName_df.to_csv(desk + 'sameName.csv', index=False)           

sub = sameName_df[sameName_df.score > 0.85]

same = len(set(sameName_df.hh_id))
total = len(set(ubd.HeadID))

# =============================================================================
# 
# =============================================================================

#households = list(set(ubd.HeadID))
#
#sameName = {}
#for house in households[:100]:
#    
#    names = list(ubd[ubd.HeadID == house].FirstName)
#    
#    while len(names) > 1:
#        
#        taken = names[0]
#        names.pop(0)
#        
#        for name in names:
#            
#            score = dist(taken, name)
#            
#            if score > 0.8:
#                
#                sameName.update({(taken, name, house): score})
#                
#
#sameName_df = pd.DataFrame(list(sameName.keys()))
#sameName_df = sameName_df.assign(score = list(sameName.values()))
#sameName_df.columns = ['name1', 'name2', 'hh_id', 'simScore']
