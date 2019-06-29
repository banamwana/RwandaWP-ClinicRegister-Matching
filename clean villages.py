# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:11:36 2019

@author: gprenti
"""
import pandas as pd
import os
import numpy as np
from difflib import SequenceMatcher as SM
from jellyfish import damerau_levenshtein_distance as damLev
import json

if os.name == 'posix':
    path = '/Users/graemepm/Box Sync/EGHI Record Linkage/Data/'
    desk = '/Users/graemepm/Desktop/'
    path_code = '/Users/graemepm/Box Sync/EGHI Record Linkage/Code/'
    
else:
    path = 'C:/Users/gprenti/Box Sync/EGHI Record Linkage/Data/'
    desk = 'C:/Users/gprenti/Desktop/'
    path_code = 'C:/Users/gprenti/Box Sync/EGHI Record Linkage/Code/'

# =============================================================================
### Create reference list of locations
#file1 = path + 'Village Lists/villages_fromUBD.csv'
#file2 = path + 'Village Lists/p2 - all study villages.csv'
#all_villages = pd.read_csv(file1)
#study_locations = pd.read_csv(file2)

#av = all_villages.applymap(lambda s: s.upper() if type(s) == str else s)
#
#VILLAGES = [(r['District'], r['Sector'], r['Cell'], r['Village']) for i, r in av.iterrows()]
#
#with open(path + 'Village Lists/villageTuples.txt', 'w') as f:
#    f.write(json.dumps(VILLAGES))
# =============================================================================

with open(path + 'Village Lists/villageTuples.txt', 'r') as f:
    villages = json.loads(f.read())
    
VILLAGES = [tuple(village) for village in villages]

### Helper functions
#def distance(a, b):
#    return SM(a = a.upper(), b = b.upper(), autojunk = False).ratio()

def distance(s1, s2):
    return 1 - damLev(str(s1).upper(), str(s2).upper()) / max([len(str(s1)), len(str(s2))])

def findLocation(location):
    
    ## If more than 2 locations empty, return blank
    if sum([l == '' for l in location]) > 2:
        return ('', '', '', '')
    
    ## Cannot guess village if village empty
    if location[3] == '':
        return ('', '', '', '')
    
    comparison = {}
    
    exactOK = [[False, True, True, True],
               [True, False, True, True],
               [True, True, False, True],
               [True, True, True, True]]
    
    for LOC in VILLAGES:
        
        exactCompare = [(a == b) for a, b in zip(location, LOC)]
        
        ## Return location if 3 out of 4 exact matches including village
        if exactCompare in exactOK:
            return LOC
        
        distDist = distance(location[0], LOC[0])
        sectDist = distance(location[1], LOC[1])
        cellDist = distance(location[2], LOC[2])
        villDist = distance(location[3], LOC[3])
        
        sumDist = sum([distDist, sectDist, cellDist, villDist])
        
        comparison.update({LOC: sumDist})
    
    return max(comparison, key=comparison.get)

def emptyNA(s):
    
    if str(s).upper() in ['NA', 'MISSING', 'NOT CLEAR', '', ' ', 'NAN']:
        return ''
    else:
        return s

### Full location cleaning function
def cleanVillages(data):
    
    if len(data.columns) != 4:
        print('invalid input')
        return
    if not all(data.dtypes == 'object'):
        print('invalid input')
        return
      
    data = data.applymap(lambda s: s.upper())
    
    ## Handle NAs
    data = data.applymap(lambda s: emptyNA(s))
    data = data.fillna('')
    
    ## Convert input to list of tuples
    villages = [(r[0], r[1], r[2], r[3]) for i, r in data.iterrows()]
    
    for i, village in enumerate(villages):
        
        if village not in VILLAGES:
            
            ## Replace
            villages[i] = findLocation(village)
            
    newData = pd.DataFrame(villages)
    newData.columns = ['district_clean', 'sector_clean', 'cell_clean', 'village_clean']
    
    return newData