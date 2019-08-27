# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 10:31:28 2019

@author: gprenti
"""
import pandas as pd
import os
import numpy as np
import json
from collections import Counter
from difflib import SequenceMatcher as SeqM
import re
from functools import reduce
from pyjarowinkler import distance as ds
import unicodedata
#import pyreadr

if os.name == 'posix':
    path = '/Users/graemepm/Box Sync/EGHI Record Linkage/Data/'
    desk = '/Users/graemepm/Desktop/'
    with open('/Users/graemepm/Box Sync/Graeme/Name Matching/Data/names.txt', 'r') as f:
        name_list = json.loads(f.read())
    with open('/Users/graemepm/Box Sync/Graeme/Name Matching/Data/names_sec.txt', 'r') as f:
        NAMES_SEC = json.loads(f.read())
    exec(open('/Users/graemepm/Box Sync/Graeme/Name Matching/Code/name-process/dbl_msp_pattern.py').read())
    exec(open('/Users/graemepm/Box Sync/EGHI Record Linkage/Code/mutuelle parser.py').read())
    exec(open('/Users/graemepm/Box Sync/EGHI Record Linkage/Code/soundex.py').read())
else:
    path = 'C:/Users/gprenti/Box Sync/EGHI Record Linkage/Data/' 
    desk = 'C:/Users/gprenti/Desktop/'
    with open('C:/Users/gprenti/Box Sync/Graeme/Name Matching/Data/names.txt', 'r') as f:
        name_list = json.loads(f.read())
    with open('C:/Users/gprenti/Box Sync/Graeme/Name Matching/Data/names_sec.txt', 'r') as f:
        NAMES_SEC = json.loads(f.read())
    exec(open('C:/Users/gprenti/Box Sync/Graeme/Name Matching/Code/name-process/dbl_msp_pattern.py').read())
    exec(open('C:/Users/gprenti/Box Sync/EGHI Record Linkage/Code/mutuelle parser.py').read())
    exec(open('C:/Users/gprenti/Box Sync/EGHI Record Linkage/Code/soundex.py').read())
    
# Create frequency dictionary of cleaned UBD names
NAMES = Counter(name_list)

## Import DATA
files = os.listdir(path)

#with open(path + files[0],encoding='ISO-8859-1') as f:
#    facility = pd.read_csv(f)
    
#with open(path + files[1],encoding='ISO-8859-1') as f:
#    hhsurvey = pd.read_csv(f)
villages = pd.read_csv(path + 'Village Lists/p2 - all study villages.csv')
villages = villages.drop_duplicates()

hhsurvey = pd.read_stata(path + 'HH Survey Data/hhsurvey_child_long.dta')
centFile = path + 'Village Lists/sector_centroids.csv'
centroids = pd.read_csv(centFile)

# =============================================================================
# HH Survey Variables
#
# a5 - village ID (note villages in health facility data haven't been matched to an ID yet.  
# a7  household ID
# pc1 - parent name Rwandan
# pc2 - parent name  French/English
# pc2n - parent nickname
# childid  - child ID - first 6 digits are household ID
# genderc  (forgot we can use for matching!)
# dobc  -child age  (forgot we can use for matching!)
# cd2c  - child name Rwandan 
# cd2ac - child name French/English
# cd2bc - child nickname
# cd8bc - mutuelle number
# =============================================================================

child_keys = list(set(hhsurvey.childid))
#pat = re.compile('[a-z]', re.I)
#[key for key in child_keys if pat.search(key)]
#[key for key in child_keys if key == '88']

# Standardize missing v

def str_score(x, y):
    #s = SeqM(isjunk=None, a= x, b = y, autojunk = False).ratio()
    s = ds.get_jaro_distance(x, y, winkler=False)
    return s

#a = 'florence'
#b = 'frolence'
#
#a = 'monique'
#b = 'monica'
#
#a = 'uzayisaba'
#b = 'uzayisenga'
#
#str_score(a, b)
    
def strip_accents(name):
    if re.search('[^a-z]', name):
        try:
            name = unicode(text, 'utf-8')
        except NameError: # unicode is a default on python 3 
            pass
        name = unicodedata.normalize('NFD', name)\
               .encode('ascii', 'ignore')\
               .decode("utf-8")
        return str(name)
    else:
        return name
    
def prob_kinya(name, N=sum(NAMES.values())): 
    "Probability of `name`."
    return NAMES[name] / N

def prob_sec(name, N=sum(NAMES_SEC.values())): 
    "Probability of `name`."
    return NAMES[name] / N

def return_names(childid, name_col, df = hhsurvey, cut = 0.8, sortkey = prob_kinya):
    subset = df[df['childid'] == childid]
    
    names = [str(name).lower()          for name in subset[name_col]]
    names = [re.sub('0', 'o', name)     for name in names]
    names = [strip_accents(name)        for name in names if re.search('[a-z]', name)] # remove name if no letters
    names = [name.replace('n/a', '')    for name in names]
    names = [re.sub(r'\\', '', name)    for name in names] # '\'
    names = [re.sub('\d+', '', name)    for name in names] # remove numbers
    names = [re.sub('(?<!n) (?!.)', '', name) for name in names] # repace white space if not between words
    names = [str(name) for name in list(set(names)) if str(name)]
    if names:
        if len(names) > 1:
            nameset = names
            names.sort(key=sortkey) # sort names by frequency in UBD namelist
            i = 0
            searched = []
            while (i+1) <= len(nameset):
                a = nameset.pop(i) # removes ith item from list and returns it
                scores = {name: str_score(a, name) for name in nameset}
                nameset = [name for name in nameset if scores[name] < cut]
                searched.append(a)
                #i += 1 #recycle the same index since first element has been removed
            cleaned = searched + nameset
            cleaned.sort(key=sortkey)
            return cleaned
        else:
            return names
    else:
        return [None]

## Parent Kinyarwanda Name
parent_kinya = {key: return_names(key, 'pc1') for key in child_keys}
parent_kinya_df = pd.DataFrame.from_dict(parent_kinya, orient='index')
parent_kinya_df.columns = ['pc11', 'pc12', 'pc13']

## Parent Other Name
parent_other = {key: return_names(key, 'pc2', sortkey = prob_sec) for key in child_keys}
parent_other_df = pd.DataFrame.from_dict(parent_other, orient='index')
parent_other_df.columns = ['pc21', 'pc22', 'pc23', 'pc24']

## Parent NickName
parent_nick = {key: return_names(key, 'pc2n') for key in child_keys}
parent_nick_df = pd.DataFrame.from_dict(parent_nick, orient='index')
parent_nick_df.columns = ['pc2n']

## Child Name Rwandan
child_kinya = {key: return_names(key, 'cd2c') for key in child_keys}
child_kinya_df = pd.DataFrame.from_dict(child_kinya, orient='index')
child_kinya_df.columns = ['cd2c1', 'cd2c2', 'cd2c3']

## Child Name Other
child_other = {key: return_names(key, 'cd2ac', sortkey = prob_sec) for key in child_keys}
child_other_df = pd.DataFrame.from_dict(child_other, orient='index')
child_other_df.columns = ['cd2ac1', 'cd2ac2', 'cd2ac3']

## Child Nick
child_nick = {key: return_names(key, 'cd2bc') for key in child_keys}
child_nick_df = pd.DataFrame.from_dict(child_nick, orient='index')
child_nick_df.columns = ['cd2bc1', 'cd2bc2', 'cd2bc3']

# =============================================================================
#### Old version ####
# Mutuelle
#def split_mutuelle(num):
#    num = num.replace('\\', '/')
#    nums = re.split(r'[/ ]', num)
#    if len(nums) < 2 and len(num) == 13:
#        s1 = num[:4]
#        s2 = num[4:6]
#        s3 = num[6:11]
#        s4 = num[11:]
#        nums = [s1, s2, s3, s4]
#    if len(nums) > 4 and len(nums[0]) + len(nums[1]) == 4:
#            nums[0] = nums[0] + nums.pop(1) # a slash mistakenly split first four digit section
#    if len(nums[0]) == 3:
#            nums[0] = '0' + nums[0] # if no leading zero
#    return '-'.join(nums)
               
#[split_mutuelle(num) for num in hhsurvey.cd8c]

#def return_mutuelle(childid, df = hhsurvey):
#    subset = df[df['childid'] == childid]
#    nums = [num for num in subset['cd8c']]
#    nums = [re.sub('[a-zA-Z]', '', num) for num in nums if num]
#    nums = [split_mutuelle(num) for num in nums]
#    return(set(list(nums)))
# =============================================================================

# splitHouseholdMutuelle() and getLens() from mutuelle_parser.py

## Sort favoring pattern-fitting then parsed mutuelle numbers
def MutuelleListSort(num):
    if getLens(num.split('-')) == '4252':
        return 1
    if '-' in num:
        return 2
    else:
        return 3

def return_mutuelle(childid, df = hhsurvey):
    subset = df[df['childid'] == childid]
    nums = [num for num in subset['cd8c'] if num]
    nums = list(set([splitHouseholdMutuelle(num) for num in nums]))
    nums.sort(key = MutuelleListSort)
    return nums

mutuelle = {key: return_mutuelle(key) for key in child_keys}
mutuelle_df = pd.DataFrame.from_dict(mutuelle, orient='index')
mutuelle_df.columns = ['cd8c1', 'cd8c2', 'cd8c3', 'cd8c4']

## Variants of mutuelle numbers sorted so first meets pattern criteria
## Split first mutuelle number into its parts: district, health center, household, individual
mutuelle_locations = pd.DataFrame(mutuelle_df['cd8c1'].fillna('nan').str.split('-').values.tolist())
mutuelle_locations.columns = ['m_dist', 'm_hc', 'm_hh', 'm_indv']

def cleanMDist(num):
    
    if '/' in num:
        return None
    if num == 'nan':
        return None
    if len(num) > 5:
        return None
    return num

m_dist_clean = mutuelle_locations.m_dist.apply(lambda x: cleanMDist(x))
mutuelle_locations = mutuelle_locations.assign(m_dist = m_dist_clean)

mutuelle_locations.index = mutuelle_df.index
mutuelle_df = mutuelle_df.join(mutuelle_locations)

## Date of birth
def add_pad(date):
    if date:
        splits = date.split('/')
        if len(splits) == 3:
            splits[0] = '{0:0>2}'.format(splits[0])
            splits[1] = '{0:0>2}'.format(splits[1])
            if len(splits[2]) != 4:
                year = splits[2]
                splits[2] = '20' + year[-2:]
            return '/'.join(splits)

        
def dob_range(childid, df = hhsurvey):
    subset = df[df['childid'] == childid]
    dates = [date for date in subset['cd5c'] if date]
    dates = [re.sub(' ','', date) for date in dates if not re.search('99', date)]
    dates = [add_pad(date) for date in dates]
    dates = [pd.to_datetime(date, format='%d/%m/%Y', errors='coerce') for date in dates if date]
    if dates:
        mindate = min(dates)
        maxdate = max(dates)
        return [mindate, maxdate]
    else:
        return [None]

dobs = {key: dob_range(key) for key in child_keys}
dobs_df = pd.DataFrame.from_dict(dobs, orient='index')
dobs_df.columns = ['min_dob', 'max_dob']

## Child gender
def return_gender(childid, df = hhsurvey):
    subset = df[df['childid'] == childid]
    gend = [gend for gend in subset['genderc'] if gend]
    return list(set(gend))
    
gend = {key: return_gender(key) for key in child_keys}
gend_df = pd.DataFrame.from_dict(gend, orient='index')
gend_df.columns = ['genderc']

## Village id
def return_village(childid, df = hhsurvey):
    subset = df[df['childid'] == childid]
    vil = [vil for vil in subset['a5'] if vil]
    return list(set(vil))
    
vil = {key: return_village(key) for key in child_keys}
vil_df = pd.DataFrame.from_dict(vil, orient='index')
vil_df.columns = ['vill ID']

## Location data
loc = hhsurvey[['childid', 'district', 'sector', 'cell', 'village']]
loc = loc.drop_duplicates('childid')

## Merge all togeher
dfs = [parent_kinya_df, parent_other_df, parent_nick_df,
       child_kinya_df, child_other_df, child_nick_df,
       mutuelle_df, dobs_df, gend_df, vil_df]


child_key = reduce(lambda l,r: pd.merge(l,r, how='inner', left_index = True, right_index = True), dfs)
child_key.index.name = 'childid'
child_key.reset_index(inplace=True)
child_key = child_key.merge(loc, on = 'childid')

## Soundex Names (on first variant only)
child_key = child_key.assign(pc11_sx = child_key.pc11.apply(lambda x: kSound(x)))
child_key = child_key.assign(cd2c1_sx = child_key.cd2c1.apply(lambda x: kSound(x)))

## Recode gender
child_key = child_key.assign(cgender_chr = child_key.genderc.apply(lambda x: 'M' if x == 2 else 'F'))

## Convert all text to Uppercase
child_key = child_key.applymap(lambda s: s.upper() if type(s) == str else s)

## Change dob NA to blank
child_key.min_dob = child_key.min_dob.fillna('')
child_key.max_dob = child_key.max_dob.fillna('')

## Sector centroids
coorDict = {sector.upper(): (centroids.iloc[i,7], centroids.iloc[i,8]) 
                        for i, sector in enumerate(centroids.Name)}

coorDict.update({'': (np.NaN, np.NaN)})

coords = [coorDict[sector] if sector in coorDict.keys() else (np.NaN, np.NaN) for sector in child_key.sector]

coords = pd.DataFrame(coords)
coords.columns = ['sectLat', 'sectLong']

child_key = pd.concat([child_key, coords], axis=1)

## Rename columns
child_key.rename(columns={'childid': 'childid',
                          'pc11': 'pkinyaname1',
                          'pc11_sx': 'pkinyaname1_sx',
                          'pc12' : 'pkinyaname2',
                          'pc13' : 'pkinyaname3',
                          'pc21': 'pothername1',
                          'pc22' : 'pothername2',
                          'pc23' : 'pothername3',
                          'pc24' : 'pothername4',
                          'pc2n' : 'pnickname',
                          'cd2c1' : 'ckinyaname1',
                          'cd2c1_sx': 'ckinyaname1_sx',
                          'cd2c2' : 'ckinyaname2', 
                          'cd2c3' : 'ckinyaname3',
                          'cd2ac1' : 'cothername1',
                          'cd2ac2' : 'cothername2', 
                          'cd2ac3' : 'cothername3',
                          'cd2bc1' : 'cnickname1', 
                          'cd2bc2' : 'cnickname2', 
                          'cd2bc3' : 'cnickname3',
                          'cd8c1' : 'mutnum1',
                          'cd8c2' : 'mutnum2', 
                          'cd8c3' : 'mutnum3',
                          'cd8c4' : 'mutnum4',
                          'm_dist': 'm_dist', 'm_hc': 'm_hc', 
                          'm_hh': 'm_hh', 'm_indv': 'm_indv',
                          'min_dob': 'min_dob', 'max_dob': 'max_dob',
                          'genderc': 'cgender', 'cgender_chr' : 'cgender_chr',
                          'vill ID': 'vill ID',
                          'district': 'district', 'sector': 'sector',
                          'cell': 'cell', 'village': 'village'}, inplace=True)
    
## Codebook
codebook = {'childid': 'ID for child assigned by researchers',
            'pkinyaname1': '1st Variant of Parent Kinya Name',
            'pkinyaname1_sx': 'Soundex of 1st variant of Parent Kinya name',
            'pothername1': '1st Variant of Parent Second Name',
            'ckinyaname1': '1st Variant of Child Kinya Name',
            'ckinyaname1_sx': 'Soundex of 1st variant of Child Kinya name',
            'cothername1': '1st Variant of Child Second Name',
            'mutnum1': '1st Variant of Mutuelle Number',
            'm_dist': 'Mutuelle Number District',
            'm_hc': 'Mutuelle Number Health Center',
            'm_hh': 'Mutuelle Number Household Number',
            'm_indv': 'Mutuelle Number Household Member Number',
            'min_dob': 'Upper range of DOB',
            'cgender': 'Gender',
            'vill ID': 'Research Study Village ID',
            'district': 'District', 'sector': 'Sector',
            'cell': 'Cell', 'village': 'Village',
            'sectX': 'Sector centroid x coordinate',
            'sectY': 'Sector centroid y coordinate'}


## Write to drive
#child_key.to_csv(path + 'Cleaned Data/child_key0611.csv', index=False)
#child_key.to_csv(path + 'Cleaned Data/child_key0611.txt', index=False, sep = '\t')
#child_key.to_csv(desk + 'child_key0611.csv', index=False, na_rep = '<empty>')
#child_key.to_csv(desk + 'child_key0611.txt', index=False, sep = '\t', na_rep = '<empty>')
#child_key.to_csv(desk + 'child_key0626.txt', index=False, sep = '\t')
