# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 10:31:28 2019

@author: gprenti
"""
# =============================================================================
# Libraries
# =============================================================================
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
from copy import deepcopy
from jellyfish import damerau_levenshtein_distance as damLev
import statistics
import datetime
import abydos.phonetic as ap
# =============================================================================
# Directories
# =============================================================================
if os.name == 'posix':
    home = '/Users/graemepm/'
else:
    home = 'C:/Users/gprenti/'
    
path = home + 'Box Sync/EGHI Record Linkage/Data/'
mclean_path = home + 'Box Sync/EGHI Record Linkage/FRIL Matching Iterations/Manual Cleaning/'
desk = home + 'Desktop/'
# =============================================================================
# Source scripts
# =============================================================================
exec(open(home + 'Box Sync/EGHI Record Linkage/Code/dbl msp pattern.py').read())
exec(open(home + 'Box Sync/EGHI Record Linkage/Code/mutuelle parser.py').read())
exec(open(home + 'Box Sync/EGHI Record Linkage/Code/soundex.py').read())

# =============================================================================
# Import Data
# =============================================================================
with open(home + 'Box Sync/Graeme/Name Matching/Data/names.txt', 'r') as f:
    name_list = json.loads(f.read())
with open(home + 'Box Sync/Graeme/Name Matching/Data/names_sec.txt', 'r') as f:
    NAMES_SEC = json.loads(f.read())
with open(mclean_path + 'manual_cleaning_mutuelle.txt', 'r') as f:
    MutCleanDict = json.loads(f.read())
with open(mclean_path + 'manual_cleaning_names.txt', 'r') as f:
    NameCleanDict = json.loads(f.read())

NAMES = Counter(name_list) # frequency dictionary of cleaned UBD names

files = os.listdir(path)

villages = pd.read_csv(path + 'Village Lists/p2 - all study villages.csv')
villages = villages.drop_duplicates()

# Read in Stata and save as txt
#dateDict = {'cd5c': '???'}
#hhsurvey = pd.read_stata(path + 'HH Survey Data/hhsurvey_child_long.dta')
#hhsurvey.to_csv(path + 'HH Survey Data/hhsurvey_child_long.txt', sep='\t', index=False)

# Coltypes
coltypes =  {'a5': 'int64', 'a7': 'int64', 'childid': 'int64',
             'round': 'float64', 'pc1': 'str', 'pc2': 'str',
             'pc2n': 'str', 'cd2c': 'str', 'cd2ac': 'str',
             'cd2bc': 'str', 'cd5c': 'str', 'cd8c': 'str',
             'dobc': 'float64', 'genderc': 'int64', 'z3': 'str',
             'z4': 'str', 'z5': 'str', 'z6': 'str', 'vcode': 'int64',
             'v2': 'str', 'v3': 'str', 'v4': 'str', 'cell_id': 'int64',
             'sector_id': 'int64', 'distr_id': 'int64', 'prov_id': 'int64',
             'village': 'str', 'cell': 'str', 'sector': 'str', 
             'district': 'str', 'province': 'str'}

# Read in data
hhsurvey = pd.read_csv(path + 'HH Survey Data/hhsurvey_child_long.txt', sep='\t', dtype=coltypes)
centFile = path + 'Village Lists/sector_centroids.csv'
centroids = pd.read_csv(centFile)

# =============================================================================
# HH Survey Variables
# =============================================================================
# a5 - village ID (note villages in health facility data haven't been matched to an ID yet.  
# a7  household ID
# pc1 - parent name Rwandan
# pc2 - parent name  French/English
# pc2n - parent nickname
# childid  - child ID - first 6 digits are household ID
# genderc - child gender
# dobc - child age
# cd5c - child dob
# cd2c  - child name Rwandan 
# cd2ac - child name French/English
# cd2bc - child nickname
# cd8bc - mutuelle number
# =============================================================================

child_keys = list(set(hhsurvey.childid))
#pat = re.compile('[a-z]', re.I)
#[key for key in child_keys if pat.search(key)]
#[key for key in child_keys if key == '88']

# =============================================================================
# Names
# =============================================================================
#def str_score(x, y):
#    #s = SeqM(isjunk=None, a= x, b = y, autojunk = False).ratio()
#    s = ds.get_jaro_distance(x, y, winkler=False)
#    return s

def str_score(s1, s2):
    
    return 1 - (damLev(s1, s2) / max(len(s1), len(s2)))

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
    
def clean_name(name):
    
    name = str(name).lower()
    name = strip_accents(name)
    
    replaceDict = {'n/a': '', '//o': '', '/o/o': '', '//0': '', '/0/0': ''}
    
    reSubDict = {'0':'o', r'\\': '', '(?<!n) (?!.)': '', 'j\.': 'jean ', 'm\.': 'marie '}
    
    for char, replacement in replaceDict.items():
        if name == char:
            name = replacement
            
    for pattern, replacement in reSubDict.items():
        name = re.sub(pattern, replacement, name)
      
    # remove all numbers
    name = re.sub('\d+', '', name)
    
    return name
    
def prob_kinya(name, N=sum(NAMES.values())): 
    "Probability of `name`."
    return NAMES[name] / N

def prob_sec(name, N=sum(NAMES_SEC.values())): 
    "Probability of `name`."
    return NAMES[name] / N

def return_names(childid, name_cols, df=hhsurvey, cut=0.8, sortkey=prob_kinya):
    
    subset = df[df['childid']==childid]
    subset = subset[name_cols]
    
    if type(name_cols)==list and len(name_cols) > 1:
        subset = subset.applymap(clean_name)
        names  = subset.apply(lambda x: ' '.join(x), axis=1)
        names = [name for name in set(names) if re.search('[a-z]', name)]
        
    else:
        subset = subset.apply(clean_name)
        names = [name for name in set(subset) if re.search('[a-z]', name)]
    
    names.sort(key=sortkey, reverse=True)
    naChars = ['nan', ' n/a', 'n/a', 'NAN', 'N/A', ' N/A']
    names = [name for name in names if name not in naChars]
    
    nameset = deepcopy(names)
    
    chosen = []
    while nameset:
        candidate = nameset.pop(0)
        
        # compare candidate to all names already chosen
        scores = []
        for name in chosen:
            score = str_score(candidate, name)
            scores.append(score)
        scores = pd.Series(scores)
        
        # true if scores empty (first candidate)
        if (scores < cut).all(): 
            chosen.append(candidate)
    
    return chosen

def re_split(df):
    
    childid = df.index
    new_df = pd.DataFrame(index=range(len(df)))
    df = df.reset_index().drop('index', axis=1)
    
    for col in df:
        split_names = df[col].str.split(' ', 1, expand=True)
        split_names.columns = ['pkinyaname' + str(col+1), 'pothername' + str(col+1)]
        new_df = pd.concat([new_df, split_names], axis=1)
        
    new_df.index = childid
        
    return new_df

## Parent Both
parent = {key: return_names(key, ['pc1', 'pc2'], cut=0.9) for key in child_keys}
parent_df = pd.DataFrame.from_dict(parent, orient='index')
parent_names_df = re_split(parent_df)

### Parent Kinyarwanda Name
#parent_kinya = {key: return_names(key, 'pc1') for key in child_keys}
#parent_kinya_df = pd.DataFrame.from_dict(parent_kinya, orient='index')
#parent_kinya_df.columns = ['pc11', 'pc12', 'pc13']

### Parent Other Name
#parent_other = {key: return_names(key, 'pc2', sortkey = prob_sec) for key in child_keys}
#parent_other_df = pd.DataFrame.from_dict(parent_other, orient='index')
#parent_other_df.columns = ['pc21', 'pc22', 'pc23', 'pc24']

### Parent NickName
#parent_nick = {key: return_names(key, 'pc2n') for key in child_keys}
#parent_nick_df = pd.DataFrame.from_dict(parent_nick, orient='index')
#parent_nick_df.columns = ['pc2n']

## Child Name Rwandan
child_kinya = {key: return_names(key, 'cd2c') for key in child_keys}
child_kinya_df = pd.DataFrame.from_dict(child_kinya, orient='index')
child_kinya_df.columns = ['ckinyaname1', 'ckinyaname2', 'ckinyaname3']

## Child Name Other
child_other = {key: return_names(key, 'cd2ac', sortkey = prob_sec) for key in child_keys}
child_other_df = pd.DataFrame.from_dict(child_other, orient='index')
child_other_df.columns = ['cothername1', 'cothername2', 'cothername3']

## Child Nick
#child_nick = {key: return_names(key, 'cd2bc') for key in child_keys}
#child_nick_df = pd.DataFrame.from_dict(child_nick, orient='index')
#child_nick_df.columns = ['cd2bc1', 'cd2bc2', 'cd2bc3']

# =============================================================================
# Mutuelle: Sort favoring pattern-fitting then parsed mutuelle numbers
# =============================================================================
def MutuelleListSort(num):
    if getLens(num.split('-')) == '4252':
        return 1
    if '-' in num:
        return 2
    else:
        return 3

def return_mutuelle(childid, df = hhsurvey):
    subset = df[df['childid'] == childid]
    nums = [str(num) for num in subset['cd8c'] if num]
    nums = list(set([splitHouseholdMutuelle(num) for num in nums]))
    nums.sort(key = MutuelleListSort)
    return nums

mutuelle = {key: return_mutuelle(key) for key in child_keys}
mutuelle_df = pd.DataFrame.from_dict(mutuelle, orient='index')
mutuelle_df.columns = ['mutnum1', 'mutnum2', 'mutnum3', 'mutnum4']
mutuelle_df = mutuelle_df.replace('nan', None)

## Variants of mutuelle numbers sorted so first meets pattern criteria
## Split first mutuelle number into its parts: district, health center, household, individual
mutuelle_locations = pd.DataFrame(mutuelle_df['mutnum1'].fillna('').str.split('-').values.tolist())
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

# =============================================================================
# Date of birth
# =============================================================================
def strip_date(datestring):
    date = pd.NaT
    try:
        date = datetime.datetime.strptime(datestring, '%d/%m/%Y')
    except:
        try:
            date = datetime.datetime.strptime(datestring, '%Y')
        except:
            pass
    return date 

def convert_date(datestring):
    
    date = pd.NaT
    datestring = str(datestring).strip().lstrip('/').rstrip('/').\
        replace('//','/').replace('"','').replace('o', '0').replace('O','0')
        
    datestring = re.sub('[a-zA-Z]+', '', datestring)
    
    oneOffDict = {'3/32011': '3/3/2011', '12/11': '1/12/2011',
                  '0503/015': '5/3/2015', '06/052014': '6/5/2014'}
    
    for key, value in oneOffDict.items():
        if datestring==key:
            datestring = value
    
    datestrings = datestring.split('/')
    datestrings = [date.strip() for date in datestrings]
    
    # Ensure proper format if only year input
    if len(datestrings) == 1:
        datestrings[0] = '20' + datestrings[0][-2:]
        if datestrings[0][-2:]=='99':
            return pd.NaT
        date = strip_date(datestrings[0])
    
    if len(datestrings) == 3:
        
        # If we don't know the year, then there's no point
        if datestrings[2]=='9999':
            return pd.NaT
        
        # Ensure proper format for year
        datestrings[2] = '20' + datestrings[2][-2:]
        
        # Ensure proper format for month, day
        # no index error if only 1 charater
        datestrings[1] = datestrings[1][-2:] 
        datestrings[0] = datestrings[0][-2:]
        
        # If month, date unknown, input as Jan or 1st
        if datestrings[0]=='99':
            datestrings[0] = '1'
        if datestrings[1]=='99':
            datestrings[1] = '1'
            
        new_datestring = '/'.join(datestrings)
        date = strip_date(new_datestring)
        
    if pd.isna(date) and len(datestrings) == 3:
         
        # Clip month/date if date hasn't converted because of invalid values
        if int(datestrings[0]) > 28:
            datestrings[0] = '28'
        
        if datestrings[0]=='0':
            datestrings[0] = '1'
            
        if int(datestrings[1]) > 12:
            datestrings[1] = '12'
            
        if datestrings[1]=='0':
            datestrings[1] = '1'
            
        new_datestring = '/'.join(datestrings)
        date = strip_date(new_datestring)
        
    if date > datetime.datetime(2017, 1, 1, 0, 0, 0, 0):

        return pd.NaT
            
    return date

# Check performance
#nas = [date for date in hhsurvey.cd5c if pd.isna(convert_date(date))]
#nas = [date for date in nas if str(date) not in ['nan', '99/99/9999']]
#pd.Series(hhsurvey.cd5c.apply(convert_date)).isna().sum()

def return_dob(childid, df = hhsurvey):
    subset = df[df['childid'] == childid]
    dates = [convert_date(date) for date in subset['cd5c']]
    dates = pd.Series(dates).dropna()
    
    if dates.isna().all():
        return pd.NaT, pd.NaT, pd.NaT
    else:
        min_dob = dates.min()
        med_dob = dates.quantile(0.5)
        max_dob = dates.max()
        return min_dob, med_dob, max_dob
    
dobs = {key: return_dob(key) for key in child_keys}
dobs_df = pd.DataFrame.from_dict(dobs, orient='index')
dobs_df.columns = ['min_dob', 'med_dob', 'max_dob']

diff = dobs_df.max_dob - dobs_df.min_dob
diff.max()

dobs_df.drop(['min_dob', 'max_dob'], axis=1, inplace=True)
dobs_df.columns = ['dob']

# =============================================================================
# Child gender
# =============================================================================
def return_gender(childid, df=hhsurvey):
    subset = df[df['childid'] == childid]
    gend = [gend for gend in subset['genderc'] if gend]
    return list(set(gend))
    
gend = {key: return_gender(key) for key in child_keys}
gend_df = pd.DataFrame.from_dict(gend, orient='index')
gend_df.columns = ['genderc']

# =============================================================================
# Village id
# =============================================================================
def return_village(childid, df=hhsurvey):
    subset = df[df['childid'] == childid]
    vil = [vil for vil in subset['a5'] if vil]
    return list(set(vil))
    
vil = {key: return_village(key) for key in child_keys}
vil_df = pd.DataFrame.from_dict(vil, orient='index')
vil_df.columns = ['vill ID']

# =============================================================================
# Administrative location data
# =============================================================================
loc_df = hhsurvey[['childid', 'district', 'sector', 'cell', 'village']]
loc_df = loc_df.drop_duplicates('childid')
loc_df.set_index('childid', inplace=True)

# =============================================================================
# Geo-location data
# =============================================================================
hhsurvey[['z3','z4']] = hhsurvey[['z3','z4']].replace('no-gps', np.NaN).astype(float)

def return_geo(childid, df=hhsurvey):
    subset = df[df['childid'] == childid]
    geoX = [geo for geo in subset['z3'] if ~np.isnan(geo)]
    geoY = [geo for geo in subset['z4'] if ~np.isnan(geo)]
    
    xLat = None
    yLat = None
    
    if geoX:
        xLat = np.median(geoX)
        
    if geoY:
        yLat = np.median(geoY)
    
    return (xLat, yLat)

geo = {key: return_geo(key) for key in child_keys}
geo_df = pd.DataFrame.from_dict(geo, orient='index')
geo_df.columns = ['lat', 'long']

# =============================================================================
# Merge all togeher
# =============================================================================
dfs = [parent_names_df,
       child_kinya_df, child_other_df,
       mutuelle_df,  gend_df, vil_df, loc_df, geo_df, dobs_df]

child_key = reduce(lambda l,r: pd.merge(l, r, how='inner', 
                                        left_index = True,
                                        right_index = True), dfs)
child_key.index.name = 'childid'
child_key.reset_index(inplace=True)

# =============================================================================
# Manual Cleaning - (Before Soundex)
# =============================================================================

# Manually Clean Names
for cleankey, cleanDict in NameCleanDict.items():
    row = child_key[child_key.childid == int(cleankey)].index[0]
    for colname, replacement in cleanDict.items():
        child_key.at[row, colname] = replacement

# Manually Clean Mutuelle
for cleankey, cleanDict in MutCleanDict.items():
    row = child_key[child_key.childid == int(cleankey)].index[0]
    for colname, replacement in cleanDict.items():
        child_key.at[row, colname] = replacement

# =============================================================================
# Soundex Names
# =============================================================================
de = ap.RefinedSoundex()
def refSX(name):
    
    if name and name is not np.NaN:
        return de.encode(name)
    
    else:
        return np.NaN

kinya_sound = child_key.filter(regex='kinyaname').fillna('').applymap(kSound)
kinya_sound.columns = [col + '_sx' for col in child_key.filter(regex='kinyaname').columns]

other_sound = child_key.filter(regex='othername').fillna('').applymap(refSX)
other_sound.columns = [col + '_sx' for col in child_key.filter(regex='othername').columns]

child_key = pd.concat([child_key, kinya_sound, other_sound], axis=1)

# =============================================================================
# Final Cleaning
# =============================================================================

# Recode gender
child_key = child_key.assign(cgender_chr = child_key.genderc.apply(lambda x: 'M' if x == 2 else 'F'))

# Convert all text to Uppercase, remove straggle NAN
child_key = child_key.applymap(lambda s: s.upper() if type(s) == str else s)
child_key = child_key.replace('NAN','').replace('NA','')

# Change dob NA to blank
child_key.dob = child_key.dob.fillna('')

# Add household ID
hhids = hhsurvey[['childid', 'a7']].drop_duplicates()
hhids.columns = ['childid', 'hhid']
child_key = child_key.merge(hhids, on='childid')

# =============================================================================
# Sector centroids
# =============================================================================
#coorDict = {sector.upper(): (centroids.iloc[i,7], centroids.iloc[i,8]) 
#                        for i, sector in enumerate(centroids.Name)}
#
#coorDict.update({'': (np.NaN, np.NaN)})
#
#coords = [coorDict[sector] if sector in coorDict.keys() else (np.NaN, np.NaN) for sector in child_key.sector]
#
#coords = pd.DataFrame(coords)
#coords.columns = ['sectLat', 'sectLong']
#
#child_key = pd.concat([child_key, coords], axis=1)

# =============================================================================
# Codebook columns
# =============================================================================
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
            'dob': 'Median DOB from range of DOBs',
            'genderc': 'Gender',
            'vill ID': 'Research Study Village ID',
            'district': 'District', 'sector': 'Sector',
            'cell': 'Cell', 'village': 'Village',
            'lat': 'Median X coordinate of gps',
            'long': 'Median Y coordinate of gps'}

# =============================================================================
# Write to drive
# =============================================================================
child_key.to_csv(desk + 'child_keyMCLEAN.txt', index=False, sep = '\t')


# Old Code
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
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