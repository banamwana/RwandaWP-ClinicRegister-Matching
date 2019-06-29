# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:55:29 2019

@author: gprenti
"""
import pandas as pd
import os
import numpy as np
import re
from difflib import SequenceMatcher as SM
import unicodedata
from math import sqrt

# =============================================================================
# Import DATA
# =============================================================================

if os.name == 'posix':
    path = '/Users/graemepm/Box Sync/EGHI Record Linkage/Data/'
    desk = '/Users/graemepm/Desktop/'
    path_code = '/Users/graemepm/Box Sync/EGHI Record Linkage/Code/'
    
else:
    path = 'C:/Users/gprenti/Box Sync/EGHI Record Linkage/Data/'
    desk = 'C:/Users/gprenti/Desktop/'
    path_code = 'C:/Users/gprenti/Box Sync/EGHI Record Linkage/Code/'

def readStata(file):
    
    return pd.read_stata(file, convert_dates = True, 
                         convert_categoricals = True, encoding = 'latin-1')

file1 = path + 'Clinic Data/Rwanda_maternity_STATA_2019-06-05.dta'
file2 = path + 'Clinic Data/RwandaCHW_STATA_2019-06-05.dta'
file3 = path + 'Clinic Data/RwandaHealthFa_STATA_2019-06-05.dta'
file4 = path + 'Village Lists/sector_centroids.csv'

mat = readStata(file1)
chw = readStata(file2)
hf = readStata(file3)
centroids = pd.read_csv(file4)

## Source Mutuelle Parser and Soundex
exec(open(path_code + 'mutuelle parser.py').read())
exec(open(path_code + 'soundex.py').read())
exec(open(path_code + 'norvig name checker.py').read())
exec(open(path_code + 'clean villages.py').read())
exec(open(path_code + 'dbl msp pattern.py').read())

## Examine Columns
hf.columns
chw.columns
mat.columns

## Check for intersection of record id's
#len([id for id in chw.recordid if id in hf.recordid])
#sum(chw.recordid in mat.recordid)
#sum(mat.recordid in hf.recordid)
#
#sum(hf.recordid in chw.recordid)
#sum(mat.recordid in chw.recordid)
#sum(hf.recordid in mat.recordid)

## Select columns in same order
hf_columns = ['recordid', 'patientvisitdate', 'patientname', 'parentname',
              'district', 'cell', 'village', 'agemois', 'sexe', 'mutuelle_number']

hf_df = hf[hf_columns]

chw = chw.assign(cell = np.NaN)

chw_columns = ['recordid', 'chw_pdate', 'chw_pname', 'chw_paren', 'chw_dist',
               'cell', 'chw_vill', 'chw_agemonthstotal', 'chw_sexe', 'chw_mutnum']

chw_df = chw[chw_columns]

columns = ['recordid', 'patientvisitdate', 'patientname', 'parentname', 
             'district', 'cell', 'village', 'age_mo', 
             'sexe', 'mutuelle_number']

hf_df.columns = columns
chw_df.columns = columns

## Unite sector columns
#hf_sect = hf.filter(regex = 'sect').apply(lambda x: ''.join(x.dropna()), axis = 1)
#chw_sect = chw.filter(regex = 'sect').apply(lambda x: ''.join(x.dropna()), axis = 1)

hf_sect = hf.filter(regex = 'sect').groupby(lambda x: 'sect', axis=1).first() # takes first
chw_sect = chw.filter(regex = 'sect').groupby(lambda x: 'sect', axis=1).first()

def removeNA(sect):
    
    if str(sect).upper() in ['NA', 'MISSING', 'NOT CLEAR', '']:
        
        return np.NaN
    
    else:
    
        return sect
    
hf_sect = pd.Series(removeNA(sect) for sect in hf_sect.sect)
chw_sect = pd.Series(removeNA(sect) for sect in chw_sect.sect)

hf_df = hf_df.assign(sector = hf_sect)
chw_df = chw_df.assign(sector = chw_sect)

# =============================================================================
# Prelimary Cleaning
# =============================================================================
def recodeSexe(sexe):
    
    if sexe == 'Gore / F':
        return 'F'
    if sexe == 'Gabo / M':
        return 'M'
    else:
        return np.NaN

chw_df = chw_df.assign(sexe = chw_df.sexe.apply(recodeSexe))

def recodeMissingSexe(sexe):
    
    if str(sexe).upper() in ['MISSING', 'NOT CLEAR', 'NA', '']:
        return np.NaN
    else:
        return sexe

hf_df = hf_df.assign(sexe = hf_df.sexe.apply(recodeMissingSexe))

# Find all the non-numbers
hf_df.age_mo.value_counts()
[age for age in hf_df.age_mo if re.search(r'[^\d.]', str(age))]

def recodeWeird(age):
    
    if age == '1/2 mois (15days only)':
        return '0.5'
    else:
        return age

hf_df = hf_df.assign(age_mo = hf_df.age_mo.apply(recodeWeird))
hf_df = hf_df.assign(age_mo = pd.to_numeric(hf_df.age_mo, errors='coerce'))

## Combine health facility and chw data
data = pd.concat([hf_df, chw_df], axis=0).reset_index()

# =============================================================================
# Clean location names
# =============================================================================
    
### Implement location cleaning
villRaw = data[['district', 'sector', 'cell', 'village']].apply(lambda s: s.astype(str).str.upper())

villCleaned = cleanVillages(villRaw)

data = pd.concat([data, villCleaned], axis=1)

#sum([village == '' for village in data.village_clean])
#2109 missing

# =============================================================================
# Parse Mutuelle
# =============================================================================

mutParse = data.mutuelle_number.apply(splitClinicMutuelle)
mutParse = [(r[0], r[1], r[2], r[3]) if r is not np.NaN else ('','','','') for r in mutParse]
mutParse = pd.DataFrame(mutParse)
mutParse.columns = ['mdist', 'mhf', 'mhh', 'mindv']

data = pd.concat([data, mutParse], axis=1)

#sum([num == '' for num in data.mdist])
#8078 missing

# =============================================================================
# Kinyarwanda Soundex and Kinyaname spell checker
# =============================================================================

pnames = data.parentname.str.upper().replace('NA', '')
cnames = data.patientname.str.upper().replace('NA', '')

## Replace abbreviations
# J. = Jean, M. = Marie, N. = Nyira
pnames = [name.replace('N.','NYIRA').replace('J.', 'JEAN'). replace('M.','MARIE') for name in pnames]
cnames = [name.replace('N.','NYIRA').replace('J.', 'JEAN'). replace('M.','MARIE') for name in cnames]

pkinyaname = [name.split(' ')[0] for name in pnames]
pothername = [' '.join(name.split(' ')[1:]) if len(name.split(' ')) > 1 else '' for name in pnames]

ckinyaname = [name.split(' ')[0] for name in cnames]
cothername = [' '.join(name.split(' ')[1:]) if len(name.split(' ')) > 1 else '' for name in cnames]

pkinyaname_sx = [kSound(name) for name in pkinyaname]
ckinyaname_sx = [kSound(name) for name in ckinyaname]

## Takes some time because of string distance comparison over many combinations
pkinyaname_cor = [kNameCorrect(name) if re.search(dmsPattern, name) else np.NaN for name in pkinyaname]
ckinyaname_cor = [kNameCorrect(name) if re.search(dmsPattern, name) else np.NaN for name in ckinyaname]

## Clean up weird accented characters
def stripAccents(name):
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
    
pothername = [stripAccents(name) for name in pothername]
cothername = [stripAccents(name) for name in cothername]

## Combine all name
names = {'ckinyaname': ckinyaname, 'ckinyaname_sx': ckinyaname_sx, 
         'ckinyaname_cor': ckinyaname_cor, 'cothername': cothername,
         'pkinyaname': pkinyaname, 'pkinyaname_sx': pkinyaname_sx, 
         'pkinyaname_cor': pkinyaname_cor, 'pothername': pothername}

names = pd.DataFrame(names)

data = pd.concat([data, names], axis=1)

# =============================================================================
# Estimate DOB
# =============================================================================

months = pd.Series([pd.Timedelta(month, 'M') for month in data['age_mo']])
data = data.assign(dob_est = data['patientvisitdate'] - months)

# =============================================================================
# Centroids
# =============================================================================

coorDict = {sector.upper(): (centroids.iloc[i,7], centroids.iloc[i,8]) 
                        for i, sector in enumerate(centroids.Name)}

coorDict.update({'': (np.NaN, np.NaN)})

coords = [coorDict[sector] for sector in data.sector_clean]

coords = pd.DataFrame(coords)
coords.columns = ['sectLat', 'sectLong']

data = pd.concat([data, coords], axis=1)

# =============================================================================
# Finish up and export
# =============================================================================

data = data.replace('', np.NaN)

#data.to_csv(desk + 'clinic0626.txt', index=False, sep = '\t')

# =============================================================================
# Codebook
# =============================================================================

codebook = {'recordid': 'Unique id of digitized hf/chw record', 
            'patientvisitdate': 'Original visit date', 
            'patientname': 'Original patient names', 
            'parentname': 'Original parent names', 
            'district': 'Original distric', 'cell': 'Original cell',
            'village': 'Original village', 
            'age_mo': 'Original age in months', 'sexe': 'Child sex', 
            'mutuelle_number': 'Original mutuelle number',
            'sector': 'First sector from all sector columns',
            'district_clean': 'Imputed disctrict', 'sector_clean': 'Imputed sector',
            'cell_clean': 'Imputed cell', 'village_clean': 'Imputed village',
            'mdist': 'District section of mutuelle number', 
            'mhf': 'Health facility section of mutuelle number',
            'mhh': 'Household section of mutuelle number',
            'mindv': 'Individual section of mutuelle number',
            'ckinyaname': 'Child kinyarwanda name', 
            'ckinyaname_sx': 'Soundex of child kinyarwanda name',
            'ckinyaname_cor': 'Spell corrected child kinyarwanda name',
            'cothername': 'Child second name', 
            'pkinyaname': 'Parent kinyarwanda name', 
            'pkinyaname_sx': 'Soundex of parent kinyarwanda name',
            'pkinyaname_cor': 'Spell corrected parent kinyarwanda name',
            'pothername': 'Parent second name', 
            'dob_est': 'DOB estimated from visit data and child age',
            'sectX': 'Sector centroid x coordinate',
            'sectY': 'Sector centroid y coordinate'}

