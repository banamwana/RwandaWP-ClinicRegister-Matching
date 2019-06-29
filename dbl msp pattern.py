# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 12:51:56 2019

@author: gprenti
"""
import itertools as it
import re
import json

### Create keyword alphabet
con = ['b','c','d','f','g','h','j','k','l','m','n',
       'p','q','r','s','t','v','w','x','y','z']

vowel = ['a','e','i','o','u']

con2 = ['b','c','d','f','g','h','j','k','l','m','n',
        'p','q','r','s','t','v','x','y','z'] # without w

con3 = ['b','c','d','f','g','h','j','k','l','m',
        'p','q','r','s','t','v','w','x','y','z'] # without n

vr = ['e', 'o'] # round vowels
vfr = ['i','e'] # front vowels
vbk = ['o','u'] # affricates
a = ['c', 'h', 'k', 'p', 's', 't'] # voiceless consonants

doubles = []
for x in it.product(con, con):
    doubles.append(x[0] + x[1])
    
for x in it.product(vowel, vowel):
    doubles.append(x[0] + x[1])
    
### Create dictionary of two letter consonant exceptions
vowel_after = ['bf', 'by','bw',
               'cy','cw',
               'dw',
               'gw',
               'kw',
               'hw',
               'jy',
               'mf','mw','my',
               'nf','nw',
               'pf',            
               'rw','ry',
               'sw','sy',
               'tw','ty',
               'zw']

vw_after = ['nc','nd','nk','ng','nj','nz','nt','ny','mb','mp','mv','ts']

excpt_dict = {key: key + '(?![aeiou]){1}' for key in vowel_after}

vw_after = {key: key + '(?![aeiouw]){1}' for key in vw_after}

### Dictionary of 2+ letter exceptions
special = {
        
        # 'hy' only when preceded by 's' AND followed by vowel or 'wa'
        'hy':'(?<!s)hy|hy(?![aeiou]{1}|wa)',
        
        # 'dy' only when preceded by 'n'
        'dy':'(?<!n){1}dy',
        
        # 'vw' only when preceded by 'm'
        'vw':'(?<!m){1}vw',
        
        # 'py' only when followed by 'i'
        'py':'py(?!i){1}',
        
        # 'yw' only when preceded by 'sh' or 'n' and followed by vowel
        'yw':'(?<!sh|.n)yw|yw(?![aeiou]{1})',
        
        # 'jw' only when followed by 'a', 'e', or 'i'
        'jw':'jw(?![aei]){1}',
        
        # 'nn' only when followed by 'ye'
        'nn':'nn(?!ye)',
        
        # 'sh' only when followed by vowel or 'y' or 'w'
        # can also be preced by 'n'
        'sh':'sh(?![aeiouyw]){1}',
        
        # 'ns' only when followed by vowel + 'w' OR 'h' and vowel + 'w'
        'ns':'ns(?![aeiouyw]{1}|h[aeiouyw]{1})'
        

              }

excpt_dict.update(special)
excpt_dict.update(vw_after)



# =============================================================================
#path = '/Users/graemepm/Box Sync/Graeme/Name Matching/Data/names_raw.txt'
#path = 'C:/Users/gprenti/Box Sync/Graeme/Name Matching/Data/names_raw.txt'
#
#with open(path, 'r') as f:
#        names = json.loads(f.read())
#    
#def look(string):
#    string = re.compile(string, re.I)
#    return [x for x in names if string.search(str(x))]
#
#pat = ''
#look(pat)
# =============================================================================

### Function to add exceptions
def recode_exc(let_list, codeDict):
    for key, value in codeDict.items():
        for pat in let_list:
            if pat == key:
                let_list.remove(pat)
                let_list.append(value)

### Execute function
recode_exc(doubles, excpt_dict)
        
### Join patterns into string
dbl_combined = '|'.join(x for x in doubles)

### Compile into regex object
dmsPattern = re.compile(dbl_combined, re.I)

del(dbl_combined, recode_exc, excpt_dict, doubles)
del(con, vowel, con2, con3, vr, vfr, vbk, a)
