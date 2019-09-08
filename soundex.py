# -*- coding: utf-8 -*-
"""
Kinyarwanda soundex
Adapted from OpenMRS phonetics alogrithm
https://github.com/openmrs/openmrs-module-namephonetics/blob/master/api/src/main/java/org/openmrs/module/namephonetics/phoneticsalgorithm/KinyarwandaSoundex.java
"""

def kSound(name):
    
    validChars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    consonants = set('BCDFGHJKLMNPQRSTVWXYZ')
    vowels = set('AEIOU')
    
    def letterFollowedByStringSetEnhancement(replacement, name, letter, possibleSeconds):
        for stSecond in possibleSeconds:
            searchString = letter + stSecond
            name = name.replace(searchString, replacement)
        return name
    
    def replaceCharSequences(replacement, strings, name):
        for strTmp in strings:
            name = name.replace(strTmp, replacement)
        return name
 
    def removeInvalidChars(name):
        letters = list(name)
        nameValid = [letter for letter in letters if letter in validChars]
        name = ''.join(nameValid)
        return name
    
    def removeMN(name):
        charsToDrop = 'MN'
        letter1 = name[0]
        if len(name) > 2 and letter1 in charsToDrop and name[1] in consonants and name[1] != 'Y':
            return name[1:]
        else:
            return name
        
    def replaceEnhancement(name):
        stSet = set(['CY','SH','SHY'])
        for string in stSet:
            name = name.replace(string, '9')
        return name
        
    def replaceC(name):
        char = 'C'
        charsToChange = [char + vowel for vowel in vowels]
        for char in charsToChange:
            charvowel = char[1]
            name = name.replace(char, '9' + charvowel)
        return name
    
    def removeDoubles(name):
        doubles = {'11':'1', '22':'2', '33':'3', '44':'4', '55': '5',
                   '66':'6', '77':'7', '88':'8', '99':'9'}
    
        for double, single in doubles.items():
            while double in name:
                name = name.replace(double, single)
        return name
    
    if type(name) is not str:
        print("RwandaPhoneticsAlgorithm was passed something that was not a string to encode.")
        return None 
    
    # To upper case
    name = name.upper()
    
    # Drop all punctuation marks and numbers and spaces
    name = removeInvalidChars(name)
    
    # Handle blanks
    if name is None or name == '':
        return np.NaN # Will give fits in R if returns None
    
    # Words starting with M or N followed by another consonant (excluding Y) should drop the first letter
    #!!! What about MPAKA? or similar names where M transforms P or another letter?
    name = removeMN(name)
    
    # SHY and CH as common phonemes enhancement
    # Replace CY, SH, SHY with 9
    #!!! What if it's at the beginning?
    name = replaceEnhancement(name)
    
    # Retain the first letter of the word
    firstLetter = name[0]
    name = name[1:]
    
    # NEW: Replace C before vowel with 9
    name = replaceC(name)
    
    # Initial C/K enhancement
    name = replaceCharSequences('K', 'C',       name)
    name = replaceCharSequences('G', ['JY'],    name)
    name = replaceCharSequences('V', 'F',       name)
    name = replaceCharSequences('R', 'L',       name)
    name = replaceCharSequences('S', 'Z',       name)

    # W followed by a vowel should be treated as a consonant enhancement
    name = letterFollowedByStringSetEnhancement('8', name, 'W', vowels)
    name = letterFollowedByStringSetEnhancement('8', name, 'Y', vowels)
    
    # Change letters from the following sets into the digit given
    name = replaceCharSequences('0', vowels,        name)
    name = replaceCharSequences('1', set('BPV'),    name)
    name = replaceCharSequences('2', set('GKQX'),   name)
    name = replaceCharSequences('3', set('DT'),     name)
    name = replaceCharSequences('4', set('LR'),     name)
    name = replaceCharSequences('5', set('MN'),     name)
    # Originally with CGKQX
    name = replaceCharSequences('6', set('SZ'),     name)
    # Originally with CGKQX
    name = replaceCharSequences('7', set('JHYW'),   name)
        
    # Remove all pairs of digits which occur beside each other from the string
    name = removeDoubles(name)
            
    # Remove all zeros from the string
    name = name.replace('0', '')

    # The original version only used four digits, not clear why
#    # Return only the first four positions
#    if len(name) < 3:
#        return firstLetter + name
#    else:
#        return firstLetter + name # as many as there are
    
    return firstLetter + name
    
