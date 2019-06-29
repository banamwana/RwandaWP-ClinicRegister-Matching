# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:17:29 2019

@author: gprenti
"""
# =============================================================================
# Purpose: Parse Mutuelle number inputs
# =============================================================================

import re
import numpy as np
import pandas as pd

# =============================================================================
# Helper Functions
# =============================================================================

## Function to get lengths of each part in split number
def getLens(nums):
    
    return ''.join([str(len(l)) for l in nums])

## Correct district
def correctDistrict(nums):
    
    validDistricts = ['0301', '0302', '0303', '0304', '0305', '0306', '0307']
    
    dist = nums[0]
    
    if len(dist) == 3 and dist[:2] == '03':
        newdist = nums[0][:2] + '0' + nums[0][2]
        if newdist in validDistricts:
            nums[0] = newdist
        
    return nums

## Functon to join all consecutive items of the split mutuelle number
def joinConsecutive(nums):
    
    return [nums[:i] + [nums[i] + nums[i+1]] + nums[i+2:] for i in range(len(nums)-1)]    

## Function to check whether joining any consecutive parts will fit pattern
def correctConsecutive(nums, pattern = '4252'):
    
    newNums = joinConsecutive(nums)
    newNumsLens = [getLens(num) for num in newNums]
    
    if pattern in newNumsLens:    
        return newNums[newNumsLens.index(pattern)]
    
    return None

## Function to check whether splitting any parts would fit pattern
def correctSplit(nums):
    
    # if first two parts need splitting
    if getLens(nums) == '652':
        
        first = nums[0]
        firstSplit = [first[:4], first[4:]]
        
        return firstSplit + nums[1:]
    
    # if second two parts need splitting
    if getLens(nums) == '472':
        
        mid = nums[1]
        midSplit = [mid[:2], mid[2:]]
        
        return [nums[0]] + midSplit + [nums[2]]
        
    # if last two parts need splitting
    if getLens(nums) == '427':
        
        end = nums[2]
        endSplit = [end[:5], end[5:]]
        
        return nums[:2] + endSplit
    
    return None

## Function to check whether the mutuelle number hadn't been split but still fits pattern
def checkBlockFit(nums):
    
    if len(nums) == 1:
        fixNum = nums[0] 
    
        if len(fixNum) == 13:
            return [fixNum[:4], fixNum[4:6], fixNum[6:11], fixNum[11:]]
        
    return None
    
## Function to check whether dropping end part enables fit
def dropEnd(nums):
    
    if getLens(nums[:4]) == '4252':
        return nums[:4]
    
    return None

## Function to generate list of patterns with each digit off by 1
def oneOff(pattern):
    
    result = []
    
    digits = list(pattern)
    
    for i, digit in enumerate(digits):
        
        minus = digits[:i] + [str(int(digits[i]) - 1)] + digits[i+1:]
        minus = ''.join(minus)
        result.append(minus)
        
        plus = digits[:i] + [str(int(digits[i]) + 1)] + digits[i+1:]
        plus = ''.join(plus)
        result.append(plus)
        
    return result

## Function to determine if joining consecutive parts is off by one
def correctConsecOneOff(nums, pattern = '4252'):
    
    newNums = joinConsecutive(nums)
    
    fit = None
    
    for i, num in enumerate(newNums):
        
        if pattern in oneOff(getLens(num)):
            fit = i

    if fit is not None:
        return newNums[fit]
            
    return None

# =============================================================================
# Clinic Mutuelle Parser
# =============================================================================

def splitClinicMutuelle(num, pattern = '4252'):
    
    if num == '':
        return np.NaN
    
    ## Remove unwanted characters and slit at desired characters
    num = num.replace('//', '/')
    nums = re.split('[/ -]', num)
    
    ## Correct first part disctrict if 3 digits
    nums = correctDistrict(nums)
    
    ## Mutuelle number should have 4 parts with fixed character lengths
    ## If so, then just return split
    if getLens(nums) == pattern:
        return nums
    
    ## See if combining any consecutive pairs fits the pattern
    consec = correctConsecutive(nums)
    if consec:
        return consec
      
    ## See if splitting any of the parts fit the pattern
    split = correctSplit(nums)
    if split:
        return split
    
    ## See if the number didn't have any split characters but still fits
    block = checkBlockFit(nums)
    if block:
        return block
        
    ## See if there's an extra end part that can be dropped
    dropend = dropEnd(nums)
    if dropend:
        return dropend
        
    ## See if the lengths are off somewhere by one
    if pattern in oneOff(getLens(nums)):
        return nums
    
    ## See if lengths are off by one after joining consecutive parts
    offOneJoin = correctConsecOneOff(nums)
    if offOneJoin:
        return offOneJoin
    
    return np.NaN

# =============================================================================
# Household Mutuelle Parser
# =============================================================================

## Find all the non-alphanumeric characters
#nonAN = [re.findall('\W+', num) for num in m3] # read in json below
#nonAN = set(itertools.chain.from_iterable(nonAN))
#{' ', ' /', '$/', '+', '-', '.', '/', '//', ':', '?', '\\'}

def splitHouseholdMutuelle(num, pattern = '4252'):
    
    if num in ['', np.NaN]:
        return np.NaN

    ## Split at select characters
    num = num.replace('//', '/').replace('\\', '').replace('.','').\
        replace('$','').replace('+','').replace(':','').replace('?','')
                
    nums = re.split('[/ -]', num)
    
    ## Correct first part disctrict if 3 digits
    nums = correctDistrict(nums)
    
    ## Mutuelle number should have 4 parts with fixed character lengths
    ## If so, then just return split
    if getLens(nums) == pattern:
        return '-'.join(nums)
    
    ## See if combining any consecutive pairs fits the pattern
    consec = correctConsecutive(nums)
    if consec:
        return '-'.join(consec)
      
    ## See if splitting any of the parts fit the pattern
    split = correctSplit(nums)
    if split:
        return '-'.join(split)
    
    ## See if the number didn't have any split characters but still fits
    block = checkBlockFit(nums)
    if block:
        return '-'.join(block)
        
    ## See if there's an extra end part that can be dropped
    dropend = dropEnd(nums)
    if dropend:
        return '-'.join(dropend)
        
    ## See if the lengths are off somewhere by one
    if pattern in oneOff(getLens(nums)):
        return '-'.join(nums)
    
    ## See if lengths are off by one after joining consecutive parts
    offOneJoin = correctConsecOneOff(nums)
    if offOneJoin:
        return '-'.join(offOneJoin)
    
    return num

# =============================================================================
# See how it's doing
# =============================================================================

#import json
#
#path = 'C:/Users/gprenti/Box Sync/Graeme/Name Matching/Data/Mutuelle Numbers/'
#
#with open(path + 'mut_clinic.txt', 'r') as f:
#    m1 = json.loads(f.read())
#    
#with open(path + 'mut_cleanedhhsurvey.txt', 'r') as f:
#    m2 = json.loads(f.read())
#    
#with open(path + 'mut_rawhhsurvey.txt', 'r') as f:
#    m3 = json.loads(f.read())
    
# m1 = clinic mutuelle numbers
# m2 = cleaned mutuelle numbers from hh survey key
# m3 = raw mutuelle numbers from hh survey
    
### Clinic test
#results = {num: splitClinicMutuelle(num) for num in m1}
#unableToParse = {key:value for key, value in results.items() if value in ['utp']}

### Household test   
#resultsHH = {num: splitHouseholdMutuelle(num) for num in m3}
#unableToParseHH = {key:value for key, value in resultsHH.items() if value in ['utp']}

