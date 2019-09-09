'''
Functions to create clerical review docs from matchpairs
'''

# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import pandas as pd
import json
import xlsxwriter

# =============================================================================
# Functions
# =============================================================================
def confSum(series, wDict):
    
    return np.nansum([wDict[col] * value for col, value in series.items() if col in wDict.keys()])

def combineTopBottom(df1, df2):

    '''
    Combine matched pairs into one-line with \n
    '''
    
    clinicID = df1.RECORDID_
    redCap = df1.redcap_event_name
    childID = df2.RECORDID_
    hhid = df2.hhid
    
    df1 = df1.drop(['RECORDID_', 'index'], axis=1)
    df2 = df2.drop(['RECORDID_', 'index'], axis=1)
    
    def strf_time(date):
        try:
            datestring = date.strftime('%B, %Y')
        except:
            datestring = date
        return datestring
        
    make_upper = lambda s: str(s).upper()
    
    df1 = df1.assign(DOB_ = df1.DOB_.apply(strf_time))
    df1 = df1.applymap(make_upper).replace('NA',' ').replace('NAN', ' ').replace('NAT', ' ')

    df2 = df2.assign(DOB_ = df2.DOB_.apply(strf_time))
    df2 = df2.applymap(make_upper).replace('NA',' ').replace('NAN', ' ').replace('NAT', ' ')
    
    merged = pd.DataFrame()
    for (i1, row1), (i2, row2) in zip(df1.iterrows(), df2.iterrows()):
        
        mergedRow = {}
        for (col1, value1), (col2, value2) in zip(row1.iteritems(), row2.iteritems()):
            
            value = '\n'.join([str(value1), str(value2)])
            
            mergedRow.update({col1: value})
            
        merged = merged.append(pd.DataFrame(mergedRow, index=[0]), ignore_index=True, sort=False)

    merged = merged.assign(clinicID = clinicID, redCap = redCap, 
                           childID = childID, hhID = hhid)
    merged = merged.sort_values(by='clinicID').reset_index(drop=True)
    
    
    return merged

def Fuze(df1, df2, matchKey, fuzePath, renameTopDict, renameBottomDict, toKeep, sx=False):

    '''
    Combine match pairs into format for easy manual review and write to excel
    '''
    
    # =============================================================================
    # Match keys and confidence score
    # =============================================================================
    index1 = matchKey.level_0
    index2 = matchKey.level_1
    confScore = [int(score) for score in matchKey.confScore]
    
    # =============================================================================
    # Clinic Data
    # =============================================================================
    
    ## Sync clinic data with matchkey using index based on match keys, reset index for merge
    sub1 = df1.reindex(index = index1).reset_index()
    dob1 = sub1.dob_est
    sub1 = sub1.drop('dob_est', axis=1).fillna(' ')
    
    ## Use either Parent or HOH name (based off the max sum of similarity scores)
    ## WhichParentName = wPN
    if sx:
        wPN1 = matchKey.ParentKinyaSXCol.apply(lambda x: x.rstrip('[1-4]_sx'))
    else:
        wPN1 = matchKey.ParentKinyaCol.apply(lambda x: x.rstrip('[1-4]'))
    
    wPN1 = ['hohname' if name == 'hkinyaname' else 'parentname' for name in wPN1]
    sub1 = sub1.assign(wPN1 = wPN1)
    sub1 = sub1.assign(parentname = sub1.apply(lambda x: x[x.wPN1], axis=1))
    
    ## Select and rename columns
    sub1 = sub1.assign(dob_est = dob1)
    sub1 = sub1.rename(columns=renameTopDict)[toKeep + ['redcap_event_name']]
    
    # =============================================================================
    # HH Survey Data
    # =============================================================================
    
    ## Sync hh survey data with matchkey using index based on match keys, reset index for merge        
    sub2 = df2.reindex(index = index2).reset_index().fillna(' ')
    
    ## Display child/parent name variant based off max sum of similarity scores
    ## Parent names are pair-dependent, i.e. kinyaname1 and othername1 considered jointly
    ## Child is assumed to be same person, i.e kinyaname1 can mix with othername1,2,3
    extractVariant = lambda x: re.sub('[^1-9]','', x)
    if sx:
        wCKN = ['ckinyaname' + extractVariant(x) for x in matchKey.ChildKinyaSXCol]
        wCON = ['cothername' + extractVariant(x) for x in matchKey.ChildOtherSXCol]
        wPKN2 = ['pkinyaname' + extractVariant(x) for x in matchKey.ParentKinyaSXCol]
        wPON2 = ['pothername' + extractVariant(x) for x in matchKey.ParentKinyaSXCol]

    else:
        wCKN = ['ckinyaname' + extractVariant(x) for x in matchKey.ChildKinyaCol]
        wCON = ['cothername' + extractVariant(x) for x in matchKey.ChildOtherCol]
        wPKN2 = ['pkinyaname' + extractVariant(x) for x in matchKey.ParentKinyaCol]
        wPON2 = ['pothername' + extractVariant(x) for x in matchKey.ParentKinyaCol]
        
    sub2 = sub2.assign(wCKN = wCKN, wCON = wCON, wPKN2 = wPKN2, wPON2 = wPON2)

    childname = sub2.apply(lambda x: str(x[x.wCKN]) + ' ' + str(x[x.wCON]), axis=1)
    parentname = sub2.apply(lambda x: str(x[x.wPKN2]) + ' ' + str(x[x.wPON2]), axis=1)
    sub2 = sub2.assign(PATIENTNAME_ = childname, PARENTNAME_ = parentname)
    
    ## Select child data columns and rename
    sub2 = sub2.rename(columns=renameBottomDict)[toKeep + ['hhid']]
    
    # =============================================================================
    # Combine both dataframes  
    # =============================================================================
    merged = combineTopBottom(sub1, sub2)
    
    ## Add confidence score
    merged = merged.assign(confScore = confScore)
    
    # =============================================================================
    # Write to Excel
    # =============================================================================

    ## Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(fuzePath, engine='xlsxwriter')

    ## Convert the dataframe to an XlsxWriter Excel object.
    merged.to_excel(writer, sheet_name='pairs') 
    
    ## Get the xlsxwriter objects from the dataframe writer object.
    workbook  = writer.book
    worksheet = writer.sheets['pairs']
    
    ## Set column widths
    worksheet.set_column('B:C', 35)
    worksheet.set_column('D:E', 13)
    worksheet.set_column('F:G', 16)
    worksheet.set_column('H:H', 22)
    worksheet.set_column('N:N', 18)
    worksheet.set_column('P:P', 20)
    
    ## Background color for every other row
    ## Bottom border at end of succession of same clinic IDs
    
    ## Count up how many times a clinic id appears in manual review
    clinicIDs = list(merged.clinicID) # already sorted
    boxDict = {}
    for i, idValue in enumerate(clinicIDs):
        
        if idValue in boxDict.keys():
            boxDict[idValue]['rows'] += 1
            
        else:
            boxDict.update({idValue: {'index': i, 'rows': 1}})
    
    ## Get indeces for writing top & bottom borders for clinic ids appearing more than once
    topBorderRows = [value['index']+1 for value in boxDict.values() if value['rows'] > 1]
    bottomBorderRows = [value['index'] + value['rows'] for value in boxDict.values() if value['rows'] > 1]
    
    ## Formats
    grey = workbook.add_format({'fg_color': '#C0C0C0', 'text_wrap': True})
    wrap = workbook.add_format({'text_wrap': True})
    grey_top = workbook.add_format({'fg_color': '#C0C0C0', 'text_wrap': True, 'top': 3, 'italic': True})
    wrap_top = workbook.add_format({'text_wrap': True, 'top': 3, 'italic': True})
    grey_bottom = workbook.add_format({'fg_color': '#C0C0C0', 'text_wrap': True, 'bottom': 3, 'italic': True})
    wrap_bottom = workbook.add_format({'text_wrap': True, 'bottom': 3, 'italic': True})
    
    ## Apply formats
    for row in range(1, len(merged) + 1, 2):
        
        if row in topBorderRows:
            worksheet.set_row(row, cell_format=grey_top)
        elif row in bottomBorderRows:
            worksheet.set_row(row, cell_format=grey_bottom)
        else:
            worksheet.set_row(row, cell_format=grey)
            
        if row+1 in topBorderRows:
            worksheet.set_row(row+1, cell_format=wrap_top)
        elif row+1 in bottomBorderRows:
            worksheet.set_row(row+1, cell_format=wrap_bottom)
        else:
            worksheet.set_row(row+1, cell_format=wrap)
        
    workbook.close()
    
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# Data Fusion - Side by side

### Column renaming dictionaries
### CAPITAL_     = left/clinic
### _lowercase   = right/child
#renameClinicDict = {'index': 'level_0', 'recordid': 'RECORDID_', 'patientname': 'PATIENTNAME_',
#                    'parentname': 'PARENTNAME_', 'district_clean': 'DISTRICT_',
#                    'sector_clean': 'SECTOR_', 'cell_clean': 'CELL_',
#                    'village_clean': 'VILLAGE_', 'mdist': 'MDIST_', 
#                    'mhf': 'MHF_', 'mhh': 'MHH_', 'mindv': 'MINDV_',
#                    'mutuelle_number': 'MUTUELLE_', 'sexe': 'GENDER_', 'dob_est': 'DOB_'}
#
#renameChildDict = {'index': 'level_1', 'childid': '_childid', 'pkinyaname1': '_parentKinyaname',
#                   'pothername1': '_parentOthername', 'ckinyaname1': '_childKinyaname',
#                   'cothername1': '_childOthername', 'mutnum1': '_mutuelle',
#                   'm_dist': '_mdist', 'm_hc': '_mhf', 'm_hh': '_mhh', 'm_indv' : '_mindv',
#                   'min_dob': '_dob', 'cgender_chr': '_gender',  'district': '_district',
#                   'sector': '_sector', 'cell': '_cell', 'village': '_village'}
#
#
#tokeep =    ['level_0', 'level_1', 'RECORDID_', '_childid', 
#             'PATIENTNAME_', '_childKinyaname', '_childOthername',
#             'PARENTNAME_', '_parentKinyaname', '_parentOthername',
#             'DISTRICT_', '_district', 'SECTOR_', '_sector',
#             'CELL_', '_cell', 'VILLAGE_', '_village',
#             'MUTUELLE_', '_mutuelle',
#             'MDIST_', '_mdist', 'MHF_', '_mhf',
#             'MHH_', '_mhh', 'MINDV_', '_mindv',
#             'GENDER_', '_gender',  'DOB_', '_dob']
#
### Subset data using index based on match keys, then reset index for merge
#d1 = clinic.reindex(index = matches.level_0).reset_index().rename(columns=renameClinicDict)
#d2 = child.reindex(index = matches.level_1).reset_index().rename(columns=renameChildDict)
#merged = pd.concat([d1,d2], axis=1)[tokeep]
#merged = merged.assign(confScore = matches.confScore)
#
#p1 = clinic.reindex(index = possible_matches.level_0).reset_index().rename(columns=renameClinicDict)
#p2 = child.reindex(index = possible_matches.level_1).reset_index().rename(columns=renameChildDict)
#manReview = pd.concat([p1, p2], axis=1)[tokeep]
#manReview = manReview.assign(confScore = possible_matches.confScore)
#
#
##merged.to_csv(path + '0625/merged.txt', index=False, sep = '\t')
##manReview.to_csv(path + '0625/manReview.txt', index=False, sep = '\t')



#def Fuze(df1, df2, matchKey, fuzePath):
#    
#    ## Match key
#    index1 = matchKey.level_0
#    index2 = matchKey.level_1
#    confScore = matchKey.confScore
#         
#    ## Subset data using index based on match keys, then reset index for merge
#    sub1 = df1.reindex(index = index1)\
#        .reset_index().rename(columns=renameClinicDict)
#    sub2 = df2.reindex(index = index2)\
#        .reset_index().rename(columns=renameBottomDict)
#        
#    childname = sub2['_childKinyaname'].astype(str) + ' ' + sub2['_childOthername']
#    parentname = sub2['_parentKinyaname'].astype(str) + ' ' + sub2['_parentOthername']
#    sub2 = sub2.assign(PATIENTNAME_ = childname)
#    sub2 = sub2.assign(PARENTNAME_ = parentname)
#    sub2 = sub2.rename(columns=renameBottomDict)[toKeepTop]
#    sub1 = sub1[toKeepTop]
#    
#    merged = pd.DataFrame()
#    for (i1, row1), (i2, row2) in zip(sub1.iterrows(), sub2.iterrows()):
#    
#        t1 = row1.to_frame().transpose()
#        t2 = row2.to_frame().transpose()
#        
#        merged = merged.append(t1, ignore_index=True)
#        merged = merged.append(t2, ignore_index=True)
#        
#    
#    confScoreDouble = []
#    for score in confScore:
#        confScoreDouble.append(int(score))
#        confScoreDouble.append(int(score))
#    
#    merged = merged.assign(confScore = confScoreDouble)    
#    merged = merged.fillna('')
#    merged = merged.applymap(lambda s:s.upper() if type(s) == str else s)
#    merged = merged.replace('NA','')
#    merged = merged.assign(DOB_ = merged.DOB_.\
#            apply(lambda x: x.strftime('%B, %Y') if x is not pd.NaT else x))
#    
#    # Create a Pandas Excel writer using XlsxWriter as the engine.
#    writer = pd.ExcelWriter(fuzePath, engine='xlsxwriter')
#
#    # Convert the dataframe to an XlsxWriter Excel object.
#    merged.to_excel(writer, sheet_name='pairs')    
#    
#    # Get the xlsxwriter objects from the dataframe writer object.
#    workbook  = writer.book
#    worksheet = writer.sheets['pairs']
#    
#    worksheet.set_column('D:E', 30)
#    worksheet.set_column('F:G', 13)
#    worksheet.set_column('H:I', 16)
#    worksheet.set_column('J:J', 17)
#    worksheet.set_column('P:P', 18)
#    
#    greyTop = workbook.add_format({'fg_color': '#C0C0C0'}) #background color
#    greyTop.set_top()
#    greyBottom = workbook.add_format({'fg_color': '#C0C0C0'})
#    greyBottom.set_bottom()
#    dateFormat = workbook.add_format({'num_format': 'mmm yyyy'})
#    
#    for row in range(1, len(merged), 4):
#        worksheet.set_row(row, cell_format=greyTop)
#        worksheet.set_row(row + 1, cell_format=greyBottom)
#    
#    workbook.close()