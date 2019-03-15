from woe import get_woe_iv
import numpy as np
import pandas as pd
##########################
### MAIN WOE FUNCTIONS ###
##########################
def WOE_train_test(X_train, X_test, y_train, y_test, max_cont_bin = 5):
    ''' 
    Computes WOE on X_train and transofmrs X_train and X-test accordingly.
    '''
    
    #Get WOE
    WOE, WOE_concise = get_woe_iv(X_train, y_train, max_cont_bin = max_cont_bin) 
    WOE_no_gaps = close_bucket_gaps(WOE, X_train)
    
    #TRANSFORM BOTH TRAIN/TEST WITH THIS OUTPUT
    X_WOE_tranformed_train = woe_transform_all_cols(X_train, WOE_no_gaps)

    #apply function here to change WOE so that that MIN and MAX of previous coincide.

    X_WOE_tranformed_test = woe_transform_all_cols(X_test, WOE_no_gaps)

    return(X_WOE_tranformed_train, X_WOE_tranformed_test, WOE, WOE_concise, WOE_no_gaps)



############################################################################################
################################# HIDDEN HELPER FUNS #######################################
############################################################################################


def woe_transform_all_cols(X, WOE_df):
    X_new = X.copy() ### TODO: IS NECESSARY? ###
    for col in X_new.columns:
        curr_WOE = WOE_df[WOE_df.VAR_NAME == col]
        X_new[col] = X_new[col].apply(lambda x: get_single_woe_value(x, curr_WOE, col)).fillna(0)
    
    return(X_new)


def get_single_woe_value(x, curr_WOE, col):
            
    #check for first occurence of True
    boo_arr = (x <= curr_WOE.MAX_VALUE)
    #look for least idx where TRUE. IF none, then use last idx.
    true_idx = [i for i,x in enumerate(boo_arr) if x]
    if len(true_idx) != 0:
        WOE_bin_idx = min(true_idx)
        # print("first case")
        return(curr_WOE.iloc[WOE_bin_idx].WOE)
    else:
        # print('second case')
        # print(boo_arr)
        WOE_bin_idx = boo_arr.index.max()
        # print(boo_arr.index.max())
        # print('curr_WOE', curr_WOE)
        # print('WOE_bin_idx', WOE_bin_idx)
        return(curr_WOE.loc[WOE_bin_idx].WOE)
        # len(boo_arr)
    # try: 
    #     WOE_bin_idx = min(min(true_idx), len(boo_arr))
    # except:
    #     print('curr_WOE', curr_WOE)
    #     print('col', col)
    #     print('true_idx', true_idx)
    #     print('x', x)
    # print(boo_arr)
    # print('true_idx', true_idx)
    # print('WOE_bin_idx', WOE_bin_idx)
    # print('x', x)
    # print('curr_WOE', curr_WOE)
    # print('curr_WOE.iloc[WOE_bin_idx].WOE', curr_WOE.iloc[WOE_bin_idx].WOE)







    # #### old version ###
    # #find the index in the WOE table that the value-to-be-transformed lies in            
    # boo_arr = (curr_WOE.MIN_VALUE <= x) & (x <= curr_WOE.MAX_VALUE)


    # #this is to handle edge case when max value of test is greater than that of train
    # if not any(boo_arr): 
    #     if x < curr_WOE.iloc[0].MIN_VALUE:
    #         WOE_bin_idx = 0
    #     elif x > curr_WOE.iloc[len(curr_WOE) - 1].MAX_VALUE:
    #         WOE_bin_idx = len(curr_WOE)
    #     # else:
    #     #     print(col, x)
    # if len([i for i, x in enumerate(boo_arr) if x]) == 0:
    #     print('assigning value to 0', x, col)
    #     return(0) #a hack because x may be inbetween row 1 max and row 2 min...
    #     #unfortunately nans aren't handled well and are getting mapped to 0 too
        
    # WOE_bin_idx = [i for i, x in enumerate(boo_arr) if x][0]
    # return(curr_WOE.iloc[WOE_bin_idx].WOE)



def close_bucket_gaps(WOE_df, X): #assumed that WOE was calculated on X.
    WOE_df_cols = WOE_df.columns
    for col in X.columns:
        if col in WOE_df.VAR_NAME.unique(): #should be, but for safety for users
            if np.issubdtype(X[col].dtype, np.number): #only numeric cols
                #for the current col, loop through WOE and close gap by bringing to middle value
                sub_WOE_df = WOE_df[WOE_df.VAR_NAME == col]
                #sort so missing value is last
                orig_ixs = sub_WOE_df.index
                sub_WOE_df.sort_values('MIN_VALUE', inplace = True)
                sub_WOE_df.index = orig_ixs
                for idx in sub_WOE_df[:-1].index:
                    curr_row_max = sub_WOE_df[sub_WOE_df.index == idx].MAX_VALUE.iloc[0] #TODO MAKE THIS PRETTIER
                    next_row_min = sub_WOE_df[sub_WOE_df.index == (idx + 1)].MIN_VALUE.iloc[0]
                    if curr_row_max < next_row_min:
                        midpoint = (next_row_min + curr_row_max) / 2
                        #update sub_WOE_df
                        sub_WOE_df.loc[sub_WOE_df.index == (idx + 1), 'MIN_VALUE'] = midpoint
                        sub_WOE_df.loc[sub_WOE_df.index == idx, 'MAX_VALUE'] = midpoint
                        # sub_WOE_df.iloc[(idx + 1)]['MIN_VALUE'] = midpoint
                        # sub_WOE_df.iloc[idx]['MAX_VALUE'] = midpoint
                #over-write WOE_df with sub_WOE_df
                WOE_df = WOE_df[WOE_df.VAR_NAME != col]
                WOE_df = pd.concat([WOE_df, sub_WOE_df])
    #TODO sort the rows by original order of VAR_NAMES so user isn't confused, unless this output is just under hood
    
    #sort columns
    WOE_df = WOE_df[WOE_df_cols]
    return(WOE_df)
    



######################################################
#         if X[col].dtype == 'object':
#             print(col)
#             display(curr_WOE)
        #if x is null, find row with min = max = null and highest counts.
#         if isinstance(x, float):
#             if np.isnan(x):
# #             print('NAN!')
#                 curr_WOE_null = curr_WOE[curr_WOE.MIN_VALUE.isna()]
# #             print(curr_WOE_null[curr_WOE_null.COUNT == curr_WOE_null.COUNT.max()].WOE)
#                 return(float(curr_WOE_null[curr_WOE_null.COUNT == curr_WOE_null.COUNT.max()].WOE))









# def woe_transform_for_arb_col(col, X, WOE_df):
#     curr_WOE = WOE_df[WOE_df.VAR_NAME == col]

#     X_new = X.copy()
#     # WARNING !!! FILLING NAS BECAUSE OF SHITTY NA HANDLING IN WOE SOFTWARE #
    
#     return(X_new)