#================================================
#
#  Utility functions for the pdinfo package
#
#================================================

import inspect
import pandas as pd


#--------------------------------------------------------------------------------------------------------------------------------------
# _calc_iqr
#
# Calculate IQR for a certain dataframe column, based on the two percentiles (val_q1, val_q3) and the multiplier (val_mul)
#--------------------------------------------------------------------------------------------------------------------------------------
def _calc_iqr(df, col, val_q1, val_q3, val_mul):
    q1 = df[col].quantile(val_q1)
    q3 = df[col].quantile(val_q3)
    iqr = q3 - q1
    iqr_min = q1 - val_mul * iqr
    iqr_max = q3 + val_mul * iqr
    ol_min = df[(df[col] < iqr_min)]
    ol_max = df[(df[col] > iqr_max)]
    return(iqr_min, iqr_max, ol_min, ol_max)


#-------------------------------------------------------------------------------
# _list_unique_values
# 
# Take a list and return a new list with the unique values in the original list
#-------------------------------------------------------------------------------
def _list_unique_values(l):
    l_set = set(l)             # 'list' to 'set'
    u_list = list(l_set)       # 'set' to 'list'
    u_list.sort()
    return(u_list)


#-------------------------------------------------------------------------
# _get_numeric_columns
#
# Get a dataframe and return the list of numeric columns in the dataframe
#-------------------------------------------------------------------------
def _get_numeric_columns(df):
    df_cols   = df.columns.tolist()             # df_cols   - List of columns
    col_types = df.dtypes.tolist()              # col_types - List of column types
    num_cols = []
    ix = 0
    for col in df_cols:
        col_type = col_types[ix].name
        if not col_type.startswith(('date', 'obj', 'cat', 'bool')):
            num_cols.append(col)
        ix += 1
    return(num_cols)


#-------------------------------------------------------------------------------------------------------------
# _retrieve_var_name
# 
# Retrieve the name of a variable. See link:
# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string/18425523#18425523
#-------------------------------------------------------------------------------------------------------------
def _retrieve_var_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

