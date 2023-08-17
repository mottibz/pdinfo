import pandas as pd
import numpy as np
#from pandas.api.types import is_numeric_dtype
#from functools import reduce
#import warnings
#import math
#from operator import itemgetter
import weakref
from scipy.stats import shapiro
from sklearn.ensemble import IsolationForest
import traceback
import sys
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl as xl

import sys
# import os
# print('Current working directory: ' + os.getcwd())
sys.path.insert(0, '/Users/motti/Documents/GitHub/DataScience/pdinfo')
from src.utils import _calc_iqr, _list_unique_values, _get_numeric_columns, _retrieve_var_name



# Define the class and its accessor
@pd.api.extensions.register_dataframe_accessor('inf')
class PandasInfo:

    __internal_call = False
    _percentiles = [0.5, 0.75, 0.90, 0.95]                                     # Set default percentiles for describing numeric columns

    def __init__(self, pandas_obj):
        self._finalizer = weakref.finalize(self, self._cleanup)
        self._validate(pandas_obj)
        self._obj = pandas_obj

    def _cleanup(self):
        del self._obj

    def remove(self):
        self._finalizer()

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, pd.DataFrame):                     # Verify that this is a pandas dataframe
            raise AttributeError("Operation can only take place on a Pandas dataframe")


    #==================================================================================================================
    # type - zero, mean
    #==================================================================================================================
    def fillna_numeric_cols_with_value(self, type='mean', cols=None):

        if cols == None:
            cols = _get_numeric_columns(self._obj)

        for col in cols:
            if type == 'mean':
                self._obj[col] = self._obj[col].fillna(self._obj[col].mean())
            elif type == 'zero':
                self._obj[col] = self._obj[col].fillna(0)

        return(self._obj)


    #==================================================================================================================
    # inspect - Inspect a dataframe's data for issues
    #
    # key_column - Column that is the key for the dataframe. If provided:
    #              1. A check is done to verify that there are no duplicate keys
    #              2. Removed from rare values check
    #==================================================================================================================
    def inspect(self, key_column=None, rare_threshold=0.05, corr_threshold=0.8):

        print('\nINSPECT DATAFRAME: ' + _retrieve_var_name(self._obj)[0])

        # Check for suplicate rows
        print('\nChecking for duplicate rows:')
        dup_rows = self._obj.duplicated().sum()
        if dup_rows > 0:
            print(f'- Found {dup_rows} duplicated rows.')
            df_dup = self._obj[self._obj.duplicated()]
            #lgr.info('Duplicated rows sample:\n' + str(df_dup.head(5)))
            if key_column is not None:
                dup_users = _list_unique_values(df_dup[key_column].values.tolist())
                print('- Found ' + str(len(dup_users)) + ' unique duplicates. Sample: ' + str(dup_users[:10]))
                #df = df.drop_duplicates()
        else:
            print('- No duplicate rows found.')

        # Check for duplicate columns
        print('\nChecking for duplicate columns:')
        dup_cols = self._obj.columns[self._obj.columns.duplicated()]
        if len(dup_cols) > 0:
            print(f'- Found {len(dup_cols)} duplicate columns.')
            # THis is the only way that dropping duplicate columns works. This is not found anywhere!
            #df = df.T[df.T.index.duplicated(keep='first')].T
        else:
            print('- No duplicate columns found.')

        # Check for missing values
        print('\nChecking for missing values:')
        missing_values = self._obj.isnull().sum()
        missing_cols   = missing_values[missing_values > 0].index.tolist()
        if len(missing_cols) > 0:
            print(f'- Found {len(missing_cols)} columns with missing values.')
            #df = df.drop(missing_cols, axis=1)
        else:
            print('- No missing values found.')

        # Check for category columns with rare values and suggest to group them together
        thresh = rare_threshold
        print('\nChecking for category columns with rare values (threshhold=' + str(thresh) + '):')
        col1 = 'Column'
        col2 = '# Values'
        col3 = '# Rare Values'
        rare_cols = pd.DataFrame(columns=[col1, col2, col3])
        cols = self._obj.select_dtypes(include=["object", "category"]).columns.tolist()
        if len(cols) > 0:
            for col in cols:
                val_cnt = self._obj[col].value_counts(normalize=True)
                vals = val_cnt[val_cnt < thresh]
                #lgr.info(f'Column: {col}, number of values: {len(val_cnt)}, number of rare values: {len(vals)}')
                #lgr.info(str(val_cnt))
                if len(vals) > 0:
                    if key_column is not None and col != key_column:
                        rare_cols = rare_cols.append({col1: col, col2: len(val_cnt), col3: len(vals)}, ignore_index=True)
            if len(rare_cols) > 0:
                print('- Recommend reviewing the following columns and grouping the rare values together:')
                print(str(rare_cols))
            else:
                print('- No rare values found.')
        else:
            print('- No category columns found.')

        # Check for columns with only one value
        #lgr.info('\nChecking for columns with only one value:')
        #one_val_cols = df.nunique()[df.nunique() == 1].index.tolist()

        # Check for columns with infinite values
        # lgr.info('\nChecking for columns with infinite values:')
        # inf_vals = df.replace([np.inf, -np.inf], np.nan).isnull().sum() # - missing_values
        # if inf_vals.sum() > 0:
    
        # Check for columns with high correlation
        corr_thresh = corr_threshold                                                                           # High correlation threshold
        print('\nChecking for high correlation columns (threshhold=' + str(corr_thresh) + '):')
        corr_matrix = self._obj.corr().abs()                                                                   # Use absolute correlation matrix
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))              # Get the upper triangle of the matrix
        high_corr_cols = [col for col in upper_triangle.columns if any(upper_triangle[col] >= corr_thresh)]    # Get the columns with high correlation
        if len(high_corr_cols) > 0:
            print(f'- Found {len(high_corr_cols)} columns with >= {corr_thresh} correlation:')
            for col in high_corr_cols:
                print(f"- '{col}' is highly correlated with {upper_triangle[col][upper_triangle[col] > corr_thresh].index.tolist()}")
        else:
            print('- Dataframe has no highly correlated columns.')


    #==================================================================================================================
    # get_outlier_max_value - Inspect a dataframe's column for outliers
    #
    # col           - Column name to inspect
    # contamination - Proportion of outliers in the dataset
    #==================================================================================================================

    def get_outlier_max_value(self, col, contamination=0.005):

        use_isolationforest = False
        if use_isolationforest:
            # Calculate the anomaly score and value
            df_col = pd.DataFrame(self._obj[col])
            dft = self._obj[col].values.reshape(-1, 1)
            modelIF = IsolationForest(contamination=contamination, n_jobs=-1, random_state=42, verbose=0)
            modelIF.fit(dft)
            df_col['anomaly']=modelIF.predict(dft)
            # df1 = df_col[df_col['anomaly'] == 1]
            # fld_allowed_max1 = df1[col].max()
            df1 = df_col[df_col['anomaly'] == -1]
            fld_allowed_max2 = df1[col].min()
            # return((fld_allowed_max1 + fld_allowed_max2) / 2)

        _, fld_allowed_max2, _, _ = _calc_iqr(self._obj, col, 0.02, 0.98, 3)

        return(fld_allowed_max2)


    #==================================================================================================================
    # inspect_outliers - Inspect a dataframe's data for outliers
    #
    # exclude_cols - List of columns to exclude from the analysis
    #==================================================================================================================

    def inspect_outliers(self, exclude_cols=None, contamination=0.005):

        print('\nINSPECT FOR OUTLIERS IN DATAFRAME: ' + _retrieve_var_name(self._obj)[0])

        df_cols   = self._obj.columns.tolist()             # df_cols   - List of columns
        col_types = self._obj.dtypes.tolist()              # col_types - List of column types
        df_cols_to_check = []
        ix = 0
        for col in df_cols:
            col_type = col_types[ix].name
            if not col_type.startswith(('date', 'obj', 'cat', 'bool')):
                if exclude_cols is not None and col in exclude_cols:
                    continue
                df_cols_to_check.append(col)
            ix += 1
        #print('Checking columns: ' + str(df_cols_to_check))

        col1 = 'Feature'
        col2 = 'Negative Min?'
        col3 = 'Negative Value'
        col4 = 'Possible Outlier?'
        col5 = 'Outlier Value 0.99'
        col6 = 'Outlier Value Max'
        col7 = 'Outlier Multiplier'
        col8 = 'IQR (0.75, 0.98)'
        #col9 = 'Anomaly Avg Score'
        col10 = 'Anomaly Info (' + str(contamination) + ')'
        df_result = pd.DataFrame(columns=[col1, col2, col3, col4, col5, col6, col7])

        iqr1_min_percentile = 0.25
        iqr1_max_percentile = 0.75
        iqr1_multiplier = 1.5
        iqr2_min_percentile = 0.02
        iqr2_max_percentile = 0.98
        iqr2_multiplier = 3

        for ix in tqdm(range(0, len(df_cols_to_check))):
            col = df_cols_to_check[ix]
            #for col in df_cols_to_check:
            # if exclude_cols is not None and col in exclude_cols:
            #     continue

            try:
                #print('Column: ' + col)
                perc = self._obj[col].describe(percentiles=[0.99])
                val_99  = perc['99%']
                val_min = perc['min']
                val_max = perc['max']

                # In our situation a field with negative values include outliers that should be zeroed
                if val_min < 0:
                    negative_min = True
                else:
                    negative_min = False

                # Field includes outliers of max value is X times (e.g. 10) larger than the 99% value
                # if val_99 < (val_max / 10):
                # If 99% value + 10% is less than the max value, then there are possible outliers
                if (val_99 * 1.1) < val_max:
                    outlier = True
                else:
                    outlier = False

                if outlier:

                    # Calculate the IQR (max for 0.75 and max for 0.98)
                    iqr_min1, iqr_max1, _, _ = _calc_iqr(self._obj, col, iqr1_min_percentile, iqr1_max_percentile, iqr1_multiplier)
                    iqr_min2, iqr_max2, _, _ = _calc_iqr(self._obj, col, iqr2_min_percentile, iqr2_max_percentile, iqr2_multiplier)
                    iqr_txt = '{:,.2f}'.format(iqr_min1) + '-' + '{:,.2f}'.format(iqr_max1) + ' - ' + '{:,.2f}'.format(iqr_min2) + '-' + '{:,.2f}'.format(iqr_max2)

                    # Calculate the anomaly score and value
                    df_col = pd.DataFrame(self._obj[col])
                    dft = self._obj[col].values.reshape(-1, 1)
                    modelIF = IsolationForest(contamination=contamination, n_jobs=-1, random_state=42, verbose=0)
                    modelIF.fit(dft)
                    #ascore = modelIF.decision_function(dft)
                    #anom   = modelIF.predict(dft)
                    # print(ascore)
                    # print(anom)
                    #ascore_avg = np.mean(ascore)
                    #df_col['scores'] =modelIF.decision_function(dft)
                    df_col['anomaly']=modelIF.predict(dft)
                    df1 = df_col[df_col['anomaly'] == 1]
                    fld_allowed_max1 = df1[col].max()

                    df1 = df_col[df_col['anomaly'] == -1]
                    #fld_allowed_max2 = df1[col].min()

                    df2 = df1[df1[col] >= fld_allowed_max1]
                    fld_allowed_max2 = df2[col].min()

                    # print("Max allowed: ", fld_allowed_max2, "Accuracy percentage:", 100 * list(dfcol['anomaly']).count(-1) / (outliers_counter))
                    #anom_val = str(len(anom[anom == -1])) + ' = ' + '{:,.2f}%'.format(len(anom[anom == -1]) / len(self._obj) * 100) + ' - ' + '{:,.2f}'.format(fld_allowed_max)
                    anom_val = str(len(df1)) + ' = ' + '{:,.2f}%'.format(len(df1) / len(self._obj) * 100) + ' - ' + '{:,.2f}'.format(fld_allowed_max1) + '-' + '{:,.2f}'.format(fld_allowed_max2)

                if negative_min and outlier:
                    df_result = df_result.append({col1: col, col2: negative_min, col3: '{:,.2f}'.format(val_min), col4: outlier, col5: '{:,.2f}'.format(val_99),
                                                col6: '{:,.2f}'.format(val_max), col7: '{:,.2f}'.format((val_max / val_99)), col8: iqr_txt, col10: anom_val}, ignore_index=True)
                elif negative_min:
                    df_result = df_result.append({col1: col, col2: negative_min, col3: '{:,.2f}'.format(val_min), col4: outlier, col5: '-', col6: '-', col7: '-', col8: '-', col10: '-'}, ignore_index=True)
                elif outlier:
                    df_result = df_result.append({col1: col, col2: negative_min, col3: '-', col4: outlier, col5: '{:,.2f}'.format(val_99),
                                                col6: '{:,.2f}'.format(val_max), col7: '{:,.2f}'.format((val_max / val_99)), col8: iqr_txt, col10: anom_val}, ignore_index=True)
            except Exception:
                print('inspect_outliers() exception for column: ' + col)
                print(traceback.format_exc())
                # or
                print(sys.exc_info()[2])                


        if len(df_result) == 0:
            print('\nNo possible outliers found.')
        else:
            print('\nPossible outliers:\n')  # + str(df_result))
            # Set title strings to center
            df_result = df_result.style.set_table_styles([dict(selector = 'th', props=[('text-align', 'center')])])
            # Set columns to display on the left
            df_result = df_result.set_properties(subset=['Feature'], **{'text-align': 'left'})

        return(df_result)


    #==================================================================================================================
    # info
    #
    # Create a table that counts the frequency of occurrence or summation of values, and also include number of zeros
    #
    # Parameter: Not provided --> Create information similar to info() + more
    #            String       --> Column name to be summarized
    #            List         --> List of column names to be summarized
    #
    # Returns:   Summary dataframe
    #==================================================================================================================
    def info(self, cols_to_report=None, percentiles=_percentiles, cum_limit=None, style=False):

        df_cols   = self._obj.columns.tolist()             # df_cols   - List of columns
        col_types = self._obj.dtypes.tolist()              # col_types - List of column types

        #-----------------------------------------------------------------------------------
        # If cols_to_report is not provided, create information similar to info() + more
        #-----------------------------------------------------------------------------------
        if cols_to_report == None:
            
            num_cols = len(df_cols)
            num_rows = len(self._obj)
            if self.__internal_call:
                df_base_info = ['{:03.2f} MB'.format(self._obj.memory_usage(deep=True).sum() / 1024 ** 2), '{:,}'.format(num_cols), '{:,}'.format(num_rows)]
            else:
                try:
                    print('Dataframe name:    {}'.format(_retrieve_var_name(self._obj)[0]))
                except:
                    pass
                print('Memory usage:      {:03.2f} MB'.format(self._obj.memory_usage(deep=True).sum() / 1024 ** 2))
                print('Number of columns: {:,}'.format(num_cols))
                print('Number of rows:    {:,}'.format(num_rows))

            # Add missing info
            info_result = pd.DataFrame(self._obj.isna().sum()).reset_index().rename(columns={'index':'Column', 0:'Missing'})
            #info_result = pd.DataFrame(self._obj.isna().sum()).reset_index().rename(columns={0:'Missing'})
            info_result['Missing %'] = (info_result['Missing'] / num_rows * 100)    # .map('{:,.2f}%'.format)
            
            info_result.insert(loc=1, column='Type', value=col_types)           # Add column 'Type'
            info_result['Zeros'] = 0                                            # Add columns for number of zeros and percent
            info_result['Zeros %'] = 0
            ix = 0
            for col in df_cols:
                zero_count = (self._obj[col] == 0).sum()
                # info_result.loc[ix, 'Zeros']   = '{:,}'.format(zero_count)
                # info_result.loc[ix, 'Zeros %'] = '{:,.2f}%'.format(zero_count / num_rows * 100)
                info_result.loc[ix, 'Zeros']   = zero_count
                info_result.loc[ix, 'Zeros %'] = zero_count / num_rows * 100
                ix += 1

            # Add Sum, Min, Max, Unique, Percentiles (for numeric columns)
            info_result['Sum'] = None # '-'
            info_result['Min'] = None # '-'
            info_result['Max'] = None # '-'
            info_result['Unique'] = '-'
            for p in percentiles:
                nm = '{:.0%}'.format(p)
                info_result[nm] = '-'

            ix = 0
            for col in df_cols:
                col_type   = col_types[ix].name
                col_unique = self._obj[col].nunique()
                info_result.loc[ix, 'Unique'] = col_unique

                if col_type.startswith(('date')):
                    col_min = self._obj[col].min()
                    col_max = self._obj[col].max()
                    info_result.loc[ix, 'Min'] = col_min
                    info_result.loc[ix, 'Max'] = col_max

                if not col_type.startswith(('date', 'obj')):
                    col_min = self._obj[col].min()
                    col_max = self._obj[col].max()
                    col_sum = self._obj[col].sum()

                    perc = self._obj[col].describe(percentiles=percentiles)
                    for p in percentiles:
                        nm = '{:.0%}'.format(p)
                        info_result.loc[ix, nm] = '{:,.2f}'.format(perc[nm])

                    if col_type.startswith(('flo')):
                        info_result.loc[ix, 'Sum'] = col_sum                # '{:,.2f}'.format(col_sum)
                        info_result.loc[ix, 'Min'] = col_min                # '{:,.2f}'.format(col_min)
                        info_result.loc[ix, 'Max'] = col_max                # '{:,.2f}'.format(col_max)
                    else:
                        info_result.loc[ix, 'Sum'] = col_sum                # '{:,}'.format(col_sum)
                        info_result.loc[ix, 'Min'] = col_min                # '{:,}'.format(col_min)
                        info_result.loc[ix, 'Max'] = col_max                # '{:,}'.format(col_max)
                ix += 1

            # Check the distribution for numeric columns
            info_result['p-value'] = '-'
            ix = 0
            for col in df_cols:
                col_type = col_types[ix].name
                if not col_type.startswith(('date', 'obj')):
                    spr = shapiro(self._obj[col])                                 # Perform Shapiro-Wilk test for normality
                    if spr.pvalue > 0.05:                                         # Check the p-value to figure out if the data is normally distrubuted
                        m = 'Gaussian'
                    else:
                        m = 'Non-gaussian'
                    # info_result.loc[ix, 'p-value'] = '{:.2f}'.format(spr.statistic) + ', ' + '{:.2f}'.format(spr.pvalue) + ': ' + m
                    info_result.loc[ix, 'p-value'] = '{:.2f}'.format(spr.pvalue) + ': ' + m
                ix += 1

            if style:
                try:
                    fields_format = {'Missing':'{:,}', 'Missing %':'{:,.2f}%', 'Zeros':'{:,}', 'Zeros %':'{:,.2f}%', 'Sum':'{:,.2f}', 'Min':'{:,.2f}', 'Max':'{:,.2f}'}
                    info_result = info_result.style.format(fields_format, na_rep='-') \
                        .set_table_styles([dict(selector = 'th', props=[('text-align', 'center')])]) \
                        .set_properties(subset=['Column', 'Type'], **{'text-align': 'left'})
                    
                    #info_result.to_excel('info_xl.xlsx', engine='openpyxl')
                    #with pd.ExcelWriter("info_xl.xlsx", mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                    #with pd.ExcelWriter("info_xl.xlsx", engine="openpyxl") as writer:
                        #info_result.to_excel(writer, sheet_name="info()")
                        # df2.to_excel(writer, sheet_name="Sheet_5")
                        # df3.to_excel(writer, sheet_name="Sheet_6")


                except:
                    pass

            if self.__internal_call:
                return df_base_info, info_result
            else:
                return info_result
        

        #=========================================================================================
        # In case column(s) name(s) provided, present groupby info
        #=========================================================================================

        #-----------------------------------------------------------------------
        # If cols_to_report is a string, convert to list
        #-----------------------------------------------------------------------
        if isinstance(cols_to_report, str):
            cols_to_report = [cols_to_report]

        #-----------------------------------------------------------------------
        # If cols_to_report is not either of the above, raise an error
        #-----------------------------------------------------------------------
        if not isinstance(cols_to_report, list):
            raise AttributeError('Parameter error. Can be none, a string, or a list')

        col_name = 'Count'
        group_data = self._obj.groupby(cols_to_report).size().reset_index(name=col_name)
        info_result = group_data.sort_values([col_name] + cols_to_report, ascending=False).reset_index(drop=True)

        # Include percents
        total = info_result[col_name].sum()
        info_result['%'] = (info_result[col_name] / total) * 100

        # Keep track of cumulative counts or totals as well as their relative percent
        info_result[f'Cumulative {col_name}'] = info_result[col_name].cumsum()
        info_result['Cumulative %'] = (info_result[f'Cumulative {col_name}'] / total) * 100

        # If a limit for the cimulative % is provided, filter the info_result table
        if cum_limit:
            if cum_limit < 1 or cum_limit > 99:
                raise AttributeError('Parameter error. Cumulative limit must be between 1 and 99')
            
            left_over   = info_result[info_result['Cumulative %'] > cum_limit]
            info_result = info_result[info_result['Cumulative %'] <= cum_limit]

            # Add a row for the left over
            cnt = left_over[col_name].sum()
            pct = left_over['%'].sum()
            cumcnt = left_over[f'Cumulative {col_name}'].max()
            cumpct = 100 - info_result['Cumulative %'].max()
            info_result = info_result.append({col_name: cnt, '%': pct, f'Cumulative {col_name}': cumcnt, 'Cumulative %': cumpct}, ignore_index=True)
            info_result.fillna('-', inplace=True)

        if style:
            try:
                # Style the output
                fields_format = {'%': '{:.2f}%', 'Cumulative %': '{:.2f}%', 'Count': '{0:,.0f}', f'{col_name}': '{0:,.0f}', f'Cumulative {col_name}': '{0:,.0f}'}
                info_result = info_result.style.format(fields_format) \
                    .set_table_styles([dict(selector = 'th', props = [('text-align', 'center')])]) \
                    .set_properties(subset = cols_to_report, **{'text-align': 'left'})
            except:
                pass

        return info_result


    #==================================================================================================================
    # to_excel - Create an Excel file with information about the dataframe
    #
    # name - Excel file name. Default is 'info.xlsx'
    #==================================================================================================================

    def to_excel(self, name='info.xlsx'):
        self.__internal_call = True

        # Call info() to get both the base info and the info() info
        #-----------------------------------------------------------
        i1, s2 = self.info(style=True)
        col1_name = 'DataFrame Information'
        col2_name = 'Amount'
        # Adding " " before the numebred text aligns the text to the left in Excel and removes the warning about the numbers stored as text
        d = {col1_name: ['Memory usage:', 'Number of columns:', 'Number of rows:'], col2_name: [" "+i1[0], " "+i1[1], " "+i1[2]]}
        s1 = pd.DataFrame(data=d)
        s1 = s1.style.set_table_styles([dict(selector = 'th', props = [('text-align', 'center')])]) \
            .set_properties(subset = [col1_name], **{'text-align': 'left'}) \
            .set_properties(subset = [col2_name], **{'text-align': 'right'})
        # display(s1)
        # display(s2)

        # Write the dataframes to Excel as separate tabs/sheets
        with pd.ExcelWriter(name, engine="openpyxl") as writer:
            s1.to_excel(writer, sheet_name="Base Info")
            s2.to_excel(writer, sheet_name="info()")

        self.__internal_call = False
        print("Excel file '" + name + "' created successfully.")


    #===========================  DRAWING FUNCTIONS  ===========================
  

    # Plot a correkation heatmap
    def plot_corr_heatmap(self, title, figsize=(30, 16), exclude_cols=None):
        cols = _get_numeric_columns(self._obj)
        if exclude_cols is not None:
            cols = [col for col in cols if col not in exclude_cols]
        plt.figure(figsize=figsize)

        corr = self._obj[cols].corr()
        # sns.heatmap(corr, annot=True, cmap='Blues')
        mask = np.triu(np.ones_like(corr))                                    # Create the mask
        sns.heatmap(corr, cmap="YlGnBu", annot=True, mask=mask)               # Plot the triangle correlation heatmap

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(title, fontsize=20)
        plt.show()


    # Plot a distribution plot (scatter / histplot / boxplot)
    def plot_dist(self, columns=None, label=None, kind='scatter', title=None, hue=None, figsize=None, exclude_cols=None):

        # if label is None:
        #     print('ERROR: label must be provided')
        #     return

        if columns is None:
            columns = _get_numeric_columns(self._obj)
            if exclude_cols is not None:
                columns = [col for col in columns if col not in exclude_cols]

        # If the input is a string (one name), convert to a list
        if isinstance(columns, str):
            columns = [columns]

        if (kind =='boxplot'):
            pics_in_row = 1
            if figsize is None:
                figsize = (18, 4)
        else:
            pics_in_row = 2
            if len(columns) < pics_in_row:
                pics_in_row = len(columns)
            if figsize is None:
                figsize = (18, 10)

        # pics_in_row = 1
        # if len(columns) < pics_in_row:
        #     pics_in_row = len(columns)
        # if figsize is None:
        #     if (kind =='boxplot'):
        #         figsize = (18, len(columns) * 2)
        #     else:
        #         figsize = (16, 12)

        plt.figure(figsize=figsize)
        
        for idx, column in enumerate(columns):
            # if title is None:
            #     title1 = column + ' Vs. ' + label

            f = plt.subplot(int(len(columns) / pics_in_row) + 1, pics_in_row, idx+1)
            if kind == 'scatter':
                #g = sns.scatterplot(data = self._obj[columns], x = column, y = label, hue=hue)
                g = sns.scatterplot(data = self._obj, x = column, y = label, hue=hue)
            elif (kind == 'histplot'):
                g = sns.histplot(data = self._obj[columns], x = column, hue=hue, kde=True)
            elif (kind =='boxplot'):
                mean_shape = dict(markerfacecolor='yellow', marker='D', markeredgecolor='yellow')
                g = sns.boxplot(x = column, data=self._obj[columns], hue=hue, showmeans=True, meanprops=mean_shape)
                #plt.semilogx()
        plt.tight_layout()
        plt.show()


    # Plot a count plot (countplot / barplot)
    def plot_count(self, columns=None, label=None, kind = 'countplot', title=None, hue=None, figsize=None, exclude_cols=None):

        if label is None:
            print('ERROR: label must be provided')
            return

        if columns is None:
            columns = _get_numeric_columns(self._obj)
            if exclude_cols is not None:
                columns = [col for col in columns if col not in exclude_cols]

        if figsize is None:
            figsize = (18, len(columns))
        plt.figure(figsize=figsize)
        
        for idx, column in enumerate(columns):
            if title is None:
                title = column + ' Vs. ' + label
            f = plt.subplot(int(len(columns) / 4) + 1, 4, idx+1)
            if kind == 'countplot':
                g= sns.countplot(x = column, data = self._obj[columns], hue=hue, title=title)
            if kind == 'barplot':
                g= sns.barplot(x = column, y=label, data = self._obj[columns], hue=hue, title=title)
        plt.tight_layout()
        plt.show()





