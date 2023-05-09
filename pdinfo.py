import pandas as pd
import numpy as np
#from pandas.api.types import is_numeric_dtype
#from functools import reduce
#import warnings
import weakref
#import math
#from operator import itemgetter
from scipy.stats import shapiro
import inspect


@pd.api.extensions.register_dataframe_accessor('inf')
class PandasInfo:

    _percentiles = [0.5, 0.75, 0.90, 0.95]           # Set default percentiles for describing numeric columns

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


    #-------------------------------------------------------------------------------------------------------------
    # _aux_retrieve_name - Retrieve the name of a dtaframe
    #
    # https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string/18425523#18425523
    #-------------------------------------------------------------------------------------------------------------
    def _retrieve_var_name(self, var):
        callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
        return [var_name for var_name, var_val in callers_local_vars if var_val is var]


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
    def info(self, cols_to_report=None, percentiles=_percentiles):

        df_cols   = self._obj.columns.tolist()             # df_cols   - List of columns
        col_types = self._obj.dtypes.tolist()              # col_types - List of column types

        #-----------------------------------------------------------------------------------
        # If cols_to_report is not provided, create information similar to info() + more
        #-----------------------------------------------------------------------------------
        if cols_to_report == None:
            
            num_cols = len(df_cols)
            num_rows = len(self._obj)
            print('Dataframe name:    {}'.format(self._retrieve_var_name(self._obj)[0]))
            print('Memory usage:      {:03.2f} MB'.format(self._obj.memory_usage(deep=True).sum() / 1024 ** 2))
            print('Number of columns: {}'.format(num_cols))
            print('Number of rows:    {}'.format(num_rows))

            # Add missing info
            results = pd.DataFrame(self._obj.isna().sum()).reset_index().rename(columns={'index':'Column', 0:'Missing'})
            results['Missing %'] = (results['Missing'] / num_rows * 100)    # .map('{:,.2f}%'.format)

            # Add column types
            results.insert(loc=1, column='Type', value=col_types)            

            # Add number of zeros and percent
            results['Zeros'] = 0
            results['Zeros %'] = 0
            ix = 0
            for col in df_cols:
                zero_count = (self._obj[col] == 0).sum()
                # results.loc[ix, 'Zeros']   = '{:,}'.format(zero_count)
                # results.loc[ix, 'Zeros %'] = '{:,.2f}%'.format(zero_count / num_rows * 100)
                results.loc[ix, 'Zeros']   = zero_count
                results.loc[ix, 'Zeros %'] = (zero_count / num_rows * 100)
                ix += 1

            # Add Min, Max, Sum, Unique for numeric columns
            results['Sum'] = '-'
            results['Min'] = '-'
            results['Max'] = '-'
            for p in percentiles:
                nm = '{:.0%}'.format(p)
                results[nm] = '-'
            results['Unique'] = '-'
            ix = 0
            for col in df_cols:
                col_type = col_types[ix].name
                col_unique = self._obj[col].nunique()
                results.loc[ix, 'Unique'] = col_unique

                if col_type.startswith(('date')):
                    col_min = self._obj[col].min()
                    col_max = self._obj[col].max()
                    results.loc[ix, 'Min'] = col_min
                    results.loc[ix, 'Max'] = col_max

                if not col_type.startswith(('date', 'obj')):
                    col_min = self._obj[col].min()
                    col_max = self._obj[col].max()
                    col_sum = self._obj[col].sum()

                    perc = self._obj[col].describe(percentiles=percentiles)
                    for p in percentiles:
                        nm = '{:.0%}'.format(p)
                        results.loc[ix, nm] = '{:,.2f}'.format(perc[nm])

                    if col_type.startswith(('flo')):
                        results.loc[ix, 'Sum'] = '{:,.2f}'.format(col_sum)
                        results.loc[ix, 'Min'] = '{:,.2f}'.format(col_min)
                        results.loc[ix, 'Max'] = '{:,.2f}'.format(col_max)
                    else:
                        results.loc[ix, 'Sum'] = '{:,}'.format(col_sum)
                        results.loc[ix, 'Min'] = '{:,}'.format(col_min)
                        results.loc[ix, 'Max'] = '{:,}'.format(col_max)
                ix += 1

            # Check the distribution for numeric columns
            results['p-value'] = '-'
            ix = 0
            for col in df_cols:
                col_type = col_types[ix].name
                if not col_type.startswith(('date', 'obj')):
                    spr = shapiro(self._obj[col])                                 # Perform Shapiro-Wilk test for normality
                    if spr.pvalue > 0.05:                                         # Check the p-value to figure out if the data is normally distrubuted
                        m = 'Gaussian'
                    else:
                        m = 'Non-gaussian'
                    results.loc[ix, 'p-value'] = '{:.2f}'.format(spr.statistic) + ', ' + '{:.2f}'.format(spr.pvalue) + ': ' + m
                ix += 1


            #format_dict = {'%': '{:.2f}%', 'Cumulative %': '{:.2f}%', 'Count': '{0:,.0f}', f'{col_name}': '{0:,.0f}', f'Cumulative {col_name}': '{0:,.0f}'}
            format_dict = {'Missing %':'{:,.2f}%', 'Zeros':'{:,}', 'Zeros %':'{:,.2f}%'}
            results = results.style.format(format_dict)

            # Set title strings to center
            results = results.set_table_styles([dict(selector = 'th', props=[('text-align', 'center')])])
            # Set columns to display on the left
            results = results.set_properties(subset=['Column', 'Type'], **{'text-align': 'left'})
            return results
        

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
        results = group_data.sort_values([col_name] + cols_to_report, ascending=False).reset_index(drop=True)

        # Include percents
        total = results[col_name].sum()
        results['%'] = (results[col_name] / total) * 100

        # Keep track of cumulative counts or totals as well as their relative percent
        results[f'Cumulative {col_name}'] = results[col_name].cumsum()
        results['Cumulative %'] = (results[f'Cumulative {col_name}'] / total) * 100

        # Style the output
        format_dict = {'%': '{:.2f}%', 'Cumulative %': '{:.2f}%', 'Count': '{0:,.0f}', f'{col_name}': '{0:,.0f}', f'Cumulative {col_name}': '{0:,.0f}'}
        results = results.style.format(format_dict)
    
        # Set title strings to center
        results = results.set_table_styles([dict(selector = 'th', props=[('text-align', 'center')])])
        # Set columns to display on the left
        results = results.set_properties(subset=cols_to_report, **{'text-align': 'left'})
        return results

