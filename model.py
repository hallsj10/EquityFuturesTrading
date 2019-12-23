
import pandas as pd
import numpy as np

# Read raw price data
raw = pd.read_excel (r'I:\Fixed Income\Equities\SPX_v_EAFE\SPX_EAFE.xlsx',
                     sheet_name='Clean', index_col=0) # Make date index.

desc = raw.describe().transpose()
min_rows = int(desc['count'].min())
clean_px = raw.iloc[-min_rows:]

def returns(df, end_dt, beg_dt):
    """Calculates percentage returns of all series in a data frame"""
    return (df.shift(periods=end_dt) / df.shift(periods=beg_dt)) - 1

# Data Preprocessing

# RESPONSE VARIABLES
response_vars = {'ES','VG','NH','Z'}    
response = clean_px[response_vars]
frcst_pd = 21 # Number of days in forecast period.  
frcst_rets = returns(response, 0, -frcst_pd) # Shift back for fwd returns.
select_Y = frcst_rets.rank(axis=1, ascending = False)

# PREDICTOR VARIABLES
# One-month reversal.
x1 = returns(clean_px, 0, 21) # 21 days in a month.
x2 = x1.add_prefix('1MR_')
# Momentum factor.
x3 = returns(clean_px, 21, 252)
x4 = x3.add_prefix('MOMO_')
# Merge predictors
select_X = x2.merge(x4,left_index=True, right_index=True)

# Merge response and predictors
select_dataset = select_Y.merge(select_X,left_index=True, right_index=True)
select_dataset = select_dataset.dropna()
stats = select_dataset.describe()

# TODO: Day forward-chaining for model selection
# Want to test countless algo/hyperparameter combinations and identify the best
# combination (model) in various out-of-sample tests




