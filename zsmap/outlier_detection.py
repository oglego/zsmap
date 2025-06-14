from submit.helper_functions import *
from submit.synthetic_data import *

"""
Generate synthetic data
"""
df = generate_data()

def zsmap(df):
    cmp_zcr = compute_zscore(df, ['latitude', 'longitude'])
    sqr_col = square_columns(cmp_zcr, ['latitude_zscore', 'longitude_zscore'])
    sum_sqr = sum_columns(sqr_col, ['latitude_zscore_square', 'longitude_zscore_square'], 'sum_of_squares')
    inv_ops = sqrt_column(sum_sqr, ['sum_of_squares'], 'inv_operation')
    ln_inv = ln_column(inv_ops, 'inv_operation', 'ln_inv')
    cmp_prc = add_percentile_column(ln_inv, 'ln_inv')
    prc_grp = add_percentile_groups(cmp_prc, 'percentile')
    return prc_grp

result_df = zsmap(df)

plot_histogram(result_df, 'inv_operation', 'Inverse Operation')
plot_histogram(result_df, 'ln_inv', 'Natural Log')
plot_colored_scatter(result_df, 'longitude', 'latitude', 'percentile_group')








