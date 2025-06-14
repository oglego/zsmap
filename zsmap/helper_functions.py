import pandas as pd
import numpy as np

def compute_zscore(df, columns):
    """
    Compute z-scores for specified columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to compute z-scores for.

    Returns:
        pd.DataFrame: A copy of the DataFrame with z-scores in the specified columns.
    """
    df_zscore = df.copy()
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df_zscore[f'{col}_zscore'] = (df[col] - mean) / std
    return df_zscore

def square_columns(df, columns):
    """
    Square each value in the specified columns of a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to square.

    Returns:
        pd.DataFrame: A copy of the DataFrame with squared values in the specified columns.
    """
    df_squared = df.copy()
    for col in columns:
        df_squared[f'{col}_square'] = df[col] ** 2
    return df_squared

def sum_columns(df, columns_to_sum, new_column_name):
    """
    Add specified columns together and store the result in a new column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns_to_sum (list): List of column names to sum.
        new_column_name (str): Name of the new column for the sum.

    Returns:
        pd.DataFrame: A copy of the DataFrame with the new summed column.
    """
    df_sum = df.copy()
    df_sum[new_column_name] = df[columns_to_sum].sum(axis=1)
    return df_sum

def sqrt_column(df, column, new_column=None):
    """
    Take the square root of values in a specified column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): Name of the column to apply square root to.
        new_column (str, optional): If provided, stores result in this new column.
                                    Otherwise, replaces the original column.

    Returns:
        pd.DataFrame: A copy of the DataFrame with square root applied.
    """
    df_sqrt = df.copy()
    if new_column:
        df_sqrt[new_column] = np.sqrt(df[column])
    else:
        df_sqrt[column] = np.sqrt(df[column])
    return df_sqrt

def ln_column(df, column, new_column=None):
    """
    Compute the natural logarithm (ln) of the values in a specified column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): Name of the column to apply ln to.
        new_column (str, optional): If provided, stores result in this new column.
                                    Otherwise, replaces the original column.

    Returns:
        pd.DataFrame: A copy of the DataFrame with ln applied.
    """
    df_ln = df.copy()
    if new_column:
        df_ln[new_column] = np.log(df[column])
    else:
        df_ln[column] = np.log(df[column])
    return df_ln

def find_outliers_percentiles(df, column):
    """
    Identify outliers as values below the 5th percentile or above the 95th percentile.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to analyze.

    Returns:
        pd.DataFrame: DataFrame with a new boolean column 'is_outlier'.
    """
    lower_bound = np.percentile(df[column], 5)
    upper_bound = np.percentile(df[column], 95)

    df_copy = df.copy()
    df_copy['comp_is_outlier'] = (df_copy[column] > lower_bound) 
    return df_copy

import pandas as pd
import matplotlib.pyplot as plt

def plot_histogram(df, column, title, bins=10):
    """
    Plot a histogram of a specified column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to plot.
        bins (int): The number of bins to divide the data into (default is 10).
    """
    plt.figure(figsize=(8, 6))
    plt.hist(df[column], bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.grid(True)
    plt.show()

import pandas as pd
import numpy as np

def add_percentile_column(df, column):
    """
    Compute the percentile for each value in a column and create a new column with percentiles.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to calculate percentiles for.

    Returns:
        pd.DataFrame: DataFrame with the percentile column added.
    """
    percentiles = np.percentile(df[column], np.linspace(0, 100, 101))
    df_copy = df.copy()
    df_copy['percentile'] = df[column].apply(lambda x: np.searchsorted(percentiles, x) - 1)
    
    return df_copy

def add_percentile_groups(df, column):
    """
    Group percentiles into categories such as 'Low', 'Medium', and 'High'.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to calculate percentiles for.

    Returns:
        pd.DataFrame: DataFrame with the group column added based on percentiles.
    """
    df_with_percentiles = add_percentile_column(df, column)

    # Define percentile thresholds for groups
    conditions = [
        (df_with_percentiles['percentile'] <= 33),
        (df_with_percentiles['percentile'] > 33) & (df_with_percentiles['percentile'] <= 66),
        (df_with_percentiles['percentile'] > 66) & (df_with_percentiles['percentile'] <= 95),
        (df_with_percentiles['percentile'] > 95)
    ]
    choices = ['Low', 'Medium', 'High', 'Higher']

    df_with_percentiles['percentile_group'] = np.select(conditions, choices, default='Unknown')
    
    return df_with_percentiles

import matplotlib.pyplot as plt

def plot_colored_scatter(df, x_column, y_column, group_column):
    """
    Plot a scatter plot of x and y coordinates, colored by a group column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        x_column (str): The name of the x-axis column.
        y_column (str): The name of the y-axis column.
        group_column (str): The column to use for coloring points.
    """
    plt.figure(figsize=(8, 6))
    
    # Define colors for each group
    groups = df[group_column].unique()
    colors = plt.cm.get_cmap('Set1', len(groups))
    
    # Create a scatter plot for each group
    for i, group in enumerate(groups):
        group_data = df[df[group_column] == group]
        plt.scatter(group_data[x_column], group_data[y_column], label=group, 
                    color=colors(i), alpha=0.7)
    
    plt.grid(True)
    plt.title("Cluster Analysis")
    plt.show()


