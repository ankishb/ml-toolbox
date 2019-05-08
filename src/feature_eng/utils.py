


import pandas as pd

def get_datetime(df, col_name, dayfirst=False, yearfirst=False, use_format=False):
    """ return a series with date_time format
    Feature list: [year, month, day, hour, minute, second, date, time, dayofyear, weekofyear, week, dayofweek, weekday, quarter, freq]
    """
    if use_format:
        format = "%d/%m/%Y"
    return pd.to_datetime(df[col_name], dayfirst=dayfirst, yearfirst=yearfirst, format=format)
