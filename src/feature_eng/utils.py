
import pandas as pd

def get_datetime(df, col_name, dayfirst=False, yearfirst=False, use_format=False):
    """ return a series with date_time format
    Feature list: [year, month, day, hour, minute, second, date, time, dayofyear, weekofyear, week, dayofweek, weekday, quarter, freq]
    """
    if use_format:
        format = "%d/%m/%Y"
    return pd.to_datetime(df[col_name], dayfirst=dayfirst, yearfirst=yearfirst, format=format)



def get_mapping(df, col_name):
    cat_codes = df[col_name].astype('category')
    
    class_mapping = {}
    i = 0
    for col in cat_codes.cat.categories:
        class_mapping[col] = i
        i += 1
    
    class_mapping_reverse = {}
    for key, value in class_mapping.items():
        class_mapping_reverse[value] = key

    return class_mapping, class_mapping_reverse