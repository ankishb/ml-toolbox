
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


def dump_data(data, file_name='punctuation.txt'):
	with open(file_name, 'w') as f:
	    json.dump(extra_punct, f)
	print("dumped all data in ", file_name)

	

def load_data(file_name = 'contraction_mapping.txt'):
	with open(file_name) as f:
    	contraction_mapping = json.loads(f.read())
    return contraction_mapping


def factorize_cat(table, cat_cols=None):
	"""
	exp: df = factorize_cat(df, cat_cols)
	"""
    print("=== Factorize Done ====")
    for col in cat_cols:
        table[col], _ = pd.factorize(table[col])
    return table

def fillna_and_create_new_cols():
	new_num_cols = []
	for col in num_cols:
	    try:
	        col_name = 'NULL_'+col
	        
	        min_tr = train1[col].min()
	        var_tr = train1[col].var()
	        
	        min_ts = test1[col].min()
	        var_ts = test1[col].var()
	        
	        train1[col] = train1[col].fillna(min_tr - var_tr/10)
	        train1[col_name] = np.where(train[col].isnull(), 1, 0)
	        
	        test1[col] = test1[col].fillna(min_ts - var_ts/10)
	        test1[col_name] = np.where(test[col].isnull(), 1, 0)
	        
	        # if get passed through all steps
	        new_num_cols.append(col_name)
	        
	    except:
	        print(col)
	return train1, test1, new_num_cols


	

import random, math

def grouping_cols(df, cat_percentage = 0.05, checking_itr = 10):
	""" grouping unknown variable using
		1. counting unique value
		2. if variable is integer or object

	example: cat_cols, num_cols = grouping_cols(train)
	"""
    cc, nc = [], []
    max_ = 0
    amount = int(df.shape[0]*cat_percentage)
    print(amount, "/", df.shape[0]," Used to differentiate num feature from cat feature")
    for col in df.columns:
        uni = df[col].unique().shape[0]
#         print(uni)
        max_ = max(max_, uni)
        if(uni <= amount):
            cc.append(col)
        else:
            nc.append(col)
        
    print("-----Filtered result after Ist stage-----")
    print("total cat cols: {}, total num cols: {}\n".format(len(cc), len(nc)))
    
    true_cat = []
    true_num = []
    for col in nc+cc:
        num = False
        if(df[col].dtype == 'object'):
            true_cat.append(col)
            continue
        for i in range(checking_itr):
            sample = np.random.choice(df[col].unique())
            if math.isnan(sample): continue
            if int(sample) != sample:
                num = True
        if num is True:
            true_num.append(col)
        else:
            true_cat.append(col)

    print("-----Filtered result after 2nd stage-----")
    print("total cat cols: {}, total num cols: {}".format(len(true_cat), len(true_num)))
    return cc, nc
