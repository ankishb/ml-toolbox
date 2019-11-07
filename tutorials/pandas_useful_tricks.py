
## Pandas Aggregation:
# Group the data frame by month and item and extract a number of stats from each group
data.groupby(['month', 'item']).agg(aggregation)
aggregation = {
        # find the min, max, and sum of the duration column
        'duration': [min, max, sum],
         # find the number of network type entries
        'network_type': "count",
        # min, first, and number of unique dates per group
        'date': [min, 'first', 'nunique']
    }


# Another way to aggregate with new columns names:

# Group the data frame by month and item and extract a number of stats from each group
data.groupby(['month', 'item']).agg(aggregation)
aggregation = {
        # find the min, max, and sum of the duration column
        'duration': {
            'total_duration'  : 'sum',
            'average_duration': 'mean',
            'num_calls'       : 'count',
        },
         # find the number of network type entries
        'network_type': {
            'count_networks' : 'count',
            'num_days'       : lambda x: max(x)-min(x),
        },
        # min, first, and number of unique dates per group
        'date': [min, 'first', 'nunique']
    }


# Load the required packages
import time
import psutil
import numpy as np
import pandas as pd
import multiprocessing as mp

# Check the number of cores and memory usage
num_cores = mp.cpu_count()
print("This kernel has ",num_cores,"cores and you can find the information regarding the memory usage:",psutil.virtual_memory())


# Writing as a function
def process_user_log(chunk):
    grouped_object = chunk.groupby(chunk.index,sort = False) # not sorting results in a minor speedup
    func = {
      'date'   : ['min','max','count'],
      'num_25' : ['sum'],
      'num_50' : ['sum'], 
      'num_75' : ['sum'],
      'num_unq': ['sum'],
      'totSec' : ['sum']
    }
    answer = grouped_object.agg(func)
    return answer

# Number of rows for each chunk
size = 4e7 # 40 Millions
reader = pd.read_csv('user_logs.csv', chunksize = size, index_col = ['msno'])
start_time = time.time()

for i in range(10):
    user_log_chunk = next(reader)
    if(i==0):
        result = process_user_log(user_log_chunk)
        print("Number of rows ",result.shape[0])
        print("Loop ",i,"took %s seconds" % (time.time() - start_time))
    else:
        result = result.append(process_user_log(user_log_chunk))
        print("Number of rows ",result.shape[0])
        print("Loop ",i,"took %s seconds" % (time.time() - start_time))
    del(user_log_chunk)    

# Unique users vs Number of rows after the first computation    
print("size of result:", len(result))
check = result.index.unique()
print("unique user in result:", len(check))

result.columns = ['_'.join(col).strip() for col in result.columns.values]
