This notebook shows how I reduce the size of the properties dataset by selecting smaller datatypes.

I noticed the size of the properties dataset is pretty big for a lower/mid-range laptop so I made a script to make the dataset smaller without losing information.

This notebook uses the following approach:

    Iterate over every column
    Determine if the column is numeric
    Determine if the column can be represented by an integer
    Find the min and the max value
    Determine and apply the smallest datatype that can fit the range of values

This reduces the dataset from approx. 1.3 GB to 466 MB
1 | load packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

2 | Function for reducing memory usage of a pandas dataframe

def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            before = props[col].dtype
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            after = props[col].dtype
            print(before, "==>", after)
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist

3 | Load Data

props = pd.read_csv(r"../input/properties_2016.csv")  #The properties dataset

#train = pd.read_csv(r"../input/train_2016_v2.csv")   # The parcelid's with their outcomes
#samp = pd.read_csv(r"../input/sample_submission.csv")  #The parcelid's for the testset

/opt/conda/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2717: DtypeWarning: Columns (22,32,34,49,55) have mixed types. Specify dtype option on import or set low_memory=False.
  interactivity=interactivity, compiler=compiler, result=result)

4 | Run function

props, NAlist = reduce_mem_usage(props)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)

Memory usage of properties dataframe is : 1320.9731750488281  MB
******************************
Column:  parcelid
dtype before:  int64
dtype after:  uint32
******************************
******************************
Column:  airconditioningtypeid
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  architecturalstyletypeid
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  basementsqft
dtype before:  float64
dtype after:  uint16
******************************
******************************
Column:  bathroomcnt
dtype before:  float64
dtype after:  float32
******************************
******************************
Column:  bedroomcnt
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  buildingclasstypeid
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  buildingqualitytypeid
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  calculatedbathnbr
dtype before:  float64
dtype after:  float32
******************************
******************************
Column:  decktypeid
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  finishedfloor1squarefeet
dtype before:  float64
dtype after:  uint16
******************************
******************************
Column:  calculatedfinishedsquarefeet
dtype before:  float64
dtype after:  uint32
******************************
******************************
Column:  finishedsquarefeet12
dtype before:  float64
dtype after:  uint32
******************************
******************************
Column:  finishedsquarefeet13
dtype before:  float64
dtype after:  uint16
******************************
******************************
Column:  finishedsquarefeet15
dtype before:  float64
dtype after:  uint32
******************************
******************************
Column:  finishedsquarefeet50
dtype before:  float64
dtype after:  uint16
******************************
******************************
Column:  finishedsquarefeet6
dtype before:  float64
dtype after:  uint32
******************************
******************************
Column:  fips
dtype before:  float64
dtype after:  uint16
******************************
******************************
Column:  fireplacecnt
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  fullbathcnt
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  garagecarcnt
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  garagetotalsqft
dtype before:  float64
dtype after:  uint16
******************************
******************************
Column:  heatingorsystemtypeid
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  latitude
dtype before:  float64
dtype after:  uint32
******************************
******************************
Column:  longitude
dtype before:  float64
dtype after:  int32
******************************
******************************
Column:  lotsizesquarefeet
dtype before:  float64
dtype after:  float32
******************************
******************************
Column:  poolcnt
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  poolsizesum
dtype before:  float64
dtype after:  uint16
******************************
******************************
Column:  pooltypeid10
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  pooltypeid2
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  pooltypeid7
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  propertylandusetypeid
dtype before:  float64
dtype after:  uint16
******************************
******************************
Column:  rawcensustractandblock
dtype before:  float64
dtype after:  float32
******************************
******************************
Column:  regionidcity
dtype before:  float64
dtype after:  uint32
******************************
******************************
Column:  regionidcounty
dtype before:  float64
dtype after:  uint16
******************************
******************************
Column:  regionidneighborhood
dtype before:  float64
dtype after:  uint32
******************************
******************************
Column:  regionidzip
dtype before:  float64
dtype after:  uint32
******************************
******************************
Column:  roomcnt
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  storytypeid
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  threequarterbathnbr
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  typeconstructiontypeid
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  unitcnt
dtype before:  float64
dtype after:  uint16
******************************
******************************
Column:  yardbuildingsqft17
dtype before:  float64
dtype after:  uint16
******************************
******************************
Column:  yardbuildingsqft26
dtype before:  float64
dtype after:  uint16
******************************
******************************
Column:  yearbuilt
dtype before:  float64
dtype after:  uint16
******************************
******************************
Column:  numberofstories
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  structuretaxvaluedollarcnt
dtype before:  float64
dtype after:  uint32
******************************
******************************
Column:  taxvaluedollarcnt
dtype before:  float64
dtype after:  uint32
******************************
******************************
Column:  assessmentyear
dtype before:  float64
dtype after:  uint16
******************************
******************************
Column:  landtaxvaluedollarcnt
dtype before:  float64
dtype after:  uint32
******************************
******************************
Column:  taxamount
dtype before:  float64
dtype after:  float32
******************************
******************************
Column:  taxdelinquencyyear
dtype before:  float64
dtype after:  uint8
******************************
******************************
Column:  censustractandblock
dtype before:  float64
dtype after:  int64
******************************
___MEMORY USAGE AFTER COMPLETION:___
Memory usage is:  478.28343963623047  MB
This is  36.20690023614986 % of the initial size
_________________
