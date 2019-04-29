
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline

def boxplot_it(df, col_x, col_y):
    """
    Args:
        df      : data-frame
        col_x   : columnn x for boxplot
        col_y   : columnn y for boxplot
    return: 
        distplot with flag[0/1]
    """
    plt.figure(figsize=(16,4))
    sns.boxplot(y=col_y, x=col_x, data=df)


def distplot_it(df, flag_col, col):
    """
    Args:
        df      : data-frame
        col     : columnn for distplot
        flag_col: flag to represent train or test
    return: 
        distplot with flag[0/1]
    """
    plt.figure(figsize=(16,4))
    sns.distplot(df[df[flag_col] == 1][col], hist=False, label='train')
    sns.distplot(df[df[flag_col] == 0][col], hist=False, label='test')
    _ = plt.xticks(rotation='vertical')
    
distplot_it(train_test, 'train_flag', 'asset_cost')



def countplot_it(df, flag_col, col):
    """
    Args:
        df      : data-frame
        col     : columnn for distplot
        flag_col: flag to represent train or test
    return: 
        distplot with flag[0/1]
    """
	plt.figure(figsize=(16,4))
	sns.countplot(x=col, hue=flag_col, data=train_test)
	_ = plt.xticks(rotation='vertical')

    










You can also choose the plot kind by using the DataFrame.plot.kind methods instead of providing the kind keyword argument.

kind :

    'line' : line plot (default)
    'bar' : vertical bar plot
    'barh' : horizontal bar plot
    'hist' : histogram
    'box' : boxplot
    'kde' : Kernel Density Estimation plot
    'density' : same as 'kde'
    'area' : area plot
    'pie' : pie plot
    'scatter' : scatter plot
    'hexbin' : hexbin plot ###### Go to top

# create a scatter plot of columns 'A' and 'C', with changing color (c) and size (s) based on column 'B'
df.plot.scatter('A', 'C', c='B', s=df['B'], colormap='viridis')












################################################################################################################################

def plot_stats(feature,label_rotation=False,horizontal_layout=True):
    """
    plot_stats('NAME_CONTRACT_TYPE')
    """
    temp = application_train[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = application_train[[feature, 'TARGET']].groupby([feature],as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12,14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x = feature, y="Number of contracts",data=df1)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    s = sns.barplot(ax=ax2, x = feature, y='TARGET', order=cat_perc[feature], data=cat_perc)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show();

def plot_distribution(var):
    
    i = 0
    t1 = application_train.loc[application_train['TARGET'] != 0]
    t0 = application_train.loc[application_train['TARGET'] == 0]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(2,2,figsize=(12,12))

    for feature in var:
        i += 1
        plt.subplot(2,2,i)
        sns.kdeplot(t1[feature], bw=0.5,label="TARGET = 1")
        sns.kdeplot(t0[feature], bw=0.5,label="TARGET = 0")
        plt.ylabel('Density plot', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show();









# Plot distribution of one feature
def plot_distribution(feature,color):
    plt.figure(figsize=(10,6))
    plt.title("Distribution of %s" % feature)
    sns.distplot(application_train[feature].dropna(),color=color, kde=True,bins=100)
    plt.show()   

# Plot distribution of multiple features, with TARGET = 1/0 on the same graph
def plot_distribution_comp(var,nrow=2):
    
    i = 0
    t1 = application_train.loc[application_train['TARGET'] != 0]
    t0 = application_train.loc[application_train['TARGET'] == 0]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(nrow,2,figsize=(12,6*nrow))

    for feature in var:
        i += 1
        plt.subplot(nrow,2,i)
        sns.kdeplot(t1[feature], bw=0.5,label="TARGET = 1")
        sns.kdeplot(t0[feature], bw=0.5,label="TARGET = 0")
        plt.ylabel('Density plot', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show();

plot_distribution('AMT_INCOME_TOTAL','green')

green, blue, tomato, brown, red

var = ['AMT_ANNUITY','AMT_GOODS_PRICE','DAYS_EMPLOYED', 'DAYS_REGISTRATION','DAYS_BIRTH','DAYS_ID_PUBLISH']
plot_distribution_comp(var,nrow=3)

################################################################################################################################






################################################################################################################################
https://www.kaggle.com/artgor/exploration-of-data-step-by-step


# to add percentage on top of the bar plot
plt.figure(figsize=(14, 6));
g = sns.countplot(x='AdoptionSpeed', data=all_data.loc[all_data['dataset_type'] == 'train'])
plt.title('Adoption speed classes rates');
ax=g.axes
for p in ax.patches:
     ax.annotate(f"{p.get_height() * 100 / train.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),
         textcoords='offset points')  




main_count = train['AdoptionSpeed'].value_counts(normalize=True).sort_index()
def prepare_plot_dict(df, col, main_count):
    """
    Preparing dictionary with data for plotting.
    
    I want to show how much higher/lower are the rates of Adoption speed for the current column comparing to base values (as described higher),
    At first I calculate base rates, then for each category in the column I calculate rates of Adoption speed and find difference with the base rates.
    
    """
    main_count = dict(main_count)
    plot_dict = {}
    for i in df[col].unique():
        val_count = dict(df.loc[df[col] == i, 'AdoptionSpeed'].value_counts().sort_index())

        for k, v in main_count.items():
            if k in val_count:
                plot_dict[val_count[k]] = ((val_count[k] / sum(val_count.values())) / main_count[k]) * 100 - 100
            else:
                plot_dict[0] = 0

    return plot_dict

def make_count_plot(df, x, hue='AdoptionSpeed', title='', main_count=main_count):
    """
    Plotting countplot with correct annotations.
    """
    g = sns.countplot(x=x, data=df, hue=hue);
    plt.title(f'AdoptionSpeed {title}');
    ax = g.axes

    plot_dict = prepare_plot_dict(df, x, main_count)

    for p in ax.patches:
        h = p.get_height() if str(p.get_height()) != 'nan' else 0
        text = f"{plot_dict[h]:.0f}%" if plot_dict[h] < 0 else f"+{plot_dict[h]:.0f}%"
        ax.annotate(text, (p.get_x() + p.get_width() / 2., h),
             ha='center', va='center', fontsize=11, color='green' if plot_dict[h] > 0 else 'red', rotation=0, xytext=(0, 10),
             textcoords='offset points')  

plt.figure(figsize=(18, 8));
make_count_plot(df=all_data.loc[all_data['dataset_type'] == 'train'], x='Type', title='by pet Type')

################################################################################################################################


print('Distributions of first 28 columns')
plt.figure(figsize=(26, 24))
for i, col in enumerate(list(train.columns)[2:30]):
    plt.subplot(7, 4, i + 1)
    plt.hist(train[col])
    plt.title(col)













################################################################################################################################
https://www.kaggle.com/gpreda/santander-eda-and-prediction

def plot_feature_scatter(df1, df2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(4,4,figsize=(14,14))

    for feature in features:
        i += 1
        plt.subplot(4,4,i)
        plt.scatter(df1[feature], df2[feature], marker='+')
        plt.xlabel(feature, fontsize=9)
    plt.show();


features = ['var_0', 'var_1','var_2','var_3', 'var_4', 'var_5', 'var_6', 'var_7', 
           'var_8', 'var_9', 'var_10','var_11','var_12', 'var_13', 'var_14', 'var_15', 
           ]
plot_feature_scatter(train_df[::20],test_df[::20], features)




def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))

    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.distplot(df1[feature], hist=False,label=label1)
        sns.distplot(df2[feature], hist=False,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
features = train_df.columns.values[2:102]
plot_feature_distribution(t0, t1, '0', '1', features)

################################################################################################################################





################################################################################################################################
https://www.kaggle.com/gpreda/donorschoose-extensive-eda



Let's show the time variation of the number of donation. We will eliminate the donations in 2018, since the values are only until April.

We represent the sum of donations, the average value of a donation, the min and max values of donations, calculated per Year, Month, Day, Weekday and Hour.

def plot_time_variation(feature):
    tmp = donations_donors.groupby(feature)['Donation Amount'].sum()
    tmp = tmp[~tmp.index.isin([2018])] 
    df1 = pd.DataFrame({feature: tmp.index,'Total sum of donations': tmp.values})
    
    tmp = donations_donors.groupby(feature)['Donation Amount'].mean()
    tmp = tmp[~tmp.index.isin([2018])] 
    df2 = pd.DataFrame({feature: tmp.index,'Mean value of donations': tmp.values})
    
    tmp = donations_donors.groupby(feature)['Donation Amount'].min()
    tmp = tmp[~tmp.index.isin([2018])] 
    df3 = pd.DataFrame({feature: tmp.index,'Min value of donations': tmp.values})
    
    tmp = donations_donors.groupby(feature)['Donation Amount'].max()
    tmp = tmp[~tmp.index.isin([2018])] 
    df4 = pd.DataFrame({feature: tmp.index,'Max value of donations': tmp.values})
    
    fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(14,4))
    s = sns.barplot(ax = ax1, x = feature, y="Total sum of donations",data=df1)
    s = sns.barplot(ax = ax2, x = feature, y="Mean value of donations",data=df2)
    plt.show();
    
    fig, (ax3, ax4) = plt.subplots(ncols=2,figsize=(14,4))
    s = sns.barplot(ax = ax3, x = feature, y="Min value of donations",data=df3)
    s = sns.barplot(ax = ax4, x = feature, y="Max value of donations",data=df4)
    plt.show();

def boxplot_time_variation(feature, width=16):
    fig, ax1 = plt.subplots(ncols=1, figsize=(width,6))
    s = sns.boxplot(ax = ax1, x=feature, y="Donation Amount", hue="Donor Is Teacher",
                data=donations_donors, palette="PRGn",showfliers=False)
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show();

plot_time_variation('Year')
boxplot_time_variation('Year',8):




Amount of project cost per state

tmp = projects_schools.groupby('School State')['Project Cost'].sum()
df1 = pd.DataFrame({'School State': tmp.index,'Total Projects Cost': tmp.values})
df1.sort_values(by='Total Projects Cost',ascending=False, inplace=True)
fig, (ax1) = plt.subplots(ncols=1, figsize=(12,6))
s = sns.barplot(ax = ax1, x = 'School State', y="Total Projects Cost",data=df1)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show();





tmp = np.log(donations_donors_projects_schools.groupby(['Donor State', 'School State'])['Donation Amount'].sum())
df1 = tmp.reset_index()
matrix = df1.pivot('Donor State', 'School State','Donation Amount')

fig, (ax1) = plt.subplots(ncols=1, figsize=(16,16))
sns.heatmap(matrix, 
        xticklabels=matrix.index,
        yticklabels=matrix.columns,ax=ax1,linewidths=.1,cmap="YlGnBu")
plt.title("Heatmap with log(Donation Amount) per donor state and school state", fontsize=14)
plt.show()





################################################################################################################################





################################################################################################################################
https://www.kaggle.com/gpreda/stack-overflow-2018-developer-survey-extensive-eda

# Let's represent the top 50 respondents countries.

def plot_stats(feature, text, size=2):
    temp = data_df[feature].dropna().value_counts().head(50)
    df1 = pd.DataFrame({feature: temp.index,'Number of respondents': temp.values})
    plt.figure(figsize = (8*size,4))
    plt.title(text,fontsize=14)
    s = sns.barplot(x=feature,y='Number of respondents',data=df1)
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show()   
    
plot_stats('Country','Countries')
plot_stats('FormalEducation','Formal Education',1)






We will define a general plot heatmap function which will receive a keyword, extract from the schema all the entries containing that keyword in the Column feature entries, count the number of occurences, parse all, build the statistics for all occurences and then plot as a histogram. This will be reused for all clusters of questions requested to select a priority for each criteria in the cluster.

def plot_heatmap(feature, text, color="Blues"):
    tmp = schema_df[schema_df['Column'].str.contains(feature)]
    features = list(tmp['Column'])
    dim = len(features)
    temp1 = pd.DataFrame(np.random.randint(low=0, high=10,size=(1+dim, dim)),columns=features)
    for feature in features:
        temp1[feature] = data_df[feature].dropna().value_counts()

    fig, (ax1) = plt.subplots(ncols=1, figsize=(16,4))
    sns.heatmap(temp1[1::], 
        xticklabels=temp1.columns,
        yticklabels=temp1.index[1::],annot=True,ax=ax1,linewidths=.1,cmap=color)
    plt.title(text, fontsize=14)
    plt.show()

plot_heatmap('AssessJob','Heatmap with Assess Job priorities 1-10 (respondants count)')




################################################################################################################################







################################################################################################################################
# vertical bar plot
students_locations = students.students_location.value_counts().sort_values(ascending=True).tail(20)
students_locations.plot.barh(figsize=(10, 8), color='b', width=1)
plt.title("Number of students by location", fontsize=20)
plt.xlabel('Number of students', fontsize=12)
plt.show()

################################################################################################################################








################################################################################################################################


################################################################################################################################






################################################################################################################################


################################################################################################################################