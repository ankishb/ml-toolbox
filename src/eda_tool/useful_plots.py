
def scatter_plot(df, cols, log=False)
	pos = df[df['target'] == 1]
	neg = df[df['target'] == 0]
	   
	fig, ax = plt.subplots(2,5,figsize=(20,8))
	axes = ax.flatten()
	for i, col in enumerate(cols):
	    try:
	    	if log:
		        axes[i].scatter(range(len(pos)), np.log(1+pos[col]))#, 'r', label='1')
		        axes[i].scatter(range(len(neg)), np.log(1+neg[col]))#, 'b', label='0')
		    else:
		        axes[i].scatter(range(len(pos)), pos[col])#, 'r', label='1')
		        axes[i].scatter(range(len(neg)), neg[col])#, 'b', label='0')
	        axes[i].set_title(col)
	    except:
	        pass
	plt.show()


def plot_distribution(df, cols=None, plot_type='dist'):
    """
    plot_type : ['dist','kde', 'count']
    Note: kdeplots are heavy, as comparison to distplot
    example:
		plot_distribution(cols)
    """
    i = 0
    t1 = df.loc[target != 0]
    t0 = df.loc[target == 0]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(2,5,figsize=(22,8))

    if cols is None:
    	feature = df.columns[:10]
    for feature in cols:
        i += 1
        plt.subplot(2,5,i)
        if plot_type == 'dist':
	        sns.distplot(t1[feature].dropna(),label="cls: 1")
	        sns.distplot(t0[feature].dropna(),label="cls: 0")
	    	plt.ylabel('Density plot', fontsize=12)
	    elif plot_type == 'kde'
	        sns.kdeplot(t1[feature], bw=0.5,label="cls: 1")
	        sns.kdeplot(t0[feature], bw=0.5,label="cls: 0")
	        plt.ylabel('Density plot', fontsize=12)
	    else if plot_type == 'count':
	        sns.countplot(t1[feature],label="cls: 1")
        	sns.countplot(t0[feature],label="cls: 0")
        	plt.ylabel('Count plot', fontsize=12)
        else:
        	print("plot_type should be from ['dist','kde', 'count']")
        	break
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()