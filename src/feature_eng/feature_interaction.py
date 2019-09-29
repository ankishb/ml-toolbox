count = 0
num_cat_feat = pd.DataFrame()
for i, col1 in enumerate(inter_cols):
    for j, col2 in enumerate(cat_cols):
        if count == 1000: break
        try:
            col_name = col1+"_/_"+col2
            num_cat_feat[col_name] = train[col1]/train[col2]
            count += 1
        except:
            pass


add_feat_save = num_cat_feat.columns
gc.collect()




imp_alls = pd.DataFrame(data=list(num_cat_feat.columns), columns=['feature'])
imp_alls['imp'] = 0
num_cat_feat = reduce_mem_usage_without_print(num_cat_feat)
# imp_alls






alls_score = []
for i in range(500):
    use_cols = random.sample(list(num_cat_feat.columns),300)
    X_tr, X_ts, y_tr, y_ts = train_test_split(num_cat_feat[use_cols], target, stratify=target, test_size=0.3, random_state=1234)

    cat = train_cat_model(X_tr, y_tr, X_ts, y_ts)
    if cat[0] < 0.868:
    	imp_alls = imp_alls.merge(cat[1].tail(30), on='feature', how='left')
    else:
    	imp_alls = imp_alls.merge(cat[1].tail(100), on='feature', how='left')
    alls_score.append(cat[0])
    
    gc.collect()


define_cols = ['feature', 'zeros']+["itr_"+str(i) for i in range(500)]
imp_alls.columns = define_cols


imp_alls.to_csv("predictions/num_cat_inter_imp.csv", index=None)
np.save("predictions/num_cat_inter_score.npy", alls_score)



del num_cat_feat, imp_alls
gc.collect()



plt.figure(figsize=(20,6))
plt.plot(alls_score, 'o')








##########################################################################################################
file_name = "predictions/good_cols.txt"
def store_cols(cols, file_name):
    with open(file_name, 'a+') as f:
        for col in cols:
            f.write(col)
            f.write("\t")
        f.write("\n")
        
        
        
def get_all_interaction(inter_cols, cat_cols):
    all_inter_cols = []
    # simple *,+,- feature
    for i, col1 in enumerate(inter_cols):
        for j, col2 in enumerate(inter_cols):
            mul_ = col1+"_*_"+col2
            min_ = col1+"_-_"+col2
            add_ = col1+"_+_"+col2
            all_inter_cols.append(mul_)
            all_inter_cols.append(min_)
            all_inter_cols.append(add_)
            
    # complicated feature
    for i, col1 in enumerate(inter_cols):
        for j, col2 in enumerate(inter_cols):
            com1 = col1+"_*_"+col1+"_+_"+col2
            com2 = col2+"_*_"+col2+"_+_"+col1
            all_inter_cols.append(col1)
            all_inter_cols.append(col2)
            
    # num and cat interaction
    for i, col1 in enumerate(inter_cols):
        for j, col2 in enumerate(cat_cols):
            div_ = col1+"_/_"+col2
            all_inter_cols.append(div_)

    return all_inter_cols


def get_feature(cols):
    feature = pd.DataFrame()
    for col in cols:
        c = col.split("_")
        try:
            if len(c) == 3:
                col_value = train[c[0]] op[c[1]] train[c[2]]
            elif len(c) == 5:
                col_value = train[c[0]] op[c[1]] train[c[2]] op[c[3]] train[c[4]]
            feature[col] = col_value
        except:
            pass
    return feature
                

baseline = 0.87
alls_score = []
all_inter_cols = get_all_interaction(inter_cols, cat_cols)

imp_alls = pd.DataFrame(
    data=all_inter_cols, 
    columns=['feature']
)
imp_alls['imp'] = 0

for i in range(500):
    # get features list and prepare table
    use_cols = random.sample(list(all_inter_cols),300)
    feature = get_feature(use_cols)
    
    # train model
    X_tr, X_ts, y_tr, y_ts = train_test_split(
        feature, target, 
        stratify=target, 
        test_size=0.3, 
        random_state=1234
    )
    cat = train_cat_model(X_tr, y_tr, X_ts, y_ts)
    
    # if score is more than baseline, store in memory
    if i%5 == 0: print(i, end=" ")
    if cat[0] >= baseline:
        store_cols(use_cols)
        imp_alls = imp_alls.merge(cat[1], on='feature', how='left')
        alls_score.append(cat[0])
    
    gc.collect()

##########################################################################################################