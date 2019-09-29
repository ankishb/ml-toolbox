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