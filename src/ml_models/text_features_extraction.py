

import numpy as np 
import pandas as pd 
import gc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def tfidf_feature(train, test, col_name, min_df=3, analyzer='word', token_pattern=r'\w{1,}', ngram=3, 
	stopwords='english', svd_component=120, svd_flag=False, max_features=None):
	"""return tfidf feature
	Args:
		train, test: dataframe
		col_name: column name of text feature
		min_df: if Int, then it represent count of the minimum words in corpus (remove very rare word)
		analyzer: [‘word’, ‘char’]
		ngram: max range of ngram
		token_pattern: [using: r'\w{1,}'] [by default: '(?u)\b\w\w+\b']
		stopwords: ['english' or customized by remove specific words]
		svd_component: n_component of svd feature transform
		svd_flag: Wheteher to run svd on top of that or not (by default: False)
		max_features: max no of features to keep, based on frequency. It will keep words with higher freq
	return:
		Transformed feature space of the text data, as well as tfidf function instance
		if svd_flag== True : train_tf, test_tf, tfv, svd
		else : train_tf, test_tf, tfv
	example:
		train_tfv, test_tfv, tfv = tfidf_feature(X_train, X_test, ['text'], min_df=3)
		train_svd, test_svd, complete_tfv, tfv, svd = tfidf_feature(X_train, X_test, ['text'], 
			min_df=3, svd_component=3, svd_flag=True)

	"""
	tfv = TfidfVectorizer(min_df=min_df,  max_features=max_features, 
	            strip_accents='unicode', analyzer=analyzer,max_df=1.0, 
	            token_pattern=token_pattern, ngram_range=(1, ngram), 
	            use_idf=1, smooth_idf=1, sublinear_tf=1,
	            stop_words = stopwords)

	complete_df = pd.concat([train[col_name], test[col_name]], axis=0)
# 	return complete_df

	tfv.fit(list(complete_df['text'].values))

	if svd_flag is False:
		train_tfv =  tfv.transform(train[col_name].values.ravel()) 
		test_tfv  = tfv.transform(test[col_name].values.ravel())

		del complete_df
		gc.collect()
		return train_tfv, test_tfv, tfv
	else:
		complete_tfv = tfv.transform(complete_df[col_name].values.ravel())
		svd = TruncatedSVD(n_components=svd_component)
		svd.fit(complete_tfv)
		complete_svd = svd.transform(complete_tfv)

		complete_svd = pd.DataFrame(data=complete_svd)
		complete_svd.columns = ['svd_'+str(i) for i in range(svd_component)]

		train_svd = complete_svd.iloc[:train.shape[0]]
		test_svd = complete_svd.iloc[train.shape[0]:].reset_index(drop=True)

		del complete_svd, complete_df
		gc.collect()
		return train_svd, test_svd, complete_tfv, tfv, svd

def countvect_feature(train, test, col_name, min_df=3, analyzer='word', token_pattern=r'\w{1,}', 
					ngram=3, stopwords='english', max_features=None):
	"""return CountVectorizer feature
	Args:
		train, test: dataset
		col_name: columns name of the text feature
		min_df: if Int, then it represent count of the minimum words in corpus (remove very rare word)
		analyzer: [‘word’, ‘char’]
		ngram: max range of ngram
		token_pattern: [using: r'\w{1,}'] [by default: '(?u)\b\w\w+\b']
		stopwords: ['english' or customized by remove specific words]
		max_features: max no of features to keep, based on frequency. It will keep words with higher freq
	return:
		Count feature space of the text data, as well as its function instance
	"""
	ctv = CountVectorizer(min_df=min_df,  max_features=max_features, 
	            strip_accents='unicode', analyzer=analyzer, 
	            token_pattern=token_pattern, ngram_range=(1, ngram), 
	            stop_words = stopwords)

	complete_df = pd.concat([train[col_name], test[col_name]], axis=0)
	ctv.fit(list(complete_df['text'].values))

	train_tf =  ctv.transform(train[col_name].values.ravel()) 
	test_tf  = ctv.transform(test[col_name].values.ravel())

	del complete_df
	gc.collect()
	return train_tf, test_tf, ctv
	




# import time
# import nltk
# import string
# from gensim.models import word2vec
# from nltk.corpus import stopwords
# from keras.preprocessing.text import text_to_word_sequence

# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))
# remove_word = []
# [remove_word.append(word) if "n't" in word else " " for word in stop_words ]
# stop_words_new = list(set(stop_words) - set(remove_word))


# stemmer = nltk.stem.porter.PorterStemmer()
# remove_punctuation_map = dict((ord(char), ' ') for char in string.punctuation)



# def stem_tokens(tokens):
#     lst = [stemmer.stem(item) for item in tokens]
#     return ' '.join(lst)

# def get_textfeats(df, col, flag=True):
#     df[col] = df[col].fillna('none').astype(str)
#     df[col] = df[col].str.lower()
#     df[col] = df[col].apply(lambda x: stem_tokens(
#     	nltk.word_tokenize(x.translate(remove_punctuation_map))))

#     return df

# def load_text_temp(df, col):
#     df = get_textfeats(df,col)
#     return df



# # def load_text(df, col):
# #     df = get_textfeats(df,col)
# #     train_desc = df[col].values
# #     train_corpus = [text_to_word_sequence(text) for text in train_desc]
# #     return train_corpus

# # def get_result(corpus, model):
# #     result = []
# #     for text in corpus:
# #         n_skip = 0
# #         for n_w, word in enumerate(text):
# #             try:
# #                 vec_ = model.wv[word]
# #             except:
# #                 n_skip += 1
# #                 continue
# #             if n_w == 0:
# #                 vec = vec_
# #             else:
# #                 vec = vec + vec_
# #         vec = vec / (n_w - n_skip + 1)
# #         result.append(vec)
        
# #     return result



# %%time

# n_components = 32
# text_features = []
# text_features_nms = []

# # tf_idf_save = []
# # Generate text features:
# for i in X_text.columns:
#     if i == 2:
#         n_components = 16
#     # Initialize decomposition methods:
#     print(f'generating features from: {i}')
#     check_result = load_text_temp(X_text[[i]], i)
#     #StopWords Removed 
#     check_result[i] = check_result[i].str.lower().str.split()
#     check_result[i] = check_result[i].apply(lambda x : [item for item in x if item not in stop])
#     check_result[i] = check_result[i].apply(lambda x: ' '.join(x))

# #     tfv = TfidfVectorizer(min_df=3,  max_features=None,
#     tfv = TfidfVectorizer(min_df=3,  max_features=None,
#                           strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
#                           ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1)
#     svd_ = TruncatedSVD(n_components=n_components, random_state=1337)
    
#     tfidf_col = tfv.fit_transform(check_result[i].values)

#     svd_col = svd_.fit_transform(tfidf_col)
#     svd_col = pd.DataFrame(svd_col)
#     svd_col = svd_col.add_prefix('TFIDF_{}_'.format(i))
    
#     text_features.append(svd_col)
    
#     from sklearn.decomposition import NMF
#     nms_model = NMF(n_components=15, random_state=1337)
#     nms_col = nms_model.fit_transform(tfidf_col)
#     nms_col = pd.DataFrame(nms_col)
#     nms_col = nms_col.add_prefix('NMS_{}_'.format(i))
    
#     text_features_nms.append(nms_col)
    
# text_features = pd.concat(text_features, axis=1)
# text_features_nms = pd.concat(text_features_nms, axis=1)

# X_temp = pd.concat([X_temp, text_features], axis=1)
# X_temp = pd.concat([X_temp, text_features_nms], axis=1)

# for ii in X_text.columns:
#     X_temp = X_temp.drop(ii, axis=1)




