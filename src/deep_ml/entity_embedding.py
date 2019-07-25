

import numpy as np
import pandas as pd
from pathlib import Path
from keras.layers import Input, Embedding, Flatten, Dense, Dropout, BatchNormalization
from keras.layers import concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.models import Model, load_model, save_model
from keras.utils import Sequence
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score
from scipy.stats import gmean
from gensim.models import KeyedVectors
np.random.seed(10001)
import tensorflow as tf
tf.random.set_random_seed(10001)

from config import UTILITY, ROOT, SUBMISSIONS
from utils import read_data, read_base_feats, Tokenizer, normalize_disbursal
from get_w2v_features import make_sentences, W2V_CONFIG



from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)





EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)



EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)




EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): 
	return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())



# contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }


def load_glove(word_index, max_features):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): 
    	return word, np.asarray(arr, dtype='float32')[:300]
    
    embeddings_index = []
    for o in tqdm(open(EMBEDDING_FILE)):
        try:
            embeddings_index.append(get_coefs(*o.split(" ")))
        except Exception as e:
            print(e)
    
    embeddings_index = dict(embeddings_index)
            
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


glove_embeddings = load_glove(tokenizer.word_index, max_features)












""" Complete example ofn using entity embedding

from gensim.models import Word2Vec
from itertools import chain
import pandas as pd
import numpy as np

train_test = pd.read_csv('dataset/loan_prediction_data.csv')
W2V_CONFIG = {"cols": ["branch_id", "manufacturer_id", "supplier_id",
                      "Current_pincode_ID", "State_ID"],
             "vector_size": 150,
             "window_size":6,
             "epochs": 30,
             "min_count": 1,
             "sample": 1e-1
             }

train_test = train_test[W2V_CONFIG['cols']]
cols = train_test.columns

new_df = pd.DataFrame()
for col in cols:
    new_df[col] = train_test[col].apply(lambda x: col[:3]+'_'+str(x))
    
new_df.head()

new_df['sentence'] = new_df.apply(lambda x: " ".join(x), axis=1)


W2V_CONFIG = {
             "cols": ["branch_id", "manufacturer_id",
                      "supplier_id",
                      "Current_pincode_ID", "State_ID"],
             "vector_size": 32,
             "window_size":6,
             "epochs": 2,
             "min_count": 1,
             "sample": 1e-1
             }

all_sentences = list(new_df['sentence'].str.split(" ").values)
print("sentence length: ", len(all_sentences))

w2v_model = Word2Vec(min_count=W2V_CONFIG["min_count"],
                 window=W2V_CONFIG["window_size"],
                 size=W2V_CONFIG["vector_size"],
                 sample=W2V_CONFIG["sample"],
                 workers=4)
w2v_model.build_vocab(all_sentences, progress_per=10000)

print("Vocabulary corpus: ", len(w2v_model.wv.vocab))
w2v_model.train(all_sentences, total_examples=w2v_model.corpus_count, 
                epochs=100, report_delay=1)

print("one example: ")
w2v_model.wv.get_vector('bra_67'), w2v_model.wv.get_vector('bra_67').shape



from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D

## Tokenize the sentences
max_features = 15000
tokenizer = Tokenizer(num_words=max_features, filters="", lower=False)
tokenizer.fit_on_texts(list(new_df.sentence.values))
train_test_X = tokenizer.texts_to_sequences(new_df.sentence)

print("keras tokenizers:  ", len(tokenizer.word_index))
print("Vocabulary corpus: ", len(w2v_model.wv.vocab))

train_test_X[:10]


i = 0
for word, idx in tokenizer.word_index.items():
    print(word, idx)
    i += 1
    if i == 10:
        break


embedding_matrix = np.zeros((len(tokenizer.word_index)+1, 32))
for word, idx in tokenizer.word_index.items():
    embedding_matrix[idx] = w2v_model.wv.get_vector(word)

embedding_matrix


max_features = min(len(tokenizer.word_counts), max_features)
print("vocabulary size: ", max_features)


from keras import Model
embed_size = 32
inp = Input(shape=(len(train_test_X[0]),))
x = Embedding(max_features+1, embed_size, 
              weights=[embedding_matrix], trainable=False)(inp)
x = GlobalMaxPool1D()(x)
model = Model(inp, x)
model.summary()



# model.fit(np.array(train_test_X[:5]))
model.predict(np.array(train_test_X[10:11]))
train_test_X[:1]
np.max(embedding_matrix[train_test_X[10:11]], axis=0)

"""