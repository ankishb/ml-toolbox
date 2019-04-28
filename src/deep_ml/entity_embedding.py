

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