

EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))


EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)



EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)






corpus = docs = ['Well done!',
                 'Good work',
                 'Great effort',
                 'nice work',
                 'Excellent!',
                 'Weak',
                 'Poor effort!',
                 'not good',
                 'poor work',
                 'Could have done better.']


EMBEDDING_FILE = 'dataset/glove.6B/glove.6B.50d.txt'
def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(corpus))
corpa = tokenizer.texts_to_sequences(corpus)
corpa

tokenizer.word_index


all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]


word_index = tokenizer.word_index
nb_words = len(word_index)
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= nb_words: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        print(word)
        embedding_matrix[i] = embedding_vector
  

good = embeddings_index.get('good')
bad = embeddings_index.get('bad')
worse = embeddings_index.get('worse')
worst = embeddings_index.get('worst')

np.linalg.norm(good-bad), np.linalg.norm(worse-bad), \
np.linalg.norm(worse-worst), np.linalg.norm(good-worst)

from scipy.spatial.distance import cosine

cosine(good, bad), cosine(worse, bad), cosine(worse, worst), cosine(good, worst)










pad_sequences

keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)

Arguments

    sequences: List of lists, where each element is a sequence.
    maxlen: Int, maximum length of all sequences.
    dtype: Type of the output sequences. To pad sequences with variable length strings, you can use object.
    padding: String, 'pre' or 'post': pad either before or after each sequence.
    truncating: String, 'pre' or 'post': remove values from sequences larger than maxlen, either at the beginning or at the end of the sequences.
    value: Float or String, padding value.

Returns

    x: Numpy array with shape (len(sequences), maxlen)




# keras.preprocessing.text.text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')

#     text: Input text (string).
#     filters: list (or concatenation) of characters to filter out, such as punctuation. Default: !"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n, includes basic punctuation, tabs, and newlines.
#     lower: boolean. Whether to convert the input to lowercase.
#     split: str. Separator for word splitting.







Embedding

keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform',input_length=None)

Arguments

    input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
    output_dim: int >= 0. Dimension of the dense embedding.
    embeddings_initializer: Initializer for the embeddings matrix (see initializers)
    embeddings_regularizer: Regularizer function applied to the embeddings matrix (see regularizer).
    activity_regularizer: Regularizer function applied to the output of the layer (its "activation"). (see regularizer).
    embeddings_constraint: Constraint function applied to the embeddings matrix (see constraints).
    input_length: Length of input sequences, when it is constant. This argument is required if you are going to connect Flatten then Dense layers upstream (without it, the shape of the dense outputs cannot be computed).

Input shape

2D tensor with shape: (batch_size, sequence_length).

Output shape

3D tensor with shape: (batch_size, sequence_length, output_dim)





LSTM

keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)

Long Short-Term Memory layer - Hochreiter 1997.

Arguments

    dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.
    implementation: Implementation mode, either 1 or 2. Mode 1 will structure its operations as a larger number of smaller dot products and additions, whereas mode 2 will batch them into fewer, larger operations. These modes will have different performance profiles on different hardware and for different applications.
    return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
    return_state: Boolean. Whether to return the last state in addition to the output. The returned elements of the states list are the hidden state and the cell state, respectively.
    stateful: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
    unroll: Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.









keras.layers.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False)

Arguments

    dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.
    implementation: Implementation mode, either 1 or 2. Mode 1 will structure its operations as a larger number of smaller dot products and additions, whereas mode 2 will batch them into fewer, larger operations. These modes will have different performance profiles on different hardware and for different applications.
    return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
    return_state: Boolean. Whether to return the last state in addition to the output.
    go_backwards: Boolean (default False). If True, process the input sequence backwards and return the reversed sequence.
    stateful: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
    unroll: Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.
    reset_after: GRU convention (whether to apply reset gate after or before matrix multiplication). False = "before" (default), True = "after" (CuDNN compatible).



