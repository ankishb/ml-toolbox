'''
#Sequence to sequence example in Keras (character-level).

This script demonstrates how to implement a basic character-level
sequence-to-sequence model. We apply it to translating
short English sentences into short French sentences,
character-by-character. Note that it is fairly unusual to
do character-level machine translation, as word-level
models are more common in this domain.

**Summary of the algorithm**

- We start with input sequences from a domain (e.g. English sentences)
    and corresponding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    It uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.

**Data download**

[English to French sentence pairs.
](http://www.manythings.org/anki/fra-eng.zip)

[Lots of neat sentence pairs datasets.
](http://www.manythings.org/anki/)

**References**

- [Sequence to Sequence Learning with Neural Networks
   ](https://arxiv.org/abs/1409.3215)
- [Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    ](https://arxiv.org/abs/1406.1078)
'''
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'fra-eng/fra.txt'

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_data[i, t:, target_token_index[' ']] = 1.
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)





def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
    return text


def tagger(decoder_input_sentence):
  bos = "<BOS> "
  eos = " <EOS>"
  final_target = [bos + text + eos for text in decoder_input_sentence] 
  return final_target

decoder_inputs = tagger(decoder_input_text)




from keras.preprocessing.text import Tokenizer

def vocab_creater(text_lists, VOCAB_SIZE):

  tokenizer = Tokenizer(num_words=VOCAB_SIZE)
  tokenizer.fit_on_texts(text_lists)
  dictionary = tokenizer.word_index
  
  word2idx = {}
  idx2word = {}
  for k, v in dictionary.items():
      if v < VOCAB_SIZE:
          word2idx[k] = v
          index2word[v] = k
      if v >= VOCAB_SIZE-1:
          continue
          
  return word2idx, idx2word

word2idx, idx2word = vocab_creater(text_lists=encoder_input_text+decoder_input_text, VOCAB_SIZE=14999)


from keras.preprocessing.text import Tokenizer
VOCAB_SIZE = 14999

def text2seq(encoder_text, decoder_text, VOCAB_SIZE):

  tokenizer = Tokenizer(num_words=VOCAB_SIZE)
  encoder_sequences = tokenizer.texts_to_sequences(encoder_text)
  decoder_sequences = tokenizer.texts_to_sequences(decoder_text)
  
  return encoder_sequences, decoder_sequences

encoder_sequences, decoder_sequences = text2seq(encoder_text, decoder_text, VOCAB_SIZE) 


from keras.preprocessing.sequence import pad_sequences

def padding(encoder_sequences, decoder_sequences, MAX_LEN):
  
  encoder_input_data = pad_sequences(encoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
  decoder_input_data = pad_sequences(decoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
  
  return encoder_input_data, decoder_input_data

encoder_input_data, decoder_input_data = padding(encoder_sequences, decoder_sequences, MAX_LEN):


# GLOVE_DIR = path for glove.6B.100d.txt
def glove_100d_dictionary(GLOVE_DIR):
  embeddings_index = {}
  f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
  f.close()
  return embeddings_index


  # this time: embedding_dimention = 100d
def embedding_matrix_creater(embedding_dimention):
  embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimention))
  for word, i in word_index.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
          embedding_matrix[i] = embedding_vector
  return embedding_matrix


def embedding_layer_creater(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, embedding_matrix):
  
  embedding_layer = Embedding(input_dim = VOCAB_SIZE, 
                              output_dim = EMBEDDING_DIM,
                              input_length = MAX_LEN,
                              weights = [embedding_matrix],
                              trainable = False)
  return embedding_layer

embedding_layer = embedding_layer_creater(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, embedding_matrix)


import numpy as np
# MAX_LEN = 20
# num_samples = len(encoder_sequences)
# VOCAB_SIZE = 15000

def decoder_output_creater(decoder_input_data, num_samples, MAX_LEN, VOCAB_SIZE):
  
  decoder_output_data = np.zeros((num_samples, MAX_LEN, VOCAB_SIZE), dtype="float32")

  for i, seqs in enumerate(decoder_input_data):
      for j, seq in enumerate(seqs):
          if j > 0:
              decoder_output_data[i][j][seq] = 1.
  print(decoder_output_data.shape)
  
  return decoder_output_data

decoder_output_data = decoder_output_creater(decoder_input_data, num_samples, MAX_LEN, VOCAB_SIZE)


from sklearn.model_selection import train_test_split

def data_spliter(encoder_input_data, decoder_input_data, test_size1=0.2, test_size2=0.3):
  
  en_train, en_test, de_train, de_test = train_test_split(encoder_input_data, decoder_input_data, test_size=test_size1)
  en_train, en_val, de_train, de_val = train_test_split(en_train, de_train, test_size=test_size2)
  
  return en_train, en_val, en_test, de_train, de_val, de_test

en_train, en_val, en_test, de_train, de_val, de_test = data_spliter(encoder_input_data, decoder_input_data)










































import warnings
from typing import Optional, Type

import numpy as np
import tensorflow as tf
from tensorflow import keras

import phasa.layers.layers as layers
import tensorflow_addons as tfa
from phasa.data.base.normalizer import TextNormalizer
from phasa.data.base.tokenizers import BaseTokenizer
from phasa.hyperparams.hyperparams import Params
from phasa.layers.encoder_layers import RecurrentEncoderLayer
from phasa.models.base import BaseModel


class ListenAttendSpell(BaseModel):
    """
    This is a RNN encoder-decoder model with
        1. RNN encoder
        2. RNNLM decoder with attention

    This is similar to a Listen-Attend-Spell (LAS) sequence-to-sequence model architecture,
    although not equivalent to the implementation in the paper
    """

    @classmethod
    def get_params(cls):
        params = super().get_params()

        # encoder: TODO(ranirudh): reconcile with RnnMultiLayer
        params.define(
            "encoder_params",
            ,
            "Params: Parameters of the RecurrentEncoderLayer encoder class",
        )
        params.define(
            "preprocessing_layer",
            None,
            "tf.keras.layers.Layer: Layer instance to apply preprocessing before feeding into encoder input. "
            "Examples include layers that can perform time reduction",
        )
        params.define(
            "encoder_embedding",
            False,
            "bool: Whether to add an embedding layer on the input side before feeding into the encoder. "
            "Relavant for tasks with input side text data",
        )

        # decoder
        params.define("decoder_input_vocab_size", None, "int: Input vocab size for the decoder")
        params.define("decoder_target_vocab_size", None, "int: Output vocab size for the decoder")
        params.define(
            "decoder_input_embedding_dim", None, "int: Input embedding dimension for the decoder"
        )
        params.define("decoder_num_layers", 1, "int: Number of recurrent layers in the decoder")
        params.define(
            "decoder_num_units",
            None,
            "int: Number of units in each of the decoder recurrent layers",
        )
        params.define(
            "decoder_rnn_cell_cls",
            keras.layers.LSTMCell,
            "keras.layers.RNNCell: Any keras compatible RNN (LSTM, GRU, etc)",
        )
        params.define(
            "tokenizer_params",
            None,
            "Params of BaseTokenizer: tokenizer used for decoding integer -> symbol",
        )

        # symbol table related
        params.define("eos_id", None, "int: integer id for end of sentence symbol </s>")
        params.define("sos_id", None, "int: integer id for start of sentence symbol <s>")
        params.define("unk_id", None, "int: integer id for unknown symbol <unk>")
        params.define(
            "prepend_sos_symbol",
            True,
            "bool: whether to prepend params.sos_id token to the input of label sequence",
        )

        

        params.define(
            "max_decode_length", 250, "int: maximum decode length while evaluating or testing"
        )

        # inference specific configuration
        params.define("beam_width", 1, "int: beam width to use for decoding")

        # debugging params
        params.define(
            "ignore_mask",
            False,
            "bool: (for debug only) ignore mask on encoder side to enable using CuDNN "
            "implementation",
        )
        params.define(
            "state_transfer",
            True,
            "bool: whether to transfer state information from encoder final state to decoder initial state",
        )

        params.define(
            "text_postprocessing_pipeline",
            None,
            "TextNormalizationPipeline: A pipeline used to postprocess decoded text",
        )

        return params

    def __init__(self, params: Params):
        super().__init__()

        if params.tokenizer_params is not None:
            self.tokenizer: BaseTokenizer = params.tokenizer_params.cls(params.tokenizer_params)
        else:
            warnings.warn(
                "WARNING: tokenizer parameter is not set. _forwarding_inference "
                "method will not work for this model."
            )
        self.sos_id: int = params.sos_id
        self.eos_id: int = params.eos_id
        self.unk_id: int = params.unk_id
        self.prepend_sos_symbol: bool = params.prepend_sos_symbol

        # encoder
        self.encoder = RecurrentEncoderLayer.get_params()

        # attention
        self.attention_num_heads = 1
        self.attention_depth = 512 # n_dims
        self.use_attention = True
        self.attention_layer_size = ([2] * self.attention_num_heads)

        # decoder
        self.decoder_num_layers = 2
        self.decoder_num_units = 512

        self.decoder_input_embedding_dim = 512

        self.decoder_input_vocab_size: int = params.decoder_input_vocab_size
        self.decoder_target_vocab_size: int = params.decoder_target_vocab_size
        self.beam_width = 1

        cells = [
            tf.keras.layers.LSTM(self.decoder_num_units)
            for _ in range(self.decoder_num_layers)
        ]
        self.decoder_cell = tf.keras.layers.StackedRNNCells(cells)

        if self.attention_num_heads is not None and self.attention_depth is not None:
            self.attention_mechanism_multi = [
                tfa.seq2seq.BahdanauAttention(self.attention_depth)
                for _ in range(self.attention_num_heads)
            ]
            # concatenate context vectors from attention heads
            self.decoder_cell = tfa.seq2seq.AttentionWrapper(
                self.decoder_cell,
                attention_mechanism=self.attention_mechanism_multi,
                attention_layer_size=self.attention_layer_size,
                )


        # prepare embeddings        

        self.encoder_embedding_layer = tf.keras.layers.Embedding(
            self.encoder_input_vocab_size, self.encoder_input_embedding_dim
        )
        self.decoder_embedding_layer = tf.keras.layers.Embedding(
            self.decoder_input_vocab_size, self.decoder_input_embedding_dim
        )

        self.ignore_mask = params.ignore_mask

        # Sampler and decoders
        self.training_sampler = tfa.seq2seq.sampler.TrainingSampler()
        self.decoder_output_layer = tf.keras.layers.Dense(self.decoder_target_vocab_size)
        self.training_decoder = tfa.seq2seq.BasicDecoder(
            self.decoder_cell, self.training_sampler, output_layer=self.decoder_output_layer
        )
        self.max_decode_length = 256

        # state transfer
        self.state_transfer = params.state_transfer

        if not self.state_transfer:
            assert self.use_attention, "Must enable attention module to disable state transference"

        self.text_postprocessor = TextNormalizer(params.text_postprocessing_pipeline)

    def _call_encoder(self, encoder_input):
        #  encoder_outputs: [batch_size, max_seq_length, num_units]
        #  and encoder_state: [[batch_size, num_units]..#num_layers]
        encoder_input = self.encoder_embedding_layer(encoder_input)
        max_seq_len = tf.shape(encoder_input)[1]

        encoder_mask = tf.sequence_mask(encoder_input_lengths, max_seq_len)
        if self.ignore_mask:
            encoder_mask = None
        encoder_outputs, encoder_final_state = self.encoder(
            encoder_input, prev_encoder_states=None, mask=encoder_mask
        )

        return encoder_outputs, encoder_final_state, encoder_mask

    def _forwarding_train(self, encoder_inputs, decoder_inputs):
        """Training mode that uses teacher forcing"""

        encoder_outputs, encoder_final_state, encoder_mask = self._call_encoder(
            encoder_inputs)

        batch_size, decoder_lengths = tf.shape(decoder_inputs)[:2]
        decoder_inputs = decoder_inputs[:, : tf.reduce_max(decoder_lengths)]

        # Prepend sos symbol at the start of each label sequence.
        # We don't adjust decoder_lengths, accounted by </s> in the labels which should not be used in the input
        if self.prepend_sos_symbol:
            sos = tf.fill([batch_size, 1], self.sos_id)
            decoder_inputs = tf.concat([sos, decoder_inputs], 1)

        decoder_initial_state = self._get_decoder_initial_state(
            encoder_outputs, 
            encoder_final_state, 
            encoder_mask
        )

        decoder_emb_input = self.decoder_embedding_layer(decoder_input)
        final_outputs, final_state, final_sequence_lengths = self.training_decoder(
            inputs=decoder_emb_input,
            initial_state=decoder_initial_state,
            sequence_length=decoder_lengths,  # this is passed to sampler.initialize() as stopping criterion
            training=True,  # relevant when dropout/recurrent_dropout is used
        )
        logits = final_outputs.rnn_output
        return logits

    def _setup_attention_memory(self, encoder_outputs, encoder_mask):
        """
        Setup up the memory (encoder outputs) for the attention mechanism
        Ensure manual resetting works in graph + eager, call attention mechanism twice:
            https://github.com/tensorflow/addons/issues/535 and https://github.com/tensorflow/addons/pull/547
        """

        for attention_mechanism in self.attention_mechanism_multi:
            attention_mechanism(encoder_outputs, mask=encoder_mask, setup_memory=True)
            attention_mechanism(encoder_outputs, mask=encoder_mask, setup_memory=True)

    def _get_decoder_initial_state_with_attention(self, encoder_outputs, encoder_final_state):
        """
        The final LSTM states of the encoder are set as initial states of the decoder
        Logic to handle getting decoder initial state with a decoder cell wrapped in tfa.seq2seq.AttentionWrapper
        Args:
            encoder_outputs: Outputs from the encoder of shape: (batch_size, seq_len, feat_dim)
            encoder_final_state: Tuple of encoder states, each corresponding to the initial state of RNN encoder layer

        Returns:
            decoder_initial_state: Tuple of initial states of the decoder, each corresponding to a layer of RNN decoder

        """

        decoder_initial_state = self.decoder_cell.get_initial_state(
            batch_size=tf.shape(encoder_outputs)[0], 
            dtype=tf.float32
        )
        if self.state_transfer:
            decoder_initial_state = decoder_initial_state.clone(
                cell_state=tuple(encoder_final_state)
            )

        return decoder_initial_state

    def _get_decoder_initial_state(self, encoder_outputs, encoder_final_state, encoder_mask):
        if self.use_attention:
            self._setup_attention_memory(encoder_outputs, encoder_mask)
            decoder_initial_state = self._get_decoder_initial_state_with_attention(
                encoder_outputs, 
                encoder_final_state
            )
        else:
            decoder_initial_state = tuple(encoder_final_state)
        return decoder_initial_state

    def _forwarding_inference(self, encoder_inputs):
        """
        Beam search decoding
        Args:
            inputs:
                dict with the following keys expected. Extra keys in the dict are ignored
                    "features": Tensor, shape=(batch_size, input_seq_len, feat_dim), dtype=tf.float32
                    "labels": Tensor, shape=(batch_size, output_seq_len), dtype=tf.int32
                    "x_lengths": Tensor, shape=(batch_size), dtype=tf.int32
                    "y_lengths": Tensor, shape=(batch_size), dtype=tf.int32

        Returns:
            Result:
                dict with following keys
                    decoded_strings: tf.Tensor shape: batch_size, nbest_size; each row is decoded string
                    decoded_indices: tf.Tensor DecoderOutputs.sequence
        """

        batch_size = tf.shape(encoder_inputs)[0]
        encoder_outputs, encoder_final_state, encoder_mask = self._call_encoder(
            encoder_inputs)

        tiled_encoder_outputs = tfa.seq2seq.tile_batch(
            encoder_outputs, 
            multiplier=self.beam_width
        )
        tiled_encoder_final_state = tfa.seq2seq.tile_batch(
            encoder_final_state, 
            multiplier=self.beam_width
        )
        tiled_encoder_mask = tfa.seq2seq.tile_batch(
            encoder_mask, 
            multiplier=self.beam_width
        )

        decoder_initial_state = self._get_decoder_initial_state(
            tiled_encoder_outputs, 
            tiled_encoder_final_state, 
            tiled_encoder_mask
        )

        decoder = tfa.seq2seq.BeamSearchDecoder(
            self.decoder_cell,
            beam_width=self.beam_width,
            embedding_fn=self.decoder_embedding_layer,
            output_layer=self.decoder_output_layer,
            maximum_iterations=self.max_decode_length,
        )

        final_outputs, final_state, final_sequence_lengths = decoder(
            None,
            initial_state=decoder_initial_state,
            start_tokens=tf.fill([batch_size], self.sos_id),
            end_token=self.eos_id,
            training=False,  # relevant when dropout/recurrent_dropout is used
        )

        # TODO(ranirudh): only 1-best exposed for now
        one_best = final_outputs.predicted_ids[:, :, 0]
        one_best_lengths = final_sequence_lengths[:, 0]

        result = dict()
        # same format as rnnt_decoder.DecoderOutputs.sequence
        result["decoded_indices"] = tf.expand_dims(
            tf.RaggedTensor.from_tensor(
                one_best, 
                lengths=one_best_lengths
            ), axis=1
        )
        # relies on the tokenizer.decode stripping all eos_id. shape: (batch_size, nbest_size)
        result["decoded_strings"] = tf.reshape(
            self.tokenizer.decode(
                result["decoded_indices"].to_tensor(default_value=self.tokenizer.eos_id)
            ),
            shape=(batch_size, -1),
        )

        if self.text_postprocessor is not None:
            result["decoded_strings"] = self.text_postprocessor.normalize(result["decoded_strings"])

        return result

