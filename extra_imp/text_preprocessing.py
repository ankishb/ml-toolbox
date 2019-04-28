

from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups(subset=”train”)
print(“Number of text samples {}”.format(len(data.data)))



import pandas as pd
df = pd.DataFrame()
df = df.assign(text=data["data"]).assign(target=data["target"])
df.head()


# remove emails
import re
def remove_emails(text):
    regex =  r'\S*@\S*\s?'
    return re.sub(regex, '', text)

remove_emails(data.data[0])


# Removing Newline characters
def remove_newlinechars(text):
    regex = r'\s+'
    return re.sub(regex, ' ', text)

test_text = remove_newlinechars(test_text)
print(test_text)


# Tokenization
import nltk
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return list(filter(lambda word: word.isalnum(), tokens))

test_text = tokenize(test_text)
print(test_text)







# Removing Stopwords

from nltk.corpus import stopwords
stop_words = stopwords.words("english")

## Add some common words from text
stop_words.extend(["from","subject","summary","keywords","article"])

def remove_stopwords(words):
    filtered = filter(lambda word: word not in stop_words, words)
    return list(filtered)

test_text = remove_stopwords(test_text)
print(test_text)





# Lemmatization

import spacy
nlp = spacy.load("en_core_web_sm")

def lemmatize(text, nlp=nlp):    
    doc = nlp(" ".join(text))
    lemmatized = [token.lemma_ for token in doc]
    return lemmatized

test_text = lemmatize(test_text,nlp)
print(test_text)









import time
t0 = time.time()

def clean_text(df):
    df["cleaned_text"] = df.text.map(lambda text:text.lower()).map(remove_emails).map(remove_newlinechars).map(remove_stopwords).map(lemmatize)
    return df

df = clean_text(df)
t1 = time.time()
print("Time to process without Dask {}".format(t1-t0))









import dask.dataframe as ddf
dask_dataframe = ddf.from_pandas(df, npartitions=6)
t0 = time.time()
result = dask_dataframe.map_partitions(clean_text, meta=df)
df = result.compute()
t1 = time.time()
print("Time to process with Dask {}".format(t1-t0))














