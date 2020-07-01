import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from pprint import pprint
import numpy as np
import pickle

def load_dataset(filename):
    df = pd.read_csv(filename, encoding = "latin1", names = ["Sentence", "Intent"], sep=';')
    intent = list(df["Intent"])
    unique_intent = list(set(intent))
    sentences = list(df["Sentence"])
    return (intent, unique_intent, sentences)
data = load_dataset("main_data.csv")

def cleaning(sentences):
    words = []
    for s in sentences:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
        w = word_tokenize(clean)
        #lemmatizing
        lemmatizer = WordNetLemmatizer()
        words.append([lemmatizer.lemmatize(i.lower()) for i in w])
    return words
cleaned_sentences=cleaning(data[2])

#creating tokenizer
def create_tokenizer(words,filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
    token = Tokenizer(filters = filters)
    token.fit_on_texts(words)
    return token

#input_tokenizer = create_tokenizer(cleaned_sentences)

#getting maximum length
def max_length(words):
    return(len(max(words, key = len)))
max_len=max_length(cleaned_sentences)

#encoding list of words
def encoding_doc(token, words):
    return(token.texts_to_sequences(words))
#encoded_doc=encoding_doc(input_tokenizer,cleaned_sentences)

def padding_doc(encoded_doc, max_length):
  return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))
#padded_doc=padding_doc(encoded_doc,max_len)
#print(padded_doc)

with open('word_index.p', 'rb') as fp:
    word_index = pickle.load(fp)

with open('embeddings_index.p', 'rb') as fp:
    embeddings_index = pickle.load(fp)


encoded_doc=[]
for sentence in cleaned_sentences:
    temp=[]
    for word in sentence:
        if word in word_index:
            temp.append(word_index[word])
        else:
            temp.append(0)
    encoded_doc.append(temp)

padded_doc = pad_sequences(encoded_doc,padding='post')
o = OneHotEncoder(sparse = False)

output_one_hot = o.fit_transform([[i] for i in data[0]])


train_X = padded_doc
train_Y = output_one_hot


EMBEDDING_DIM=100
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, Bidirectional, BatchNormalization
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import load_model
# def create_model(vocab_size, max_length):
#     model = Sequential()
#     model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length = None, weights=[embedding_matrix],  trainable = False))
#     model.add(Bidirectional(LSTM(128)))
#     model.add(Dense(64, activation = "relu"))
#     model.add(Dropout(0.5))
#     model.add(Dense(64, activation = "relu"))
#     model.add(Dropout(0.5))
#     model.add(BatchNormalization())
#     model.add(Dense(4, activation = "softmax"))
#     return model
# vocab_size = len(word_index) + 1
# model=create_model(vocab_size, max_len)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# print(model.summary())
# model.fit(train_X, train_Y, epochs=200)
# model.save('model.h5')
# exit()
model = load_model('model.h5')
# print("--------------------------------")
# print("The output intent mapping is as follows:")
# print([o.inverse_transform([[1,0,0,0]])[0][0],o.inverse_transform([[0,1,0,0]])[0][0],o.inverse_transform([[0,0,1,0]])[0][0],o.inverse_transform([[0,0,0,1]])[0][0]])
def predictions(text):
    global o
    global max_len
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
    test_word = word_tokenize(clean)
    lemmatizer = WordNetLemmatizer()
    test_word = [lemmatizer.lemmatize(w.lower()) for w in test_word]
    test_ls=[]
    for word in test_word:
        if word in word_index:
            test_ls.append(word_index[word])
        else:
            test_ls.append(0)
    test_ls = np.array(test_ls).reshape(1, len(test_ls))
    #print(test_ls)
    #x = padding_doc(test_ls, max_len)
    x=test_ls
    pred = model.predict(x)
    #return pred
    round_pred = np.around(pred)
    print(round_pred)
    return o.inverse_transform(round_pred)[0][0]                                          

