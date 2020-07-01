import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import csv
import pickle
import os

EMBEDDING_DIM=100
files = ['entity_training.csv']

x_train=[]
y_train=[]

for file in files:
    with open(file) as csvfile: 
    # creating a csv reader object 
        csvreader = csv.reader(csvfile) 
        fields = next(csvreader)
        for row in csvreader:
            x_train.append(row[0])
            temp = row[1]
            temp = temp.split()
            temp = [float(x) for x in temp]
            y_train.append(temp)


embeddings_index = {}
word_index={}
index=1
f = open(os.path.join('./', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
    word_index[word]=index
    index=index+1
f.close()

with open('word_index.p', 'wb') as fp:
    pickle.dump(word_index, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('embeddings_index.p', 'wb') as fp:
    pickle.dump(embeddings_index, fp, protocol=pickle.HIGHEST_PROTOCOL)


vocab_size = len(word_index)
print(vocab_size)
x_train_final=[]
for sent in x_train:
    sent = sent.split()
    temp=[]
    for word in sent:
        if word in word_index:
            temp.append(word_index[word])
        else:
            temp.append(0)
    x_train_final.append(temp)
x_train=x_train_final
x_train = pad_sequences(x_train)
y_train = pad_sequences(y_train)
x_train = np.array(x_train)
y_train= np.array(y_train,dtype='float')
print(np.shape(x_train))
print(np.shape(y_train))

## Training Data complete

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector




input_layer = keras.Input(shape=(None,), name="InputLayer")
embedding_layer = layers.Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(input_layer)


Birnn_layer = layers.Bidirectional(layers.LSTM(int(EMBEDDING_DIM/2), return_sequences=True))(embedding_layer)
time_dist_1 = layers.TimeDistributed(layers.Dense(50))(Birnn_layer)
time_dist_2 = layers.TimeDistributed(layers.Dense(16))(time_dist_1)
output_layer = layers.TimeDistributed(layers.Dense(2, activation="softmax"), name="OutputLayer")(time_dist_2)

model = keras.Model(
    inputs=[input_layer],
    outputs=[output_layer]
)


keras.utils.plot_model(model, "entity_recognizer.png", show_shapes=True)
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-5)

model.compile(
    optimizer=opt,               
    loss= 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    {"InputLayer": x_train},
    {"OutputLayer": y_train},
    epochs=11,
    batch_size= 1
)

model.save('entity_recognizer')

