import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D,Conv2D, Flatten, Dense,Embedding,Input,LSTM,Dropout,MaxPooling1D,Concatenate,GRU,BatchNormalization,Activation
from tensorflow.keras import Model
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from sklearn.metrics import classification_report
from collections import Counter
import matplotlib.pyplot as plt
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

lyrics_df=pd.read_csv("cleaned_lyrics.csv").astype(str)


X_train,X_test,Y_train,Y_test=train_test_split(lyrics_df.lyrics,lyrics_df.genre,test_size=0.2)

max_words = 600
max_len = 3000
tok = Tokenizer(num_words=max_words,split=" ")
tok.fit_on_texts(lyrics_df.lyrics)
vocab = tok.word_index

data_seq = tok.texts_to_sequences(X_train)
X_train_data_seq_mat = sequence.pad_sequences(data_seq,maxlen=max_len)
#
train_data_y = Y_train
le = LabelEncoder()
train_data_y = le.fit_transform(train_data_y).reshape(-1,1)
ohe = OneHotEncoder()
train_data_y = ohe.fit_transform(train_data_y).toarray()
#
data_seq = tok.texts_to_sequences(X_test)
#data_seq=tok.texts_to_matrix(X_test,mode="tfidf")
X_test_data_seq_mat = sequence.pad_sequences(data_seq,maxlen=max_len)
#
test_data_y = Y_test
le = LabelEncoder()
test_data_y = le.fit_transform(test_data_y).reshape(-1,1)
ohe = OneHotEncoder()
test_data_y = ohe.fit_transform(test_data_y).toarray()


embeddings_index = {}
f = open("glove/glove.6B.100d.txt", encoding = 'utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
# 预训练的词向量中没有出现的词用0向量表示
embedding_matrix = np.zeros((len(vocab) + 1, 100))
for word, i in vocab.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
# 模型中需要修改的仅仅是这里

## TextCNN
main_input = Input(shape=(max_len,), dtype='float64')
# 词嵌入（使用预训练的词向量）
embedder = Embedding(len(vocab) + 1, 100, input_length = max_len, weights = [embedding_matrix], trainable = False)
opt=RMSprop(learning_rate=10**-3)
embed = embedder(main_input)
conv1_1 = Conv1D(256, 3, padding='same')(embed)
bn1_1 = BatchNormalization()(conv1_1)
relu1_1 = Activation('relu')(bn1_1)
conv1_2 = Conv1D(128, 3, padding='same')(relu1_1)
bn1_2 = BatchNormalization()(conv1_2)
relu1_2 = Activation('relu')(bn1_2)
cnn1 = MaxPooling1D(pool_size=4)(relu1_2)
# cnn2模块，kernel_size = 4
conv2_1 = Conv1D(256, 4, padding='same')(embed)
bn2_1 = BatchNormalization()(conv2_1)
relu2_1 = Activation('relu')(bn2_1)
conv2_2 = Conv1D(128, 4, padding='same')(relu2_1)
bn2_2 = BatchNormalization()(conv2_2)
relu2_2 = Activation('relu')(bn2_2)
cnn2 = MaxPooling1D(pool_size=4)(relu2_2)
# cnn3模块，kernel_size = 5
conv3_1 = Conv1D(256, 5, padding='same')(embed)
bn3_1 = BatchNormalization()(conv3_1)
relu3_1 = Activation('relu')(bn3_1)
conv3_2 = Conv1D(128, 5, padding='same')(relu3_1)
bn3_2 = BatchNormalization()(conv3_2)
relu3_2 = Activation('relu')(bn3_2)
cnn3 = MaxPooling1D(pool_size=4)(relu3_2)
# 拼接三个模块
cnn= Concatenate(axis=1)([cnn1, cnn2, cnn3])
flat = Flatten()(cnn)
drop = Dropout(0.2)(flat)
fc = Dense(512)(drop)
bn = BatchNormalization()(fc)
main_output = Dense(4, activation='softmax')(bn)
model1 = Model(inputs = main_input, outputs = main_output)
model1.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])
model1.summary()
history=model1.fit(X_train_data_seq_mat , train_data_y,
          batch_size=32,
          epochs=10,
          validation_data=(X_test_data_seq_mat, test_data_y))

tcnn_res=model1.predict(X_test_data_seq_mat)

labels=["country","hiphop","rnb","rock"]
tcnn_label = tcnn_res.argmax(axis=-1)
tcnn_label_res=[labels[x] for j,x in enumerate(tcnn_label)]
print(classification_report(Y_test,tcnn_label_res,digits=4))

tcnn_count=Counter(tcnn_label_res)
plt.figure(figsize=(5,5))
plt.pie(x=tcnn_count.values(),labels=tcnn_count.keys(),autopct = '%3.2f%%')
plt.show()



## GRU
model=Sequential()
model.add(Embedding(len(vocab) + 1, 100, input_length = max_len, weights = [embedding_matrix], trainable = False))
opt=RMSprop()
#model.add(LSTM(128))
model.add(Dense(128,activation="relu"))
model.add(GRU(256))
model.add(Dense(128,activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(4,activation="softmax"))
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

model_fit_GRU = model.fit(X_train_data_seq_mat,train_data_y,validation_data=(X_test_data_seq_mat,test_data_y),batch_size=32,epochs=10)


## LSTM

model=Sequential()
model.add(Embedding(len(vocab) + 1, 100, input_length = max_len, weights = [embedding_matrix], trainable = False))
opt=RMSprop()
#model.add(Dense(128,activation="relu",name="FC3"))

model.add(Dense(128,activation="relu"))
model.add(LSTM(256))
model.add(Dense(128,activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(4,activation="softmax"))
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

model_fit_LSTM = model.fit(X_train_data_seq_mat,train_data_y,validation_data=(X_test_data_seq_mat,test_data_y),batch_size=32,epochs=10)
LSTM_res=model.predict(X_test_data_seq_mat)

import matplotlib.pyplot as plt

plt.subplot(211)
plt.title("Accuracy")
plt.plot(model_fit_LSTM.history["accuracy"], color="g", label="Train")
plt.plot(model_fit_LSTM.history["val_accuracy"], color="b", label="Test")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Loss")
plt.plot(model_fit_LSTM.history["loss"], color="g", label="Train")
plt.plot(model_fit_LSTM.history["val_loss"], color="b", label="Test")
plt.legend(loc="best")

plt.tight_layout()
plt.show()


LSTM_res=model_fit_LSTM.model.predict(X_test_data_seq_mat)

labels=["country","hiphop","rnb","rock"]
LSTM_label = LSTM_res.argmax(axis=-1)
LSTM_label_res=[labels[x] for j,x in enumerate(LSTM_label)]
print(classification_report(Y_test,LSTM_label_res,digits=4))

LSTM_count=Counter(LSTM_label_res)
plt.figure(figsize=(5,5))
plt.pie(x=LSTM_count.values(),labels=LSTM_count.keys(),autopct = '%3.2f%%')
plt.show()



GRU_res=model_fit_GRU.model.predict(X_test_data_seq_mat)

labels=["country","hiphop","rnb","rock"]
GRU_label = GRU_res.argmax(axis=-1)
GRU_label_res=[labels[x] for j,x in enumerate(GRU_label)]
print(classification_report(Y_test,GRU_label_res,digits=4))

GRU_count=Counter(GRU_label_res)
plt.figure(figsize=(5,5))
plt.pie(x=GRU_count.values(),labels=GRU_count.keys(),autopct = '%3.2f%%')
plt.show()