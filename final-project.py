import pandas as pd
import os, sys
from keras.models import Sequential
from keras.layers import *
import keras
import tensorflow as tf
import numpy as np
import math
import random
from tensorflow.keras import regularizers, initializers
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import json
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

labels = [0, 1]
Input = sys.argv[3]
RBNS5 = sys.argv[-1]
RNCMPT = sys.argv[2]
ofile = sys.argv[1]
names = [Input, RBNS5]

def generate_data(names):
    concentrations = list()
    for name in names:
        df = pd.read_csv(name, sep="\t", usecols=[0], names=['sequence'])
        df = df[0:250000]
        df = [seq for seq in df['sequence'] if all(ch != 'N' for ch in seq)]
        df = pd.DataFrame(data=df, columns=['sequence'])
        concentrations.append(df)
    return concentrations

def label_data(RBP, labels):
    for j in range(0, len(RBP)):
        RBP[j]['label'] = labels[j]
    return RBP

def merge_data(RBP):
    data = RBP[0]
    for j in range(1, len(RBP)):
        data = data.append(RBP[j], ignore_index=True)
    return data

# from string import maketrans    # in python free use str.maketrans isntead
def oneHot(string):
    trantab = str.maketrans('ACGTU', '01233')
    string = str(string)
    data = [int(x) for x in list(string.translate(trantab))]
    return np.eye(4)[data]

RBP = generate_data(names)
RBP = label_data(RBP, labels)
RBP = merge_data(RBP)
RBP = RBP.sample(frac=1, random_state=42)

X = RBP["sequence"]
Y = RBP["label"].values
X = np.array(list(map(oneHot, X)))

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)

acc = 0.
best_acc = 0
times = 0
while times < 6:

    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=5, strides=1, activation='relu', input_shape=(20, 4), use_bias=False,
                     padding='valid'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=128, kernel_size=5, strides=1, activation='relu', use_bias=False, padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # initiate RMSprop optimizer
    opt = keras.optimizers.Adam(lr=0.0005)

    # Let's train the model using RMSprop
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    history = model.fit(x_train, y_train,
                    batch_size=256,
                    epochs=1,
                    validation_data=(x_test, y_test),
                    verbose=0)
    acc = history.history['accuracy'][0]
    times += 1
    if acc > best_acc:
        best_model = model
        best_acc = acc
model = best_model
if best_acc <0.51 :
    model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])
    
    history = model.fit(x_train, y_train,
                        batch_size=256,
                        epochs=2,
                        validation_data=(x_test, y_test),
                        verbose=0)

x_result = pd.read_csv(RNCMPT, header=None)
x_result = list(map(oneHot, x_result[0]))
lengths = [len(seq) - 20 + 1 for seq in x_result]
x_result = [x_result[j][k:k + 20] for j in range(len(lengths)) for k in range(lengths[j])]
x_result = np.asarray(x_result)

y_predict = model.predict(x_result, batch_size=256)
splits = [0] * (len(lengths) - 1)
splits[0] = lengths[0]
for k in range(1, len(lengths) - 1):
    splits[k] += (lengths[k] + splits[k - 1])
y_pred = np.split(y_predict, splits)
y_hat = [np.max(result) for result in y_pred]
y_hat = np.asarray(y_hat)

df = pd.DataFrame(y_hat)
df.to_csv(ofile, sep="\t", index=False, header=False)

## #################################################################################### ##
from scipy.stats.stats import pearsonr
num = int(sys.argv[4])
if num < 17:
    y_result = pd.read_csv('RNCMPT_training/RBP{}.txt'.format(num), sep="\t", header=None)
    y_result = np.asarray(y_result[0])
    per = pearsonr(y_result, y_hat)
    end = time.time()
    print(f'pearson correlation = {per}')
    f = open(f"Results/Person.txt", "a")
    f.write(f"RBP{num}\t{str(per[0])}\n")
    f.close()

    # nStatics 
    f = open(f"Results/AUC.txt", "a")
    f.write(f'\nStatics Of of RBN{num}: {history.history}\n')
    f.close()