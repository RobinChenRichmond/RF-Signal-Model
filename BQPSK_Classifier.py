from __future__ import print_function
import math
import numpy as np
import cPickle, random, sys
import matplotlib.pyplot as plt

# Load the dataset ...
#  You will need to seperately download or generate this file
Xd = cPickle.load(open("/Users/guanyuchen/Desktop/RML2016.10a_dict.dat",'rb'))
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])

BPSK = Xd[('BPSK',18)][0:800,:,:]
QPSK = Xd[('QPSK',18)][0:800,:,:]
X = np.concatenate((BPSK,QPSK))

BPSK_TEST = Xd[('BPSK',18)][800:1000,:,:]
QPSK_TEST = Xd[('QPSK',18)][800:1000,:,:]
test = np.concatenate((BPSK_TEST,QPSK_TEST))

b = np.zeros(800)
q = np.ones(800)
lbl = np.concatenate((b,q))

btest = np.zeros(200)
qtest = np.ones(200)
lbltest = np.concatenate((btest,qtest))

import keras
from keras import metrics
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
#from sklearn.model_selection import ShuffleSplit

batch_size = 128
num_classes = 2
epochs = 50

# input image dimensions
img_rows, img_cols = 2, 128

#rs = ShuffleSplit(n_splits=3, test_size=.25, random_state=0)
# the data, shuffled and split between train and test sets
x_train = X
y_train = lbl
x_test = test
y_test = lbltest

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train)
print(y_train)

print(x_test)
print(y_test)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(1, 2),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=150,
          verbose=1,
          validation_split = 0.1,shuffle=True)
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

x = model.predict(x_test)