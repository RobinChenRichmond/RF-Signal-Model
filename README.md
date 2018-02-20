# RF-Signal-Model

We are trying to build different machine learning models to solve the Signal Modulation Classification problem.

With the dataset from [RadioML](https://radioml.org/datasets/radioml-2016-10-dataset/), we work from 2 approaches to improve the classification performance for the dataset itself and its subset:

## Improved CNN model for RadioML dataset
For this model, we use a GTX-980Ti GPU to speed up the execution time.

With our new architecture, the CNN model has the total data's Validation Accuracy improved to 56.04% from 49.49%, normal data's Validation Accuracy improved to 82% from 70%, with the running time for each epoch decreased to 13s from 15s(With the early stopping mechanism, it usually takes 40-60 epochs to train the model).

### Here's the summary of model:
```
Layer (type)                   Output Shape              Param #   
=================================================================
reshape_1 (Reshape)            (None, 2, 128, 1)         0         
_________________________________________________________________
zero_padding2d_1 (ZeroPadding) (None, 2, 132, 1)         0         
_________________________________________________________________
conv2d_1 (Conv2D)              (None, 2, 129, 64)        320       
_________________________________________________________________
dropout_1 (Dropout)            (None, 2, 129, 64)        0         
_________________________________________________________________
zero_padding2d_2 (ZeroPadding) (None, 2, 133, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)              (None, 1, 130, 64)        32832     
_________________________________________________________________
dropout_2 (Dropout)            (None, 1, 130, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)              (None, 1, 123, 128)       65664     
_________________________________________________________________
dropout_3 (Dropout)            (None, 1, 123, 128)       0         
_________________________________________________________________
conv2d_4 (Conv2D)              (None, 1, 116, 128)       131200    
_________________________________________________________________
dropout_4 (Dropout)            (None, 1, 116, 128)       0         
_________________________________________________________________
flatten_1 (Flatten)            (None, 14848)             0         
_________________________________________________________________
dense1 (Dense)                 (None, 256)               3801344   
_________________________________________________________________
dropout_5 (Dropout)            (None, 256)               0         
_________________________________________________________________
dense2 (Dense)                 (None, 11)                2827      
_________________________________________________________________
reshape_2 (Reshape)            (None, 11)                0         
=================================================================
Total params: 4,034,187
Trainable params: 4,034,187
Non-trainable params: 0
```


## Spectrogram-CNN for RadioML subset

In our second approach, we converted the given data set into spectrogram images of size 41px x 108px and ran CNN models on the image data set. Sice this is a highly time and memory intensive process, we chose a smaller subets of the data. The subsets chosen are:
1. Modulations - BPSK, QAM16, AM-DSB, WBFM with SNR ranging from +8 to +18 dB with steps of 2
2. Modulations - BPSK, QAM16, AM-DSB, WBFM with SNR ranging from 􀀀10 to +8 dB with steps of 2
3. Modulations - BPSK, QAM16, AM-DSB, WBFM, AB-SSB, QPSK with SNR ranging from 0 to +18 dB with steps of 2

The results of the model are shown below:

![alt text](https://github.com/RobinChenRichmond/RF-Signal-Model/blob/master/acc.JPG)

![alt text](https://github.com/RobinChenRichmond/RF-Signal-Model/blob/master/confusion.JPG)

