#!/usr/bin/env python
# coding: utf-8

# ## Convolution 1-D for RMM2 network for GPU tuning 

# Written by Abirlal Metya, Panini Dasgupta, Manmeet Singh (16/01/2020)

# import modules

# In[ ]:


import numpy as np
from numpy.random import seed
seed(4)
import random as rn
rn.seed(4)
import tensorflow
tensorflow.random.set_seed(4)
import os
os.environ["PYTHONHASHSEED"] = '4'

import pandas as pd
import datetime
import hilbert_data1_jgrjd_20CRV3
from sklearn.preprocessing import MinMaxScaler
import itertools
import multiprocessing
from IPython.display import clear_output
import tqdm

import keras 
from keras.models import Sequential
from keras.layers import Input,Dense, Conv1D, Flatten,MaxPooling1D,Dropout, Activation, Flatten,Add
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping


# ### Test and Train Splitter:

# #### RMM2

# In[ ]:


x_train,_,y_train = hilbert_data1_jgrjd_20CRV3.data_hilbert(datetime.datetime(1979,1,1),datetime.datetime(2008,12,31))
x_test,_,y_test = hilbert_data1_jgrjd_20CRV3.data_hilbert(datetime.datetime(1974,6,1),datetime.datetime(1978,3,16))
x_test2,_,y_test2 = hilbert_data1_jgrjd_20CRV3.data_hilbert(datetime.datetime(2009,1,1),datetime.datetime(2015,12,31))


# #### Historical pressure

# In[ ]:


x_test3 = hilbert_data1_jgrjd_20CRV3.data_pres(datetime.datetime(1905,1,1),datetime.datetime(2015,12,31))


# #### scale the data

# In[ ]:


sc3 = MinMaxScaler()
sc5 = MinMaxScaler()

sc5.fit(x_test3[:])

test_x3 =  sc5.transform(x_test3[:])
train_x = sc5.transform(x_train[:])
test_x  = sc5.transform(x_test[:])
test_x2  = sc5.transform(x_test2[:])


sc3.fit(y_train[:])

train_y = sc3.transform(y_train)
test_y  = sc3.transform(y_test)
test_y2  = sc3.transform(y_test2)

#train_x.max(),test_x.max(),test_x3.max(),test_x2.max(),train_y.max(),test_y.max(),test_y2.max()


# In RNN we have to choose a window. Here we choose first 120 points as predictor and next RMM value as predicted. That means RMM will be fitted using previous 120 time steps's pressure of every point

# #### split the sequence data for training

# In[ ]:


def split_sequence(window,x,*args):
    xout  = []
    for i in range(window,len(x)):
        xout.append(x[i-window:i,:])
    
    xout = np.array(xout)
    xout = np.reshape(xout,(xout.shape[0],xout.shape[1],xout.shape[2]))
        
    if np.any(len(args)):
        for y in args:
            yout = []
            for i in range(window,len(y)):
                yout.append(y[i,0])
            yout = np.array(yout)
            yout = yout.reshape(yout.shape[0])
    else:
        yout = [] 
    
    return xout,yout


# In[ ]:


window = 120
xtrain , ytrain = split_sequence(window,train_x,train_y)
xtest , ytest   = split_sequence(window,test_x,test_y)
xtest2 , ytest2 = split_sequence(window,test_x2,test_y2)
xtest3,_        = split_sequence(window, test_x3)


# #### Cut the data according to batch size

# In[ ]:


par_b =100 

#print(x_test3.shape)
te3_lc = ((len(x_test3)-window)//par_b)*par_b

xtest3 = xtest3[:te3_lc,:,:]
#print(xtest3.shape)

#x_test3.iloc[window:window+te3_lc,:].index
## THis perid data will be available


# In[ ]:


#print(xtrain.shape,ytrain.shape,xtest.shape,ytest.shape)

tr_lc = ((len(x_train)-window)//par_b)*par_b
te_lc =  ((len(x_test)-window)//par_b)*par_b
te_lc2 =  ((len(x_test2)-window)//par_b)*par_b

xtrain = xtrain[:tr_lc,:,:]
ytrain = ytrain[:tr_lc]
xtest = xtest[:te_lc,:,:]
ytest = ytest[:te_lc,]
xtest2 = xtest2[:te_lc2,:,:]
ytest2 = ytest2[:te_lc2,]

#print(xtrain.shape,ytrain.shape,xtest.shape,xtest2.shape,ytest.shape,ytest2.shape)


# ### Using Simple Convolution 1D
# * 1. Basic conv1d
# * 2. wavenet
# * 3. ENSO  paper model
# 

# In[ ]:


nf=[]
for i in range(1,4,1):
    nf.append([2**i,2**i,2**i,2**i])
    nf.append([2**(i+4),2**(i+3),2**(i+1),2**i])
    nf.append([2**(2*i+2),2**(2*i),2**(2*i-1),2**(2*i-2)])
    
kz=[]
for i in range(1,4,1):
    kz.append([2**i,2**i,2**i,2*i,2*i])
    kz.append([2**(i+3),2**(i+2),2**(i+1),2**i])

ds = [10,20,30,40,50]

lr = [0.01,0.001,0.005,0.0001]

dr = [1e-7,1e-6,1e-5]

# In[ ]:


def models(params):
    """
       Random number initializer is needed 
    """
    
    seed(4)
    rn.seed(4)
    tensorflow.random.set_seed(4)
    os.environ["PYTHONHASHSEED"] = '0'
    
    
    i = params[0]; k = params[1]; j = params[2];  l = params[3] ; d = params[4]
    print('running on iteration ' + str(i)+','+str(k)+','+str(j)+','+ str(l)+','+str(d))
    
    model = Sequential()
    # Use the Keras Conv1D function to create a 1-dimensional convolutional layer, with kernel size (filter) of 5X5 pixels and a stride of 1 in x and y directions. The Conv2D command automatically creates the activation function for you━here we use ReLu activation.

    model.add(Conv1D(nf[i][0] ,kernel_size=kz[k][0], strides=1,activation='relu',
                     input_shape=(xtrain.shape[1],xtrain.shape[2])))
    # Then use the MaxPooling2D function to add a 2D max pooling layer, with pooling filter sized 2X2 and stride of 2 in x and y directions.

    model.add(MaxPooling1D(pool_size=1, strides=1))

    model.add(Conv1D(nf[i][1], kernel_size=kz[k][1], strides=1,activation='relu'))
    # Then use the MaxPooling2D function to add a 2D max pooling layer, with pooling filter sized 2X2 and stride of 2 in x and y directions.

    model.add(MaxPooling1D(pool_size=1, strides=1))

    model.add(Conv1D(nf[i][2], kernel_size=kz[k][2], strides=1, activation='relu'))
    
    model.add(MaxPooling1D(pool_size=1, strides=1))

    model.add(Conv1D(nf[i][3], kernel_size=kz[k][3], strides=1, activation='relu'))
    
    model.add(Flatten())
    model.add(Dense(ds[j], activation='relu'))


    model.add(Dense(1, activation='linear'))
    opt = keras.optimizers.Adam(lr= lr[l], decay=dr[d])
    model.compile(loss='mae', optimizer=opt)
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0,patience=30)
    model.fit(xtrain, ytrain, validation_data=(xtest, ytest),batch_size=100, epochs=200,callbacks=[es],verbose=0)

    predict1   = model.predict(xtrain)
    yy_train   = sc3.inverse_transform(predict1)
    yy_train   = yy_train/yy_train.std()
    train_corr_ = np.corrcoef(yy_train[:,0],ytrain)[0,1]

    predict1  = model.predict(xtest2)
    yy_test1   = sc3.inverse_transform(predict1)
    yy_test1   = yy_test1/yy_test1.std()
    test1_corr_ = np.corrcoef(yy_test1[:,0],ytest2)[0,1]

    predict2  = model.predict(xtest)
    yy_test2   = sc3.inverse_transform(predict2)
    yy_test2   = yy_test2/yy_test2.std()
    test2_corr_ = np.corrcoef(yy_test2[:,0],ytest)[0,1]
    
    print('running on iteration ' + str(i)+','+str(k)+','+str(j))
    print(train_corr_,test1_corr_,test2_corr_)
    
    return i,k,j,l,d,train_corr_,test1_corr_,test2_corr_


# In[ ]:


train_corr = np.zeros((len(nf),len(kz),len(ds),len(lr),len(dr)));train_corr.fill(-1)
test1_corr = np.zeros((len(nf),len(kz),len(ds),len(lr),len(dr)));test1_corr.fill(-1)
test2_corr = np.zeros((len(nf),len(kz),len(ds),len(lr),len(dr)));test2_corr.fill(-1)



ii = range(len(nf))
kk = range(len(kz))
jj = range(len(ds))
ll = range(len(lr))
dd = range(len(dr))
          
paramlist = list(itertools.product(ii,kk,jj,ll,dd))
print(multiprocessing.cpu_count())
        
pool = multiprocessing.Pool(processes=72)

for i,k,j,l,d,train_corr_,test1_corr_,test2_corr_  in tqdm.tqdm(pool.imap_unordered(models, paramlist), total=len(paramlist)):
    train_corr[i,k,j,l,d] = train_corr_
    test1_corr[i,k,j,l,d] = test1_corr_
    test2_corr[i,k,j,l,d] = test2_corr_ 
  


# In[ ]:


np.save('train_rmm2_job_layers.npy',train_corr)
np.save('test1_rmm2_job_layers.npy',test1_corr)
np.save('test2_rmm2_job_layers.npy',test2_corr)
