
#!/usr/bin/env python
# coding: utf-8

# ## Multiple linear regression using Hilbert transform

# Written by Abirlal Metya, Panini Dasgupta, Manmeet Singh (25/12/2019)

# import modules

# In[ ]:


# EXAMPLE of butterwidth filter
import numpy as np
import scipy as sc
from scipy import signal
from scipy import fftpack

import math    

def filter_signal_scipy(signal,sample_freq,ltime_period,htime_period,keep_mean):
    filter_signal=np.zeros(signal.shape)
    if any(np.isnan(signal)):
        raise ValueError('There is NaN in the signal')
    else:
        hf=1./ltime_period
        lf=1./htime_period

        temp_fft = sc.fftpack.fft(signal)

        fftfreq = np.fft.fftfreq(len(signal),sample_freq) ### daily data it is 1./365 ## monthly data 1./12 ## yearly data=1
          
        i1=(np.abs(fftfreq) >= lf) & (np.abs(fftfreq) <= hf)  
        inv_fft=np.zeros(temp_fft.size,dtype=complex)
        inv_fft[i1]=temp_fft[i1]
        if keep_mean:
            inv_fft[0]=temp_fft[0]
        filter_signal= np.real_if_close(sc.fftpack.ifft(inv_fft))
    
    return filter_signal

def lowpass_scipy_butter(signal1,wn,lt):
    from scipy import signal
    w = 2/lt # Normalize the frequency
    b, a = signal.butter(wn, w, 'low')
    lowpass_signal = signal.filtfilt(b, a, signal1)

    return lowpass_signal


# In[ ]:


def data_hilbert(st,en):

    import pandas as pd
    import numpy as np
    from scipy.signal import hilbert, chirp

   ## Read Data

    df       = pd.read_csv('full_data_pressure_20CRV3.txt',index_col='date')
    df.index = pd.to_datetime(df.index)
    df=df[(df.index>=st) & (df.index<=en)]
    
    
    df2       = pd.read_csv('full_data_nn_20CR_V3.txt',index_col='date')
    df2.index = pd.to_datetime(df2.index)
    df2=df2[(df2.index>=st) & (df2.index<=en)]

    
    
    
    X = df.iloc[:,:12]
    
    def runing_mean(ddt,window=5):
        import datetime

        run  = pd.DataFrame([])
        st   = ddt.index[0]

        for i in range(1,100000):
            if (st <= ddt.index[-window]):

                dt   = ddt[(ddt.index >= st) & (ddt.index  < st+datetime.timedelta(days=window-1))]

                dtt  = pd.DataFrame(dt.mean()).T
                
                dtt['Datetime'] = st+datetime.timedelta(days= int(window/2.0))
                dtt.set_index('Datetime', inplace=True)
                run  = run.append(dtt)
                st   = st + datetime.timedelta(days=1)
                
        return run

    def rm_run_mean(ddt,window=5):
        import datetime

        run  = pd.DataFrame([])
        st   = ddt.index[0]

        for i in range(1,100000):
            if (st <= ddt.index[-window]):

                dt   = ddt[(ddt.index >= st) & (ddt.index  < st+datetime.timedelta(days=window-1))]
                dg   = dt.reset_index()
                dg   = dg.iloc[:,1:] 
                anom = dg[-1:].values - pd.DataFrame(dt.mean()).T.values
                
                dtt = pd.DataFrame(anom)

                dtt['Datetime'] = st+datetime.timedelta(days= window-1)
                dtt.set_index('Datetime', inplace=True)
                run  = run.append(dtt)
                st   = st + datetime.timedelta(days=1)

        return run

    
    X2 = X.copy()
    for i in range(X2.shape[1]):
        signal = X2.iloc[:,i].values
        temp = filter_signal_scipy(signal,sample_freq=1,ltime_period=20,htime_period=100,keep_mean=1)
        X2.iloc[:,i] = np.real(temp)
    
    X              = rm_run_mean(X,120)

    ## 10 days lowpass filter #######
    
    
    X1 = X.copy()
    lf = 10;wn = 3
    for i in range(X.shape[1]):
        signal = X.iloc[:,i].values
        temp = lowpass_scipy_butter(signal,wn,lf)
        
        X1.iloc[:,i] = np.real(temp)
#     #####################################
    
    hilbertx       = pd.DataFrame(np.imag(hilbert(X1,axis=0)))
    hilbertx.index = X1.index
    
   #######################################################
    

    X3              = pd.concat([X1,hilbertx,X2.iloc[119:]],axis=1)

    ################ RMM1 #########################
    y = df2.iloc[:,12:13]

#    ### 10 days lowpass #############
    y1 = y.copy()
    for i in range(y.shape[1]):
        signal = y.iloc[:,i].values
        temp = lowpass_scipy_butter(signal,wn,lf)
        y1.iloc[:,i] = np.real(temp)
        
    RMM1 = y1.iloc[119:]
    del y,y1 

    ###################################    
        

    
    
    
     ################ RMM2 #########################
    y = df2.iloc[:,13:14]

#     ### 10 days lowpass #############
    y1 = y.copy()
    for i in range(y.shape[1]):
        signal = y.iloc[:,i].values
        temp = lowpass_scipy_butter(signal,wn,lf)
        y1.iloc[:,i] = np.real(temp)
    RMM2 = y1.iloc[119:]
    del y,y1 
#     ###################################    
       
    

    
    return X3,RMM1,RMM2


# In[ ]:


def data_pres(st,en):

    import pandas as pd
    import numpy as np
    from scipy.signal import hilbert, chirp

   ## Read Data

    df       = pd.read_csv('full_data_pressure_20CRV3.txt',index_col='date')
    df.index = pd.to_datetime(df.index)
    df=df[(df.index>=st) & (df.index<=en)]
    
    
    
    
    
    X = df.iloc[:,:12]
    
    
    def runing_mean(ddt,window=5):
        import datetime

        run  = pd.DataFrame([])
        st   = ddt.index[0]

        for i in range(1,100000):
            if (st <= ddt.index[-window]):

                dt   = ddt[(ddt.index >= st) & (ddt.index  < st+datetime.timedelta(days=window-1))]

                dtt  = pd.DataFrame(dt.mean()).T
                dtt['Datetime'] = st+datetime.timedelta(days= int(window/2.0))
                dtt.set_index('Datetime', inplace=True)
                run  = run.append(dtt)
                st   = st + datetime.timedelta(days=1)
                
        return run

    def rm_run_mean(ddt,window=5):
        import datetime

        run  = pd.DataFrame([])
        st   = ddt.index[0]

        for i in range(1,100000):
            if (st <= ddt.index[-window]):

                dt   = ddt[(ddt.index >= st) & (ddt.index  < st+datetime.timedelta(days=window-1))]
                dg   = dt.reset_index()
                dg   = dg.iloc[:,1:] 
                anom = dg[-1:].values - pd.DataFrame(dt.mean()).T.values
               
                dtt = pd.DataFrame(anom)

                dtt['Datetime'] = st+datetime.timedelta(days= window-1)
                dtt.set_index('Datetime', inplace=True)
                run  = run.append(dtt)
                st   = st + datetime.timedelta(days=1)

        return run

                                 
    X2 = X.copy()
    for i in range(X2.shape[1]):
        signal = X2.iloc[:,i].values
        temp = filter_signal_scipy(signal,sample_freq=1,ltime_period=20,htime_period=100,keep_mean=1)
        X2.iloc[:,i] = np.real(temp)
                             
    X              = rm_run_mean(X,120)


    ## 10 days lowpass filter #######
    
    
    X1 = X.copy()
    lf = 10;wn = 3
    for i in range(X.shape[1]):
        signal = X.iloc[:,i].values
        temp = lowpass_scipy_butter(signal,wn,lf)
        
        X1.iloc[:,i] = np.real(temp)
#     #####################################
    
    hilbertx       = pd.DataFrame(np.imag(hilbert(X1,axis=0)))
    hilbertx.index = X1.index
    
   #######################################################
   
    X3              = pd.concat([X1,hilbertx,X2.iloc[119:,]],axis=1)
    
    
    return X3


# In[ ]:


## EXAMPLE of butterwidth filter


# def lowpass_scipy_butter(signal1,wn,lt):
#     from scipy import signal
#     w = 2/lt # Normalize the frequency
#     b, a = signal.butter(wn, w, 'low')
#     lowpass_signal = signal.filtfilt(b, a, signal1)

#     return lowpass_signal

# def lowpass_scipy(signal,sample_freq,time_period,keep_mean):
#     import numpy as np
#     import scipy as sc
#     from scipy import fftpack
    
#     lowpass_signal=np.zeros(signal.shape)
#     if any(np.isnan(signal)):
#         raise ValueError('There is NaN in the signal')
#     else:
#         hf = 1./time_period

#         temp_fft = sc.fftpack.fft(signal)

#         fftfreq = np.fft.fftfreq(len(signal),sample_freq) ### daily data it is 1./365 ## monthly data 1./12 ## yearly data=1
          
#         i1 = np.abs(fftfreq) > hf  
        
#         temp_fft[i1] = 0
#         if not(keep_mean):
#             temp_fft[0]=0
#         lowpass_signal= np.real_if_close(sc.fftpack.ifft(temp_fft))
    
#     return lowpass_signal


# In[ ]:


# import numpy as np
# import matplotlib.pyplot as plt

# fs = 1000  # Sampling frequency
# # Generate the time vector properly
# t = np.arange(1000) / fs

# signala = np.sin(2*np.pi*100*t)
# signald = np.sin(2*np.pi*200*t) # frequency 20
# # with frequency of 100
# signalb = np.sin(2*np.pi*5*t) # frequency 20
# signalc = signala + signalb +signald

# plt.figure(figsize=(15,4))
# output = lowpass_scipy_butter(signalc,3,300)
# lowpass = lowpass_scipy(signalc,1,300,1)
# plt.plot(signalc[0:400])
# plt.plot(output[0:400])
# plt.plot(lowpass[0:400])


# In[ ]:





# In[ ]:




