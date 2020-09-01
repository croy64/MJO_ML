#!/usr/bin/env python
# coding: utf-8

# ## Multiple linear regression using Hilbert transform

# Written by Abirlal Metya, Panini Dasgupta, Manmeet Singh (25/12/2019)

# import modules

# In[ ]:



def lowpass_scipy(signal,sample_freq,time_period,keep_mean):
    import numpy as np
    import scipy as sc
    from scipy import fftpack
    
    lowpass_signal=np.zeros(signal.shape)
    if any(np.isnan(signal)):
        raise ValueError('There is NaN in the signal')
    else:
        hf = 1./time_period

        temp_fft = sc.fftpack.fft(signal)

        fftfreq = np.fft.fftfreq(len(signal),sample_freq) ### daily data it is 1./365 ## monthly data 1./12 ## yearly data=1
          
        i1 = np.abs(fftfreq) >= hf  
        
        temp_fft[i1] = 0
        if not(keep_mean):
            temp_fft[0]=0
        lowpass_signal= np.real_if_close(sc.fftpack.ifft(temp_fft))
    
    return lowpass_signal


# In[ ]:


def data_hilbert(st,en):

    import pandas as pd
    import numpy as np
    from scipy.signal import hilbert, chirp

   ## Read Data

    df       = pd.read_csv('/home/cccr/supriyo/panini/filtered_data/historical/full_data_pressure.txt',index_col='date')
    df.index = pd.to_datetime(df.index)
    df=df[(df.index>=st) & (df.index<=en)]
    
    
    df2       = pd.read_csv('/home/cccr/supriyo/panini/filtered_data/historical/full_data_nn.txt',index_col='date')
    df2.index = pd.to_datetime(df2.index)
    df2=df2[(df2.index>=st) & (df2.index<=en)]

    #df2=df2[df2.index.year<2009]
    
    
    X = df.iloc[:,:12]
    #y = df.iloc[:,12:13]
    
    def runing_mean(ddt,window=5):
        import datetime

        run  = pd.DataFrame([])
        st   = ddt.index[0]
        #print(ddt.index[-1])
        for i in range(1,100000):
            if (st <= ddt.index[-window]):

                dt   = ddt[(ddt.index >= st) & (ddt.index  < st+datetime.timedelta(days=window-1))]

                dtt  = pd.DataFrame(dt.mean()).T
                #dtt['Datetime'] = st+datetime.timedelta(days=np.round(window/2,0)) 
                dtt['Datetime'] = st+datetime.timedelta(days= int(window/2.0))
                dtt.set_index('Datetime', inplace=True)
                run  = run.append(dtt)
                st   = st + datetime.timedelta(days=1)
                #print(i,st)
        return run

    def rm_run_mean(ddt,window=5):
        import datetime

        run  = pd.DataFrame([])
        st   = ddt.index[0]
        #print(ddt.index[-1])
        for i in range(1,100000):
            if (st <= ddt.index[-window]):

                dt   = ddt[(ddt.index >= st) & (ddt.index  < st+datetime.timedelta(days=window-1))]
                dg   = dt.reset_index()
                dg   = dg.iloc[:,1:] 
                anom = dg[-1:].values - pd.DataFrame(dt.mean()).T.values
                #print(pd.DataFrame(dt.mean()).T.values)
                #dtt  = pd.DataFrame(dt.iloc[-1,:]) - pd.DataFrame(dt.mean()).T
                dtt = pd.DataFrame(anom)
                #dtt['Datetime'] = st+datetime.timedelta(days=np.round(window/2,0)) 
                dtt['Datetime'] = st+datetime.timedelta(days= window-1)
                dtt.set_index('Datetime', inplace=True)
                run  = run.append(dtt)
                st   = st + datetime.timedelta(days=1)
                #print(i,st)

        return run

    X              = rm_run_mean(X,120)
#     X1 = runing_mean(X,11)

    ## 10 days lowpass filter #######
    
    
    X1 = X.copy()
    lf = 20;sample_freq = 1;keep_mean = 1
    for i in range(X.shape[1]):
        signal = X.iloc[:,i].values
        temp = lowpass_scipy(signal,sample_freq,lf,keep_mean)
        X1.iloc[:,i] = np.real(temp)
#     #####################################
    
    hilbertx       = pd.DataFrame(np.imag(hilbert(X1)))
    hilbertx.index = X1.index
    X2              = pd.concat([X1,hilbertx],axis=1)
    
    
    ################ RMM1 #########################
    y = df2.iloc[:,12:13]
#     y1 = runing_mean(y,11)

#    ### 10 days lowpass #############
    y1 = y.copy()
    for i in range(y.shape[1]):
        signal = y.iloc[:,i].values
        temp = lowpass_scipy(signal,sample_freq,lf,keep_mean)
        y1.iloc[:,i] = np.real(temp)
        
    RMM1 = y1.iloc[119:]
    del y,y1 

    ###################################    
        

    
    
    
     ################ RMM2 #########################
    y = df2.iloc[:,13:14]
#     y1 = runing_mean(y,11)

#     ### 10 days lowpass #############
    y1 = y.copy()
    for i in range(y.shape[1]):
        signal = y.iloc[:,i].values
        temp = lowpass_scipy(signal,sample_freq,lf,keep_mean)
        y1.iloc[:,i] = np.real(temp)
    RMM2 = y1.iloc[119:]
    del y,y1 
#     ###################################    
       
    

    
    return X2,RMM1,RMM2


# In[ ]:


def data_pres(st,en):

    import pandas as pd
    import numpy as np
    from scipy.signal import hilbert, chirp

   ## Read Data

    df       = pd.read_csv('/home/cccr/supriyo/panini/filtered_data/historical/full_data_pressure.txt',index_col='date')
    df.index = pd.to_datetime(df.index)
    df=df[(df.index>=st) & (df.index<=en)]
    
    
    
    
    
    X = df.iloc[:,:12]
    
    
    def runing_mean(ddt,window=5):
        import datetime

        run  = pd.DataFrame([])
        st   = ddt.index[0]
        #print(ddt.index[-1])
        for i in range(1,100000):
            if (st <= ddt.index[-window]):

                dt   = ddt[(ddt.index >= st) & (ddt.index  < st+datetime.timedelta(days=window-1))]

                dtt  = pd.DataFrame(dt.mean()).T
                #dtt['Datetime'] = st+datetime.timedelta(days=np.round(window/2,0)) 
                dtt['Datetime'] = st+datetime.timedelta(days= int(window/2.0))
                dtt.set_index('Datetime', inplace=True)
                run  = run.append(dtt)
                st   = st + datetime.timedelta(days=1)
                #print(i,st)
        return run

    def rm_run_mean(ddt,window=5):
        import datetime

        run  = pd.DataFrame([])
        st   = ddt.index[0]
        #print(ddt.index[-1])
        for i in range(1,100000):
            if (st <= ddt.index[-window]):

                dt   = ddt[(ddt.index >= st) & (ddt.index  < st+datetime.timedelta(days=window-1))]
                dg   = dt.reset_index()
                dg   = dg.iloc[:,1:] 
                anom = dg[-1:].values - pd.DataFrame(dt.mean()).T.values
                #print(pd.DataFrame(dt.mean()).T.values)
                #dtt  = pd.DataFrame(dt.iloc[-1,:]) - pd.DataFrame(dt.mean()).T
                dtt = pd.DataFrame(anom)
                #dtt['Datetime'] = st+datetime.timedelta(days=np.round(window/2,0)) 
                dtt['Datetime'] = st+datetime.timedelta(days= window-1)
                dtt.set_index('Datetime', inplace=True)
                run  = run.append(dtt)
                st   = st + datetime.timedelta(days=1)
                #print(i,st)

        return run

    X              = rm_run_mean(X,120)
#     X1 = runing_mean(X,11)

    ## 10 days lowpass filter #######
    
    
    X1 = X.copy()
    lf = 20;sample_freq = 1;keep_mean = 1
    for i in range(X.shape[1]):
        signal = X.iloc[:,i].values
        temp = lowpass_scipy(signal,sample_freq,lf,keep_mean)
        X1.iloc[:,i] = np.real(temp)
#     #####################################
    
    hilbertx       = pd.DataFrame(np.imag(hilbert(X1)))
    hilbertx.index = X1.index
    X2              = pd.concat([X1,hilbertx],axis=1)
    
    
    
    
    return X2


# In[ ]:



# import matplotlib.pyplot as plt
# a=np.random.random(366)
# temp_fft = sc.fftpack.fft(a)
# a1=lowpass_scipy(a,1,1,1)
# plt.plot(a)
# plt.plot(a1)
# a1


# In[ ]:


# import matplotlib.pyplot as plt
# a=np.random.random(730)
# temp_fft = sc.fftpack.fft(a[0:365])
# fftfreq = np.fft.fftfreq(len(a[0:365]),1) ### daily data it is 1./365 ## monthly data 1./12 ## yearly data=1

# a1=lowpass_scipy(a,1,1,1)
# plt.plot(a)
# plt.plot(a1)

# a11=lowpass_scipy(a[0:365],1,10,1)
# a22=lowpass_scipy(a[365:],1,10,1)
# a33=lowpass_scipy(a,1,10,1)
# plt.figure(figsize=(16,8))
# plt.plot(a11)
# plt.plot(a33)
# plt.plot(np.arange(365,730,1),a22)


# In[ ]:




