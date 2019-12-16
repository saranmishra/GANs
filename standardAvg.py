from data import Stock_Data
import numpy as  np 
#import pandas as pd 
#import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import matplotlib.pyplot as plt


# This code will set the standard avg

#path = os.getcwd() 

#print(path)

#Build LSTM 

class StandardAvg():
    
    high_prices = Stock_Data.stock_df.loc[:,'high'].as_matrix()
    low_prices = Stock_Data.stock_df.loc[:,'low'].as_matrix()
    mid_prices = (high_prices+low_prices)/2.0

    
#Train and test data    
    train_data = mid_prices[:2517] 
    test_data = mid_prices[2517:]
    print(train_data)
    
# Scale the data to be between 0 and 1
# When scaling remember! You normalize both test and train data with respect to training data
# Because you are not supposed to have access to test data
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1,1)
    test_data = test_data.reshape(-1,1)
    print(test_data)
    
# Train the Scaler with training data and smooth data
    smoothing_window_size = 1258
    for di in range(0,1000,smoothing_window_size):
        scaler.fit(train_data[di:di+smoothing_window_size,:])
        train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

# You normalize the last bit of remaining data
    scaler.fit(train_data[di+smoothing_window_size:,:])
    train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])
    
    # Reshape both train and test data
    train_data = train_data.reshape(-1)

# Normalize test data
    test_data = scaler.transform(test_data).reshape(-1)
# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
    EMA = 0.0
    gamma = 0.1

    for ti in range(2517):
      EMA = gamma*train_data[ti] + (1-gamma)*EMA
      train_data[ti] = EMA

# Used for visualization and test purposes
    all_mid_data = np.concatenate([train_data,test_data],axis=0)
    
# Calculate Standard Average
    window_size = 100
    N = train_data.size
    std_avg_predictions = []
    std_avg_x = []
    mse_errors = []
    
    for pred_idx in range(window_size,N):
    
        if pred_idx >= N:
            date = dt.datetime.strptime(Stock_Data.d, '%Y-%m-%d').date() + dt.timedelta(days=1)
        else:
            date =  Stock_Data.stock_df.loc[pred_idx,'date']
    
        std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
        mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
        std_avg_x.append(date)

    print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))

#For plotting    
#    plt.figure(figsize = (16,9))
#    plt.plot(range(Stock_Data.stock_df.shape[0]),all_mid_data,color='b',label='True')
#    plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
#    #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
#    plt.xlabel('Date')
#    plt.ylabel('Mid Price')
#    plt.legend(fontsize=18)
#    plt.show()
