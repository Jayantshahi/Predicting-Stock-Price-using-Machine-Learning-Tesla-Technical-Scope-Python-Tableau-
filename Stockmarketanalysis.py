#!/usr/bin/env python
# coding: utf-8

# In[61]:


from google.colab import drive
drive.mount('/content/drive')


# In[62]:


from sklearn import datasets
import numpy as np 
import pandas as pd 
from subprocess import check_output
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


# In[63]:


Tsla_us = pd.read_csv("/content/drive/My Drive/Tsla_us.csv")
Tsla_us.head()


# In[64]:


print(Tsla_us.head())
print(Tsla_us.shape)
print(Tsla_us.columns)


# In[65]:


Tsla_us[['Close']].plot()
plt.title("Tesla")
plt.show()


# In[66]:


plt.figure(figsize=(10,10))
lag_plot(Tsla_us['Close'], lag=5)
plt.title('Autocorrelation plot for Tesla ')


# In[45]:


## ARIMA (AutoRegressive Integrated Moving Average) for Time Series Prediction


# In[76]:


training_data, testing_data = Tsla_us[0:int(len(Tsla_us)*0.7)], Tsla_us[int(len(Tsla_us)*0.7):]
plt.figure(figsize=(12,7))
plt.title('Closing Prices for Tesla')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.plot(Tsla_us['Close'], 'orange', label='Training Data')
plt.plot(testing_data['Close'], 'yellow', label='Testing Data')
plt.xticks(np.arange(0,1857, 300), Tsla_us['Date'][0:1857:300])
plt.legend()


# In[68]:


training_ar_m = training_data['Close'].values
testing_ar_m = testing_data['Close'].values

Past_values = [x for x in training_ar_m]
predictions = list()
t=0
while t < len(testing_ar_m):
    model = ARIMA(Past_values, order=(5,1,0))
    model_fit = model.fit(disp=0)
    result = model_fit.forecast()
    y_h = result[0]
    predictions.append(y_h)
    obs = testing_ar_m[t]
    Past_values.append(obs)
    t=t+1

error = mean_squared_error(testing_ar_m, predictions)
print('MSE error for testing: %.3f' % error)


# In[74]:


plt.figure(figsize=(12,7))
plt.plot(Tsla_us['Close'], 'green', color='orange', label='Training Data')
plt.plot(testing_data.index, predictions, color='blue', marker='o', linestyle='dashed', 
         label='Predicted Price')
plt.plot(testing_data.index, testing_data['Close'], color='red', label='Actual Price')
plt.title('Closing Prices Prediction for Tesla')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.xticks(np.arange(0,1857, 300), Tsla_us['Date'][0:1857:300])
plt.legend()


# In[75]:


plt.figure(figsize=(12,7))
plt.plot(testing_data.index, predictions, color='blue', marker='o', linestyle='dashed', 
         label='Predicted Price')
plt.plot(testing_data.index, testing_data['Close'], color='red', label='Actual Price')
plt.xticks(np.arange(1486,1856, 60), Tsla_us['Date'][1486:1856:60])
plt.title('Closing Prices Prediction for Tesla')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.legend()

