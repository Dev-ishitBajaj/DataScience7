#!/usr/bin/env python
# coding: utf-8

# # Krishna Sharma
# ## 101903755
# ## Building Innovative Systems - Prediction in time series dataset

# ### Libraries Import

# In[63]:


import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import math
from sklearn.ensemble import RandomForestRegressor


# ### Dataset Import 

# In[4]:


dataset = pd.read_excel('DATASET.xlsx')


# In[5]:


dataset


# In[6]:


j = 0
for i in range(100):
    dataset.iloc[j:j + 10, 0] = dataset.iloc[j, 0]
    j += 10


# ### Data Preprocessing

# In[8]:


dataset.columns = dataset.columns.str.replace('Unnamed: 0', 'Group')


# In[9]:


dataset.fillna(0, inplace=True)


# In[10]:


test_dataset = dataset.loc[dataset['year'] == 10]
test_dataset


# In[11]:


test_dataset.columns = test_dataset.columns.str.replace('Unnamed: 0', 'Group')


# In[12]:


dataset.head(12)


# In[ ]:





# ## Model Definition 

# In[39]:


etr = ExtraTreesRegressor(n_estimators=200,bootstrap=False, criterion='mse', max_depth=None,
                    max_features='auto', max_leaf_nodes=1000,
                    min_impurity_decrease=0.0, 
                    min_samples_leaf=1,
                    min_weight_fraction_leaf=0.1,
                    n_jobs=None, oob_score=False, random_state=123, verbose=0,
                    warm_start=False)


# In[40]:


predict_col_names = ['Para-9', 'Para-10', 'Para-11', 'Para-12', 'Para-13']


# In[41]:


result = {'Para-9': [], 'Para-10': [], 'Para-11': [], 'Para-12': [], 'Para-13': []}


# In[64]:


rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)


# ## Model Application

# In[76]:


def selectedModel(val):
    RMSE = []
    N = 0
    index = 0
    if(val==0):
        for j in [10, 11, 12, 13, 14]:
            ans = 0
            count = 0
            for i in range(0, 1000, 10):
                xtrain = dataset.iloc[i:i+9, 1:10]
                ytrain = dataset.iloc[i:i+9, j]
                xtest = dataset.iloc[i+9, 1:10].to_numpy()
                ytest = dataset.iloc[i+9, j]
                etr.fit(xtrain, ytrain)
                ypred = etr.predict(xtest.reshape(1,-1))[0]
                result[predict_col_names[index]].append(ypred)
                ans += (ytest - ypred)**2
                count += 1
                N += 1

            RMSE.append(ans)
            index += 1  
    else:
        for j in [10, 11, 12, 13, 14]:
            ans = 0
            count = 0
            for i in range(0, 1000, 10):
                xtrain = dataset.iloc[i:i+9, 1:10]
                ytrain = dataset.iloc[i:i+9, j]
                xtest = dataset.iloc[i+9, 1:10].to_numpy()
                ytest = dataset.iloc[i+9, j]
                rf.fit(xtrain, ytrain)
                ypred = rf.predict(xtest.reshape(1,-1))[0]
                result[predict_col_names[index]].append(ypred)
                ans += (ytest - ypred)**2
                count += 1
                N += 1

            RMSE.append(ans)
            index += 1
    return (RMSE, count)


# In[77]:


result_RMSE, result_count = selectedModel(1)


# In[78]:


dataset_predicted = pd.DataFrame.from_dict(result)
dataset_predicted


# ## RMSE Calculations- 

# In[79]:


for i in result_RMSE:
    print((i/result_count)**0.5)


# In[80]:


result_RMSE


# In[81]:


#Total RMSE
print((sum(result_RMSE)/500)**0.5)


# ## Result Plots 

# In[82]:



import random
ind = []
for i in range(40):
    ind.append(random.randint(0,100))


# In[83]:



for i in range(5):
    plt.figure(figsize=(6,6))
    true_value = test_dataset.iloc[ind, i+10]
    predicted_value = dataset_predicted.iloc[ind, i]
    plt.scatter(true_value, predicted_value, c='crimson')
    # plt.yscale('log')
    # plt.xscale('log')

    p1 = max(max(predicted_value), max(true_value))
    p2 = min(min(predicted_value), min(true_value))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.title(predict_col_names[i])
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




