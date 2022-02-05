#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("C:/Users/amans/Downloads/Boston.csv")


# In[3]:


data.columns


# In[4]:


data.head()


# In[5]:


data.describe()


# In[6]:


data.shape


# In[7]:


data = data.drop(['Unnamed: 0'],axis=1)


# In[8]:


data.shape


# In[9]:


data.head()


# In[10]:


X = data.drop(['medv'],axis=1)
Y = data['medv']


# In[11]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)


# In[12]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[13]:


from sklearn.linear_model import LinearRegression
housing = LinearRegression()


# In[14]:


housing.fit(X_train,Y_train)


# In[15]:


from sklearn import metrics
y_pred = housing.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))


# In[16]:


print(y_pred)


# In[17]:


import pickle


# In[18]:


pickle_final=open("housing.pkl","wb")


# In[19]:


pickle.dump(housing,pickle_final)


# In[20]:


pickle_final.close()

