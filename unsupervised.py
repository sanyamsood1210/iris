#!/usr/bin/env python
# coding: utf-8

# Sanyam Sood
# GRIP @The sparks foundation

# Prediction Using Unsupervised learning

# In[1]:


#inporting libraries
import numpy as np
import pandas as pd
import os


# In[2]:


import matplotlib.pyplot as plt


# In[6]:


#importing file
df=pd.read_csv(r'Downloads/IRIS.csv')


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


#for label encoding
df['species']=df['species'].map({'Iris-setosa':0,'Iris-df['species']=df['species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})':1,'Iris-virginica':2})


# In[10]:


df.info()


# In[11]:


df.isnull().sum()


# In[12]:


df.describe()


# In[13]:


df.corr()


# In[14]:


import seaborn as sns


# In[15]:


sns.pairplot(df,hue='species')


# In[16]:


plt.figure(figsize=(15,6))
sns.scatterplot(data=df)


# In[17]:


plt.figure(figsize=(15,6))
sns.boxplot(data=df)


# In[18]:


x=df


# In[20]:


#elbow method k=4
from sklearn.cluster import KMeans
model=KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0)
model.fit(x)
y_pred=model.predict(x)


# In[21]:


labels=model.labels_
centroid=model.cluster_centers_


# In[27]:


y_pred
x=np.array(x)


# In[28]:


plt.figure(figsize=(15,5))
plt.scatter(x[y_pred ==0,0],x[y_pred ==0,1],s=100,c='red',label='Iris-setosa')
plt.scatter(x[y_pred == 1,0],x[y_pred==1,1],s=100,c='blue',label='Iris-versicolor')
plt.scatter(x[y_pred == 2,0],x[y_pred == 2,1],s=100,c='green',label='Iris-virginica')
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],s=100,c='yellow',label='Centroid')
plt.legend()
   


# In[ ]:




