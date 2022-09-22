#!/usr/bin/env python
# coding: utf-8

# # Recommendation System

# Problem statement.
# 
# Build a recommender system by using cosine similarities score.
# 

# In[46]:


import pandas as pd
import numpy as np


# In[9]:


df = pd.read_csv('bookr.csv',encoding = 'latin-1')
df.head()


# In[10]:


df.shape


# In[19]:


#dropping index
df1 = df.iloc[:,1:]


# In[22]:


df1.describe()


# In[23]:


df1.info()


# In[24]:


df1.isna().sum()


# In[27]:


df1[df1.duplicated()]


# In[39]:


#renaming column names
df1.rename(columns = {'User.ID':'UID', 'Book.Title':'Title','Book.Rating':'Rating'}, inplace = True)


# In[40]:


df1


# In[41]:


#drop duplicates
df1.drop_duplicates()


# In[42]:


len(df1.Title.unique())


# In[51]:


len(df1.UID.unique())


# In[49]:


user_df = df1.pivot_table(index='UID',columns='Title',values='Rating').reset_index(drop=True)


# In[50]:


user_df


# In[52]:


user_df.index = df1.UID.unique()


# In[53]:


user_df


# In[54]:


#Impute NaNs with 0 values
user_df.fillna(0, inplace=True)


# In[57]:


user_df


# In[56]:


#Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[58]:


#Subtratcting cosine value from1 to get similarity
user_sim = 1 - pairwise_distances(user_df.values,metric='cosine')


# In[59]:


user_sim


# In[60]:


user_sim_df = pd.DataFrame(user_sim)


# In[61]:


user_sim_df


# In[62]:


#Set the index and column names to user ids 
user_sim_df.index = df1.UID.unique()
user_sim_df.columns = df1.UID.unique()


# In[63]:


user_sim_df


# In[64]:


user_sim_df.iloc[0:10,0:10]


# In[66]:


#fill diagonal values with 0
np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5, 0:5]


# In[67]:


#Most Similar Users
user_sim_df.idxmax(axis=1)[0:5]


# In[72]:


#displaying similar books rated by users
df1[(df1["UID"]== 276737) | (df1["UID"]== 276726)]


# ### Recommendation For user

# In[74]:


user_1 = df1[df1['UID']== 276729]
user_1 


# In[76]:


user_2 = df1[df1['UID']== 276726]
user_2


# In[80]:


#Creating data frame for recommendation of books to user.
pd.merge(user_1,user_2,on='Title',how='outer')


# In[ ]:





# In[ ]:




