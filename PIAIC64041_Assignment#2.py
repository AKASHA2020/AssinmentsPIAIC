#!/usr/bin/env python
# coding: utf-8

# # Numpy_Assignment_2::
# 
# **PIAIC64041**
# 
# **NAME AKASHA MAHFOOZ**

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[1]:


import numpy as np


# In[9]:


arr = np.arange(10).reshape(2,5)
arr


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[10]:


arr1 = np.arange(10).reshape(2,5)
arr2 = np.ones(10,dtype=int).reshape(2,5)


# In[11]:


np.vstack((arr1,arr2))


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
# In[12]:


arr1 = np.arange(10).reshape(2,5)
arr2 = np.ones(10,dtype=int).reshape(2,5)


# In[13]:


np.hstack((arr1,arr2))


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[16]:


flat = np.array([[0,1,2,3,4],[5,6,7,8,9]])
flat


# In[17]:


flat.flatten()


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[24]:


arr = np.arange(15).reshape(1,3,5)
arr


# In[26]:


arr  = arr.flatten()
arr


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[27]:


arr = np.arange(15)
arr


# In[28]:


arr.reshape(-1,3)


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[29]:


arr = np.arange(25).reshape(5,5)
arr


# In[30]:


np.square(arr)


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[31]:


arr = np.random.randint(30,size=(5,6))
arr


# In[32]:


arr.mean()


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[34]:


np.random.seed(0)
arr = np.random.randint(30,size=(5,6))
print(arr)
np.std(arr)


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[35]:


np.random.seed(0)
arr = np.random.randint(30,size=(5,6))
print(arr)
np.median(arr)


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[38]:


np.random.seed(0)
arr = np.random.randint(30,size=(5,6))
print(arr)
arr.transpose()


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[39]:


arr =  np.arange(16).reshape(4,4)
arr


# In[40]:


np.diagonal(arr)


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[41]:


arr =  np.arange(16).reshape(4,4)
arr


# In[42]:


np.linalg.det(arr)


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[45]:


a = np.array([1,2,3,4,5,6,7,8,9,10])
a


# In[46]:


print("Find the percentile  5th")
np.percentile(a,5)


# In[48]:


print("Find the percentile  95th")

np.percentile(a,95)


# ## Question:15

# ### How to find if a given array has any null values?

# In[ ]:




