#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**
# **PIAIC64041**
# 
# **NAME AKASHA MAHFOOZ**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[1]:


import numpy as np


# 2. Create a null vector of size 10 

# In[2]:


z = np.zeros(10)
print(z)


# 3. Create a vector with values ranging from 10 to 49

# In[3]:


vect = np.arange(10,49)
vect


# 4. Find the shape of previous array in question 3

# In[4]:


print(vect.shape)


# 5. Print the type of the previous array in question 3

# In[5]:


type(vect)


# 6. Print the numpy version and the configuration
# 

# In[6]:


print(np.__version__)
print(np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[7]:


vect.ndim


# 8. Create a boolean array with all the True values

# In[8]:


np.ones(6, dtype=bool)


# 9. Create a two dimensional array
# 
# 
# 

# In[9]:


a = np.ones((2,2))
a


# In[10]:


a.ndim


# 10. Create a three dimensional array
# 
# 

# In[11]:


array_2d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14,15])
array_re_3d = array_2d.reshape (1,3,5) 
array_re_3d 


# In[12]:


array_re_3d.ndim


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[13]:


r = np.arange(5,15)
print(r)


# In[14]:


r = r[::-1]
print(r)


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[15]:


first=np.zeros(10)
first[4]=1
print(first)


# 13. Create a 3x3 identity matrix

# In[16]:


matrix = np.identity(3)
matrix


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[22]:


arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(arr.dtype)


# In[23]:


arr = arr.astype('float64') 
print(arr)
print(arr.dtype)


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[20]:


arr1 = np.array([[1., 2., 3.],[4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])
Multiply = arr1*arr2
Multiply


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[25]:


arr1 = np.array([[1., 2., 3.],[4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])
Max = np.maximum(arr1,arr2) 
Max


# 17. Extract all odd numbers from arr with values(0-9)

# In[26]:


arr = np.arange(0,10)
arr


# In[27]:


arr[arr % 2==1]


# 18. Replace all odd numbers to -1 from previous array

# In[28]:


arr = np.arange(0,10)
arr


# In[29]:


arr[1::2]
arr


# In[32]:


arr[1::2] = -1
arr


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[42]:


arr = np.arange(10)
arr


# In[43]:


arr[5:9] = 12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[45]:


arr = np.ones((4,4))
print("Create a 2d array")
print(arr)


# In[46]:


print("1 on the border and 0 inside in the array")
arr[1:-1,1:-1] = 0
print(arr)


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[47]:


arr2d = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
arr2d[1,1]=12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[48]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0]=64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[67]:


arr = np.arange(9)
arr.reshape(3,3)


# In[68]:


arr[0]


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[70]:


arr = np.arange(0,9).reshape((3,3))
arr


# In[71]:


arr[1,1]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[73]:


arr = np.arange(0,9).reshape((3,3))
arr


# In[74]:


arr[1::-1,2]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[81]:


arr=np.random.randint(50,size=(10,10))
arr


# In[82]:


print(np.min(arr))
print(np.max(arr))


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[83]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])


# In[84]:


np.intersect1d(a,b)


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[85]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.where(np.in1d(a, b))


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[89]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
data


# In[90]:


print(data[names!="will"])


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[91]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
data


# In[92]:


print(data[names!="will"])
print(data[names!="joe"])


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[99]:


arr2d = np.random.randn(1,15).reshape(5,3)
arr2d


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[102]:


arr = np.random.randn(1,16).reshape(2,2,4)
arr


# 33. Swap axes of the array you created in Question 32

# In[103]:


arr.T


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[109]:


x= np.arange(10)
x= np.sqrt(x)
np.where(x<0.5,0,x)


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[112]:


a = np.random.randint(12)
b = np.random.randint(12)
np.maximum(a,b)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[113]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
names


# In[114]:


names = set(names)
names


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[115]:


a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
ans = np.setdiff1d(a, b)
print(ans)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[116]:


sampleArray= np.array([[34,43,73],[82,22,12],[53,94,66]])
sampleArray


# In[117]:


sampleArray = np.delete(sampleArray, 1, axis=1)
sampleArray


# In[118]:


newColumn = np.array([[10,10,10]])


# In[119]:


np.append(sampleArray,newColumn )


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[120]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])


# In[121]:


np.dot(x,y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[122]:


a= np.random.randn(4,5)
a


# In[123]:


b = np.cumsum(a)
b


# In[ ]:




