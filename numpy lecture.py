#!/usr/bin/env python
# coding: utf-8

# NumPy is a Python library used for working with arrays.
# 
# It also has functions for working in domain of linear algebra, fourier transform, and matrices.
# 
# NumPy was created in 2005 by Travis Oliphant. It is an open source project and you can use it freely.
# 
# NumPy stands for Numerical Python.

# In[ ]:


# cài đặt numpy nếu chưa cài
#pip install numpy


# In[1]:


import numpy as np #nhập thư viện numpy
#(là thư viện cốt lõi cho tính toán khoa học trong Python. 
#Nó cung cấp một đối tượng mảng đa chiều hiệu suất cao và các công cụ để làm việc với các mảng này.)
print(np.__version__)


# Mục lục:
# 1. Tạo mảng (Creating Array)
# 2. Kích thước khuôn và ép khuôn (Shape and reshape)
# 3. Chỉ số phần tử trong mảng (Array Indexing)
# 4. Tách mảng con (Array Slicing) 
# 5. Lọc giá trị mảng bằng điều kiện (Array extraction by Condition)
# 6. Tính toán trên ma trận (Operations on matrices)

# In[2]:


# 1. creating array
# one dimension
arr = np.array([1,3,5,7])
print('one dim:\n',arr)
# two dimensions 
arr = np.array([[1,3,5],
                [2,4,6]])
print('two dim:\n',arr)
# three dimensions: it means array of 2-dim arrays 
arr = np.array( [ [[1,3,5],[2,4,6]],
                     [[10,30,50],[20,40,60]] 
                    ]   )
print('three dim:\n',arr)


# In[35]:


# convert from list to numpy array 
a = [1,2,3]
print(a) 
arr = np.array(a)
print(arr)


# In[34]:


# create arrays randomly
# - array with the same value
a = np.full((2,3),4.5)
print(a)
a = np.full((5),2)
print(a)
# create array by multiple 
a = np.array([2]*5)
print(a)

# random values

#Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
a = np.random.rand(2,3) # array (2,3) with values in [0,1]
print(a) 
a = np.random.rand(3) 
print(a) 
a = np.random.rand(3,2,4) 
print(a) 
# randint for values of integers 
a = np.random.randint(10,size=5) # values in [0,10]
print(a)
a = np.random.randint(10,size=(2,5))
print(a)


# In[ ]:





# In[122]:


import random 
n = random.randint(10,20)
print(n)
a = [random.randint(10,20) for i in range(5) ]
print(a)
b = np.array(a)
print(b) 


# In[121]:


# 2. shape and reshape 
a = np.array([[1,3,5,7],
                [2,4,6,8],[1,1,1,1]])
print(a.shape)
print(a.shape[0],a.shape[1])
# reshape to (m,n)
print(np.reshape(a,(2,6)))
print(a.reshape(2,6))
print(a.reshape(2,-1))

# reshape to one line
print(a.ravel())
print(a.reshape(-1,6))
print(a.reshape(6,-1))


# In[50]:


# 2,3. array indexing and slicing 
# how to get/extract values by indexs in arrays
a = np.array([1,3,5,7])
for i in range(len(a)):
    print(a[i], end=' ')
b = np.array([[1,3,5],[2,4,6]])
print('\n\rshape =',b.shape)
print(b)
for i in range(b.shape[0]):
    for j in range(b.shape[1]):
        print(b[i][j])


# In[55]:


# how to get sub-array by indexs
a = np.array([1,2,3,4,5,6,7,8,9])
print(a[:3])
print(a[1:3])
print(a[-1])
print(a[3:-1])
print(a[:-1])


# In[61]:


# how to get sub-array of 2-dim arrays by indexs
a = np.array([[1,3,5],[2,4,6]])
print(a)
print(a[0:2,1:3])
print(a[:,:-1])
print(a[:,:])


# In[71]:


#4. Array extraction by Condition
a = np.array([i for i in range(10)])
print(a)
# get odd values
b = a[a%2==1]
print(b)


# In[75]:


# 5. Data type
arr = np.array(['apple', 'banana', 'cherry'])
print(arr.dtype)
arr = np.array([1, 2, 3, 4])
print(arr.dtype)


# In[87]:


# 6. Operation on darray
a = np.array([[1,2,3],[2,3,4]])
b = np.array([[1,1,1],[2,2,2]])
print(a)
print(b)


# In[95]:


print(a+b)
print('add',np.add(a,b))
print(a-b)
print('sub',np.subtract(a,b))
print(a*b)
print('multiply',np.multiply(a,b))
print(a/b)
print('divide',np.divide(a,b))


# In[98]:


print(a)
print(b.T) # Transpose of a matrix (ma trận chuyển vị))
print(np.dot(a,b.T))
print(np.sqrt(a))


# In[108]:


# operation by axis of array
print(a)
print(b)
# operation on by axis
print(a.sum(axis=0))
print(a.sum(axis=1))
# combine a and b: numpy.concatenate 
print('concat:')
print(np.concatenate((a,b),axis=0))
print(np.concatenate((a,b),axis=1))


# In[ ]:




