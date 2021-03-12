#!/usr/bin/env python
# coding: utf-8

# # Import

# In[1]:


import numpy as np
import os
import cv2
import glob
import matplotlib.pyplot as plt


# In[2]:


DIR = "/Users/Jigs/PythonAI/Datasets/HRK/data500"


# In[3]:


if not os.getcwd()==DIR:
    os.chdir(DIR)


# In[4]:


os.getcwd()


# # Define a Function to READ, CROP, (RESIZE) Image    

# In[9]:


def Image_Preprocessing(img_path):
    #Read Image
    img = cv2.imread(img_path)
    
    #Crop Image
    crop_img = img[50:500, 400:850]
    
    #Resize Image (optional)
    #cv2.resize(crop_img, (224,224))
    
    #Save Image
    cv2.imwrite(img_path, crop_img)
    


# # Call 'Image_Preprocessing' Function

# ## 'test/no' and 'test/yes'

# In[10]:


test_no = DIR+'/test/no'
list_no = glob.glob(test_no+'/*.jpg')
for c in list_no:
    Image_Preprocessing(c)


# In[11]:


test_yes = DIR+'/test/yes'
list_yes = glob.glob(test_yes+'/*.jpg')
for c in list_yes:
    Image_Preprocessing(c)


# ## 'valid/no' and 'valid/yes'

# In[12]:


valid_no = DIR+'/valid/no'
list_no = glob.glob(valid_no+'/*.jpg')
for c in list_no:
    Image_Preprocessing(c)


# In[13]:


valid_yes = DIR+'/valid/yes'
list_yes = glob.glob(valid_yes+'/*.jpg')
for c in list_yes:
    Image_Preprocessing(c)


# ## 'train/no' and 'train/yes'

# In[14]:


train_no = DIR+'/train/no'
list_no = glob.glob(train_no+'/*.jpg')
for c in list_no:
    Image_Preprocessing(c)


# In[15]:


train_yes = DIR+'/train/yes'
list_yes = glob.glob(train_yes+'/*.jpg')
for c in list_yes:
    Image_Preprocessing(c)


# In[ ]:




