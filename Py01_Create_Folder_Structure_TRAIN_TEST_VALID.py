#!/usr/bin/env python
# coding: utf-8

# # Create a Folder Structure for Image Classification

# In[11]:


import os
import random
import glob
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[17]:


DIR = "/Users/Jigs/PythonAI/Datasets/HRK/data500"
CATEGORY = ["no", "yes"]

SOURCE_NO = "/Users/Jigs/Pictures/Misc/HRK4/Source_NO"
SOURCE_YES = "/Users/Jigs/Pictures/Misc/HRK4/Source_YES"


# In[13]:


os.chdir(DIR)


# In[14]:


os.getcwd()


# ## Create Folders

# In[15]:


if not os.path.exists(DIR+'/train/no'):
    os.makedirs('train/no')
    os.makedirs('train/yes')
    os.makedirs('valid/no')
    os.makedirs('valid/yes')
    os.makedirs('test/no')
    os.makedirs('test/yes')


# In[16]:


os.listdir()


# ## List all the Images in 'Source_NO' Folder
# ### Select 70% of them Randomly and Move them to 'train/no' Folder
# ### Move half of Remaining 30% to 'valid/no' Folder
# ### Move rest of them to 'test/no' Folder

# In[34]:


list_no = glob.glob(SOURCE_NO+'/*.jpg')


# In[19]:


# move 80% of images to 'train' folder
for c in random.sample(list_no, round(len(list_no)*0.8)):
    shutil.copy(c, 'train/no')
    list_no.remove(c)


# In[22]:


# move half of remaining 30% of images to 'valid' folder
for c in random.sample(list_no, round(len(list_no)*0.5)):
    shutil.copy(c, 'valid/no')
    list_no.remove(c)


# In[23]:


# move remaining images to 'test' folder
for c in random.sample(list_no, len(list_no)):
    shutil.copy(c, 'test/no')
    list_no.remove(c)


# ## Same Procedure for 'Source_YES' Folder

# In[30]:


list_yes = glob.glob(SOURCE_YES+'/*.jpg')


# In[31]:


# move 80% of images to 'train' folder
for c in random.sample(list_yes, round(len(list_yes)*0.8)):
    shutil.copy(c, 'train/yes')
    list_yes.remove(c)


# In[32]:


# move half of remaining 30% of images to 'valid' folder
for c in random.sample(list_yes, round(len(list_yes)*0.5)):
    shutil.copy(c, 'valid/yes')
    list_yes.remove(c)


# In[33]:


# move remaining images to 'test' folder
for c in random.sample(list_yes, len(list_yes)):
    shutil.copy(c, 'test/yes')
    list_yes.remove(c)


# In[ ]:




