#!/usr/bin/env python
# coding: utf-8

# # Import

# In[34]:


import numpy as np
import os
import cv2
import glob
import matplotlib.pyplot as plt


# # Create Folder to save Videos/Images

# In[35]:


try:
    if not os.path.exists('/Users/Jigs/PythonAI/Datasets/HRK4/ExtractedFrames'):
        os.makedirs('/Users/Jigs/PythonAI/Datasets/HRK4/ExtractedFrames')

# if not created then show an ERROR
except OSError:
    print('Could not Create requesed Folder')


# In[36]:


os.getcwd(), os.listdir()


# In[37]:


if not os.getcwd()=='/Users/Jigs/PythonAI/Datasets/HRK4/ExtractedFrames':
    os.chdir('/Users/Jigs/PythonAI/Datasets/HRK4/ExtractedFrames')


# # Read Video from Folder

# In[38]:


cam = cv2.VideoCapture("/Users/Jigs/PythonAI/Datasets/HRK4/HRK_100Frct_5C.mp4")


# In[39]:


type(cam)


# # Extract Frame from Video, Crop & Resize It and Save as a IMAGES

# In[40]:


img_array = []
i=0
j=1000
yes_counter = 0
while(cam.isOpened()):
    ret, frame = cam.read()
    
    if ret==False: # if video is over then 'break'
        break
        
    if i==5000: # if 5000 images then 'break'
        break
    i+=1
    
    img = frame
    crop_img = img[50:500, 400:850]
    
    resize_img = cv2.resize(crop_img, (224,224))
    
    img_name = 'Measure_'+ str(j)+ '.jpg'
    # print('Creating...'+ img_name)
    cv2.imwrite(img_name, resize_img)
    j+=1

## Release everything if job is finished 
cam.release()
cv2.destroyAllWindows()


# In[ ]:




