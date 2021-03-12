#!/usr/bin/env python
# coding: utf-8

# # Import

# In[11]:


import numpy as np
import os
import cv2
import glob
import matplotlib.pyplot as plt
import tensorflow as tf


# # Create Folder to save Videos/Images

# In[12]:


try:
    if not os.path.exists('/Users/Jigs/PythonAI/Datasets/SampleVideo'):
        os.makedirs('/Users/Jigs/PythonAI/Datasets/SampleVideo')

# if not created then show an ERROR
except OSError:
    print('Could not Create requesed Folder')


# In[13]:


os.getcwd(), os.listdir()


# In[14]:


if not os.getcwd()=='/Users/Jigs/PythonAI/Datasets/SampleVideo':
    os.chdir('/Users/Jigs/PythonAI/Datasets/SampleVideo')


# # Read Video from Folder

# In[15]:


cam = cv2.VideoCapture("/Users/Jigs/PythonAI/Datasets/SampleVideo/Sample-Measurement.mp4")


# In[16]:


type(cam)


# # Extract Frame from Video, Crop It and Save as a new Video

# In[17]:


img_array = []
i=0
yes_counter = 0
while(cam.isOpened()):
    ret, frame = cam.read()
    
    if ret==False: # if video is over then 'break'
        break
        
    if i==1000: # if 2000 images then 'break'
        break
    i+=1
    
    img = frame
    crop_img = img[50:500, 400:850]
    
    resize_img = cv2.resize(crop_img, (224,224))
    
    if i>800:
        cv2.putText(resize_img, 'Yes', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (300, 300, 0, 255), 4)
        yes_counter+=1
        yes_timer = str(round(yes_counter/30))+ ' sec'
        cv2.putText(resize_img, yes_timer, (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (300, 300, 0, 255), 3)
    else:
        cv2.putText(resize_img, 'No', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 300, 300, 255), 4)
        cv2.putText(resize_img, yes_timer, (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (300, 300, 0, 255), 3)
    
    height, width, layers = resize_img.shape
    size = (width, height)
    img_array.append(resize_img)

## Define the codec and create VideoWriter object (MJPG-> .mp4, DIVX-> .avi)
out = cv2.VideoWriter('Measure_Image2Video.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30 , size)

for j in range(len(img_array)):
    out.write(img_array[j])

## Release everything if job is finished 
cam.release()
out.release()
cv2.destroyAllWindows()


# ## Reading Frames from Video and Save to target Folder
i=0
os.chdir('Datasets/SampleVideo/Extracted_Images')

while(cam.isOpened()):
    ret, frame = cam.read()
    
    if ret==False: # if video is over then 'break'
        break
        
    if i==2000: # if 2000 images then 'break'
        break
    
    img_name = 'Measure_'+ str(i)+ '.jpg'
    # print('Creating...'+ img_name)
    cv2.imwrite(img_name, frame)
    
    i+=1
    
cam.release()
cv2.destroyAllWindows()
# In[8]:


plt.imshow(crop_img), crop_img.shape


# In[9]:


resize_img = cv2.resize(crop_img, (224,224))


# In[10]:


plt.imshow(resize_img), resize_img.shape


# In[ ]:




