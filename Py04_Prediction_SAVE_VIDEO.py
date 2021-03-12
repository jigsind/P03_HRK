#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import load_model


# In[2]:


import numpy as np
import os
import cv2
import glob
import matplotlib.pyplot as plt


# In[3]:


tf.__version__


# # Load Model

# In[4]:


#ResNet50
model = load_model('/Users/Jigs/PythonAI/Projects/P03_HRK4_Preprocessing_Training_Prediction_VGG16_VIDEO/MODEL_20210305_HRK_Epochs7_ResNet60-DENSE_Train400.h5')

#VGG16-DENSE
#model = load_model('/Users/Jigs/PythonAI/Projects/P03_HRK4_Preprocessing_Training_Prediction_VGG16_VIDEO/MODEL_20210305_HRK_Epochs9_VGG16-DENSE_Train400.h5')

#VGG16-Sequential
#model = load_model('/Users/Jigs/PythonAI/Projects/P03_HRK4_Preprocessing_Training_Prediction_VGG16_VIDEO/MODEL_20210305_HRK_Epochs7_VGG16_Train400.h5')


# # Single Image Prediction

# In[5]:


def prepare_image(img_path):
    #img_path= '/Users/Jigs/PythonAI/Datasets/HRK/yes/Measure_4007.jpg'
    img = cv2.imread(img_path)
    img_resize = cv2.resize(img, (224,224))
    img_dim = np.expand_dims(img_resize, axis=0)
    return img_dim


# In[6]:


prepro_image = prepare_image('/Users/Jigs/PythonAI/Datasets/HRK/data500/test/yes/YA_HRK00337.jpg')
pred_image = model.predict(prepro_image)

if pred_image[:,1]>0.7:
    pred_result= 'Yes'
else:
    pred_result= 'No'

print(pred_result)


# # Video Prediction

# ## Read Video from Folder

# In[7]:


cam = cv2.VideoCapture("/Users/Jigs/PythonAI/Datasets/SampleVideo/Sample_Measurement_minus12C.mp4")


# In[8]:


type(cam)


# # Extract Frame from Video, Crop, Predict and Save as a new Video

# In[9]:


img_array = []
i=0
j=0

yes_counter = 0
yes_timer = str(round(yes_counter/30))+ ' sec'

while(cam.isOpened()):
    ret, frame = cam.read()
    
    if ret==False: # if video is over then 'break'
        break
        
    if i==1000: # if 2000 images then 'break'
        break
    i+=1
    
    #Crop the Image
    img = frame
    crop_img = img[50:500, 400:850]
    
    #Resize Image to (224,224) 
    resize_img = cv2.resize(crop_img, (224,224))
    
    #Expand Dimension (n,224,224,3)
    dim_img = np.expand_dims(resize_img, axis=0)
    
    if i>800:
        #Prediction Time!
        pred_image = model.predict(dim_img)

        if pred_image[:,1]>0.7:
            cv2.putText(crop_img, 'Yes', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (300, 300, 0, 255), 4)
            yes_counter+=1
            yes_timer = str(round(yes_counter/30))+ ' sec'
            cv2.putText(crop_img, yes_timer, (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (300, 300, 0, 255), 3)
        else:
            cv2.putText(crop_img, 'No', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 300, 300, 255), 4)
            cv2.putText(crop_img, yes_timer, (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (300, 300, 0, 255), 3)

        height, width, layers = crop_img.shape
        size = (width, height)
        img_array.append(crop_img)

## Define the codec and create VideoWriter object (MJPG-> .mp4, DIVX-> .avi)
out = cv2.VideoWriter('/Users/Jigs/PythonAI/Datasets/SampleVideo/Predict_Video_minus12C_05_ResNet50-DENSE.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30 , size)

for j in range(len(img_array)):
    out.write(img_array[j])

## Release everything if job is finished 
cam.release()
out.release()
cv2.destroyAllWindows()


# In[10]:


## Release everything if job is finished 
cam.release()
out.release()
cv2.destroyAllWindows()


# In[9]:


del model
img_array = []
i=0
yes_counter = 0


# In[25]:


print(i)


# In[7]:


model.summary()


# In[ ]:




