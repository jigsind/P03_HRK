#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


import numpy as np
import os
import cv2
import glob
import matplotlib.pyplot as plt
import time


# In[3]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy


# In[4]:


tf.__version__


# In[5]:


os.chdir('/Users/Jigs/PythonAI/Datasets/SampleVideo')
os.getcwd()


# # Load Model

# In[7]:


#ResNet50
model = load_model('/Users/Jigs/PythonAI/Projects/P03_HRK_VGG16_ResNet50_VIDEO/MODEL_20210305_HRK_Epochs7_ResNet50-DENSE_Train400.h5')

#VGG16-DENSE
#model = load_model('/Users/Jigs/PythonAI/Projects/P03_HRK4_Preprocessing_Training_Prediction_VGG16_VIDEO/MODEL_20210305_HRK_Epochs9_VGG16-DENSE_Train400.h5')

#VGG16-Sequential
#model = load_model('/Users/Jigs/PythonAI/Projects/P03_HRK4_Preprocessing_Training_Prediction_VGG16_VIDEO/MODEL_20210305_HRK_Epochs7_VGG16_Train400.h5')


# # Single Image Prediction

# In[8]:


def prepare_image(img_path):
    #img_path= '/Users/Jigs/PythonAI/Datasets/HRK/yes/Measure_4007.jpg'
    img = cv2.imread(img_path)
    img_resize = cv2.resize(img, (224,224))
    img_dim = np.expand_dims(img_resize, axis=0)
    return img_dim


# In[9]:


prepro_image = prepare_image('/Users/Jigs/PythonAI/Datasets/HRK/data500/test/yes/YA_HRK00337.jpg')
pred_image = model.predict(prepro_image)

if pred_image[:,1]>0.7:
    pred_result= 'Yes'
else:
    pred_result= 'No'

print(pred_result)


# # Video Prediction

# ## Read Video from Folder

# In[62]:


cam = cv2.VideoCapture("/Users/Jigs/PythonAI/Datasets/SampleVideo/Sample_Measurement_Finland_01.mp4")


# In[63]:


type(cam)


# # Extract Frame from Video, Crop-Resize and Prepare Array for Batch-Prediction

# In[64]:


experimental_relax_shapes=True


# In[65]:


start_frame = 1150
max_frame = 2150  ##4500
dfr = round((max_frame-start_frame)/3)

img_array = np.empty([max_frame,224,224,3])

i=0
j=0

while(cam.isOpened()):
    
    ret, frame = cam.read()
    
    if ret==False: # if video is over then 'break'
        break
        
    if i==max_frame: # if 2000 images then 'break'
        break
    i+=1
    
    #Crop the Image
    img = frame
    crop_img = img[50:500, 400:850]
    
    #Resize Image to (224,224) 
    resize_img = cv2.resize(crop_img, (224,224))
    
    #Expand Dimension (n,224,224,3)
    dim_img = np.expand_dims(resize_img, axis=0)
    
    #Prepare Array for Batch-Prediction
    img_array[j,:,:,:]= dim_img
    j+=1

## Release everything if job is finished 
cam.release()
cv2.destroyAllWindows()


# # Use 'ImageDataGenerator().flow()' to generate Batches for the Prediction

# In[68]:


pred_dataset = ImageDataGenerator().flow(img_array[start_frame:max_frame], batch_size = 16, shuffle=False)


# # Prediction Time!

# In[69]:


t1= time.time()
pred_image_array = model.predict(x=pred_dataset, verbose=1)
t2= time.time()


# In[20]:


print(t2-t1)


# ## Create a Video from Predicted Images

# In[34]:


cam1 = cv2.VideoCapture("/Users/Jigs/PythonAI/Datasets/SampleVideo/Sample_Measurement_Finland_01.mp4")


# In[35]:


t3= time.time()
img_array_video = []

i=0
j=0
n=0

yes_counter = 0
yes_timer = str(round(yes_counter/30))+ ' sec'

while(cam1.isOpened()):
    ret, frame = cam1.read()
    
    if ret==False: # if video is over then 'break'
        break
        
    if i==max_frame: # if 2000 images then 'break'
        break
    i+=1
    
    #Crop the Image
    img = frame
    crop_img = img[50:500, 400:850]
    
    if n>=start_frame:
        if pred_image_array[j,1]>0.7:
            cv2.putText(crop_img, 'Yes', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (300, 300, 0, 255), 4)
            yes_counter+=1
            yes_timer = str(round(yes_counter/30))+ ' sec'
            cv2.putText(crop_img, yes_timer, (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (300, 300, 0, 255), 3)
        else:
            cv2.putText(crop_img, 'No', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 300, 300, 255), 4)
            cv2.putText(crop_img, yes_timer, (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (300, 300, 0, 255), 3)
            
        
        height, width, layers = crop_img.shape
        size = (width, height)
        img_array_video.append(crop_img)
        j+=1
    n+=1

## Define the codec and create VideoWriter object (MJPG-> .mp4, DIVX-> .avi)
out = cv2.VideoWriter('/Users/Jigs/PythonAI/Datasets/SampleVideo/Predict_Video_Finland01_RESNET50_DENSE_02.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30 , size)

for k in range(len(img_array_video)):
    out.write(img_array_video[k])

## Release everything if job is finished 
cam1.release()
out.release()
cv2.destroyAllWindows()
t4= time.time()
print(t4-t3)


# In[36]:


## Release everything if job is finished 
cam.release()
cam1.release()
out.release()
cv2.destroyAllWindows()


# In[37]:


del model


# In[34]:


pred_imange_array.shape


# In[ ]:




