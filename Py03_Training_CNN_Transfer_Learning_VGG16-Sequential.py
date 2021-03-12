#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import os
import matplotlib.pyplot as plt


# In[4]:


import tensorflow as tf


# In[5]:


tf.__version__


# In[6]:


from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy


# In[7]:


DIR = "/Users/Jigs/PythonAI/Datasets/HRK/data500"


# In[8]:


if not os.getcwd()==DIR:
    os.chdir(DIR)
    
os.getcwd()


# In[9]:


train_dir = 'train/'
valid_dir = 'valid/'
test_dir  = 'test/'


# # Create Dataset for the CNN Training with 'ImageDataGenerator' and 'flow_from_directory'

# In[10]:


train_dataset = ImageDataGenerator().flow_from_directory(directory= train_dir, batch_size = 16, class_mode= "categorical", target_size = (224, 224))
valid_dataset = ImageDataGenerator().flow_from_directory(directory= valid_dir, batch_size = 8,  class_mode= "categorical", target_size = (224, 224))
test_dataset  = ImageDataGenerator().flow_from_directory(directory= test_dir,  batch_size = 8,  class_mode= "categorical", target_size = (224, 224), shuffle=False)


# # Import 'VGG16' from Keras

# In[11]:


vgg16_model = tf.keras.applications.vgg16.VGG16()


# In[12]:


vgg16_model.summary()


# # Create a new 'model' by removing last layer of VGG16
# ## Here new 'model' is 'Sequential'
# ## Set 'no training' for all VGG16 layers 

# In[13]:


model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)


# In[14]:


for layer in model.layers:
    layer.trainable = False


# # Now add (Dense) an Output Layer with '2 units' and 'softmax' Activation
# ## Here we will train only last Layer

# In[15]:


model.add(Dense(units=2, activation='softmax'))


# In[16]:


model.summary()


# # Compile 'model'
# ## Optimizer: Adam with Learning-Rate: 0.0001
# ## Loss: categorical_crossentropy
# ## Metrics: accuracy

# In[17]:


model.compile(optimizer = Adam(learning_rate=0.0001),  loss = 'categorical_crossentropy',  metrics = ['accuracy'])


# # Training 'model' with fit()

# In[18]:


hist = model.fit(x=train_dataset, validation_data= valid_dataset,  epochs=7, verbose=1)


# In[19]:


print(hist.history.keys())


# In[20]:


model.save('HRK_Epochs7_VGG16_Train400_20210305.h5')

# Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)

# Add 'dropout' rate
x = layers.Dropout(0.4)(x)

# Add final 'sigmoid' layer for classification
x = layers.Dense(2, activation='softmax')(x)


model = tf.keras.models.Model(base_model.input, x)from tensorflow.keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
# In[21]:


im, lb = next(test_dataset)
im.shape


# In[22]:


lb


# In[23]:


pred = model.predict(x=test_dataset, verbose=1)


# In[24]:


test_dataset.classes


# In[25]:


pred.shape


# In[26]:


np.around(pred[:,0])


# In[59]:


def prepare_image(img_path):
    #img_path= '/Users/Jigs/PythonAI/Datasets/HRK/yes/Measure_4007.jpg'
    img = cv2.imread(img_path)
    img_dim = np.expand_dims(img, axis=0)
    return img_dim


# In[51]:


import cv2


# In[52]:


img = cv2.imread('/Users/Jigs/PythonAI/Datasets/HRK/yes/Measure_4007.jpg')
img.shape


# In[53]:


img_dim = np.expand_dims(img, axis=0)
img_dim.shape


# In[76]:


prepro_image = prepare_image('/Users/Jigs/PythonAI/Datasets/HRK/data/test/noo/Measure_2141.jpg')


# In[77]:


prepro_image.shape


# In[78]:


pred_image = model.predict(prepro_image)
pred_image
if pred_image[:,1]>0.7:
    pred_result= 'Yes'
else:
    pred_result= 'No'

print(pred_result)


# In[ ]:





# im, lb = next(train_dataset)

# im.shape

# (im[5,:,:,:])

# lb
