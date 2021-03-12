#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import matplotlib.pyplot as plt


# In[2]:


import tensorflow as tf


# In[3]:


tf.__version__


# In[4]:


from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy


# In[5]:


DIR = "/Users/Jigs/PythonAI/Datasets/HRK/data500"


# In[6]:


if not os.getcwd()==DIR:
    os.chdir(DIR)
    
os.getcwd()


# In[7]:


train_dir = 'train/'
valid_dir = 'valid/'
test_dir  = 'test/'


# # Create Dataset for the CNN Training with 'ImageDataGenerator' and 'flow_from_directory'

# In[8]:


train_dataset = ImageDataGenerator().flow_from_directory(directory= train_dir, batch_size = 16, class_mode= "categorical", target_size = (224, 224))
valid_dataset = ImageDataGenerator().flow_from_directory(directory= valid_dir, batch_size = 8,  class_mode= "categorical", target_size = (224, 224))
test_dataset  = ImageDataGenerator().flow_from_directory(directory= test_dir,  batch_size = 8,  class_mode= "categorical", target_size = (224, 224), shuffle=False)


# # Import 'VGG16' from Keras

# In[38]:


from tensorflow.keras.applications.vgg16 import VGG16
vgg16_model = VGG16()
base_model  = VGG16(input_shape=(224,224,3),  include_top= False,  weights='imagenet')


# ## Compare VGG16 and 'base_model'

# In[39]:


vgg16_model.summary()


# ### 'base_model' does not include last 3 Dense layers

# In[40]:


base_model.summary()


# ## Set 'no training' for all VGG16 layers 

# In[41]:


for layer in base_model.layers:
    layer.trainable = False


# # Add fully connected DENSE layers to 'base_model'
# ## Here new 'model' is not 'Sequential', but it's 'Functional'

# In[42]:


#Flatten the output of 'base_model' to 1-dimension
x = layers.Flatten()(base_model.output)

#Add fully connected layer with 512 units and 'ReLU' activation
x = layers.Dense(512, activation='relu')(x)

#Add dropout rate of 0.5
x = layers.Dropout(0.5)(x)

#Add a final 'softmax' layer of classification
x =layers.Dense(2, activation='softmax')(x)


# ### New 'model' is 'base_model + fully-connected layers'

# In[43]:


model = tf.keras.models.Model(base_model.input, x)


# In[44]:


model.summary()


# # Compile 'model'
# ## Optimizer: Adam with Learning-Rate: 0.0001
# ## Loss: categorical_crossentropy
# ## Metrics: accuracy

# In[45]:


model.compile(optimizer = Adam(learning_rate=0.0001),  loss = 'categorical_crossentropy',  metrics = ['accuracy'])


# # Training 'model' with fit()

# In[47]:


hist = model.fit(x=train_dataset, validation_data= valid_dataset,  steps_per_epoch = 20, epochs=7, verbose=1)


# In[48]:


print(hist.history.keys())


# In[49]:


model.save('HRK_Epochs9_VGG16-DENSE_Train400_20210305.h5')


# In[50]:


im, lb = next(test_dataset)
im.shape


# In[51]:


lb


# In[52]:


pred = model.predict(x=test_dataset, verbose=1)


# In[53]:


test_dataset.classes


# In[54]:


pred.shape


# In[55]:


np.around(pred[:,0])


# In[56]:


def prepare_image(img_path):
    #img_path= '/Users/Jigs/PythonAI/Datasets/HRK/yes/Measure_4007.jpg'
    img = cv2.imread(img_path)
    img_dim = np.expand_dims(img, axis=0)
    return img_dim


# In[57]:


import cv2


# In[59]:


img = cv2.imread('/Users/Jigs/PythonAI/Datasets/HRK/data500/test/no/NO_HRK00118.jpg')
img.shape


# In[60]:


img_dim = np.expand_dims(img, axis=0)
img_dim.shape


# In[61]:


prepro_image = prepare_image('/Users/Jigs/PythonAI/Datasets/HRK/data/test/noo/Measure_2141.jpg')


# In[62]:


prepro_image.shape


# In[63]:


pred_image = model.predict(prepro_image)
pred_image
if pred_image[:,1]>0.7:
    pred_result= 'Yes'
else:
    pred_result= 'No'

print(pred_result)


# del model
