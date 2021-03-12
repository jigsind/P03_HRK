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


# # Import 'ResNet50' from Keras

# In[9]:


from tensorflow.keras.applications.resnet50 import ResNet50
resnet50_model = ResNet50()


# In[16]:


base_model  = ResNet50(input_shape=(224,224,3),  include_top= False,  weights='imagenet')


# ## Compare ResNet50 and 'base_model'

# In[17]:


resnet50_model.summary()


# ### 'base_model' does not include last 3 Dense layers

# In[18]:


base_model.summary()


# ## Set 'no training' for all VGG16 layers 

# In[19]:


for layer in base_model.layers:
    layer.trainable = False


# # Add fully connected DENSE layers to 'base_model'
# ## Here new 'model' is not 'Sequential', but it's 'Functional'

# In[20]:


#Flatten the output of 'base_model' to 1-dimension
x = layers.Flatten()(base_model.output)

#Add fully connected layer with 512 units and 'ReLU' activation
x = layers.Dense(512, activation='relu')(x)

#Add dropout rate of 0.5
x = layers.Dropout(0.5)(x)

#Add a final 'softmax' layer of classification
x =layers.Dense(2, activation='softmax')(x)


# ### New 'model' is 'base_model + fully-connected layers'

# In[21]:


model = tf.keras.models.Model(base_model.input, x)


# In[22]:


model.summary()


# # Compile 'model'
# ## Optimizer: Adam with Learning-Rate: 0.0001
# ## Loss: categorical_crossentropy
# ## Metrics: accuracy

# In[23]:


model.compile(optimizer = Adam(learning_rate=0.0001),  loss = 'categorical_crossentropy',  metrics = ['accuracy'])


# # Training 'model' with fit()

# In[24]:


hist = model.fit(x=train_dataset, validation_data= valid_dataset,  steps_per_epoch = 20, epochs=7, verbose=1)


# In[25]:


print(hist.history.keys())


# In[26]:


model.save('HRK_Epochs7_ResNet60-DENSE_Train400_20210306.h5')


# In[27]:


im, lb = next(test_dataset)
im.shape


# In[28]:


lb


# In[29]:


pred = model.predict(x=test_dataset, verbose=1)


# In[30]:


test_dataset.classes


# In[31]:


pred.shape


# In[32]:


np.around(pred[:,0])


# In[33]:


def prepare_image(img_path):
    #img_path= '/Users/Jigs/PythonAI/Datasets/HRK/yes/Measure_4007.jpg'
    img = cv2.imread(img_path)
    img_dim = np.expand_dims(img, axis=0)
    return img_dim


# In[34]:


import cv2


# In[35]:


img = cv2.imread('/Users/Jigs/PythonAI/Datasets/HRK/data500/test/no/NO_HRK00118.jpg')
img.shape


# In[36]:


img_dim = np.expand_dims(img, axis=0)
img_dim.shape


# In[37]:


prepro_image = prepare_image('/Users/Jigs/PythonAI/Datasets/HRK/data500/test/no/NO_HRK00118.jpg')


# In[38]:


prepro_image.shape


# In[39]:


pred_image = model.predict(prepro_image)
pred_image
if pred_image[:,1]>0.7:
    pred_result= 'Yes'
else:
    pred_result= 'No'

print(pred_result)


# In[40]:


del base_model
del model


# In[41]:


model.summary()


# In[ ]:




