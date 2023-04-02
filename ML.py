#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
rebar = "rebars"
debris = "cementitiousDebris"
image_size = (256, 256)
image = []
label = []
for filename in os.listdir(rebar):
    if filename.endswith(".png"):
        img = Image.open(os.path.join(rebar, filename))
        img = img.resize(image_size)
        img_array = np.asarray(img)
        img_array = img_array/255.0
        img_array = img_array.flatten()
        image.append(img_array)
        label.append(0)
for filename in os.listdir(debris):
    if filename.endswith(".png"):
        img = Image.open(os.path.join(debris, filename))
        img = img.resize(image_size)
        img_array = np.asarray(img)
        img_array = img_array/255.0
        img_array = img_array.flatten()
        image.append(img_array)
        label.append(1)
image = np.array(image)
label = np.array(label)
image = image.reshape(-1, 256, 256, 1) 


# In[8]:


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(image, label, test_size=0.2, random_state=42)


# In[9]:


input_shape = (256, 256, 1) # grayscale image with 256x256 size
num_classes = 2 # rebar and debris
model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=num_classes, activation='softmax')
])


# In[10]:


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[11]:


history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))


# In[14]:


test_loss, test_acc = model.evaluate(image, label)
print('Test accuracy:', test_acc)


# In[17]:


y = model.predict(image)


# In[20]:


predicted_class = np.argmax(y, axis=1)


# In[21]:




