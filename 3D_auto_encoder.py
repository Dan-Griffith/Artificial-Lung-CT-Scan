import numpy as np
import tensorflow as tf
from keras import layers, Model
import keras
import keras.backend as K
from PIL import Image
import os
import matplotlib.pyplot as plt
from skimage import measure 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import data_process as ps

class DP:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        
def build_autoencoder(input_shape, latent_dim):
    # Encoder
    inputs = tf.keras.Input(shape=input_shape)
    print(inputs.shape)
    x = layers.Conv3D(1, (7, 7,7), activation=tf.keras.layers.LeakyReLU(alpha=0.2), padding='same')(inputs)
    
    x = layers.AveragePooling3D((2, 2,2), padding='same')(x)
    
    x = layers.Conv3D(1, (7,7,7), activation=tf.keras.layers.LeakyReLU(alpha=0.2), padding='same')(x)
    
    x = layers.AveragePooling3D((5, 5,5), padding='same')(x)
    
    x = layers.Conv3D(1, (7, 7,7), activation=tf.keras.layers.LeakyReLU(alpha=0.2), padding='same')(x)
    
    x = layers.AveragePooling3D((7,7,7), padding='same')(x)
    
    x = layers.Conv3D(1, (7, 7,7), activation=tf.keras.layers.LeakyReLU(alpha=0.2), padding='same')(x)
    x = layers.AveragePooling3D((1,1,7), padding='same')(x)
    
    x = layers.Conv3D(1, (7, 7,7), activation=tf.keras.layers.LeakyReLU(alpha=0.2), padding='same')(x)
    x = layers.Flatten()(x)
  
    print(x.shape)
    encoded = layers.Dense(latent_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(x)

   
    x = layers.Dense(343, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(encoded)
    x = layers.Reshape((7,7,7,1))(x)
    
    x = layers.Conv3DTranspose(1, (7, 7,7), activation=tf.keras.layers.LeakyReLU(alpha=0.2), padding='same')(x)
    
    x = layers.UpSampling3D((7, 7,7))(x)
    
    x = layers.Conv3DTranspose(1, (7, 7,7), activation=tf.keras.layers.LeakyReLU(alpha=0.2), padding='same')(x)
    
    x = layers.UpSampling3D((5, 5,5))(x)
    
    x = layers.Conv3DTranspose(1, (7, 7,7), activation=tf.keras.layers.LeakyReLU(alpha=0.2), padding='same')(x)
    
    x = layers.UpSampling3D((2, 2,2))(x)
    
    x = layers.Conv3DTranspose(1, (7, 7,7), activation=tf.keras.layers.LeakyReLU(alpha=0.2), padding='same')(x)
    
    decoded = layers.Conv3DTranspose(1, (1,1, 1), activation='tanh', padding='same')(x) 
 
    autoencoder = Model(inputs, decoded)
    return autoencoder



input_shape = (490, 490,490,1) 
latent_dim = 343
autoencoder = build_autoencoder(input_shape, latent_dim)
autoencoder.compile( loss="huber",optimizer= keras.optimizers.Adam(learning_rate=0.01))
autoencoder.summary()

labels = []
features = []
sample_size = 1
for i in range(sample_size):

    labels.append(np.expand_dims(ps.gen_CT(), axis=-1))

    features.append(np.expand_dims(ps.gen_test_img(), axis=-1))

print(features[0].shape)
print(labels[0].shape)

# Train the model
history = autoencoder.fit(features[0], labels[0], epochs=10, batch_size=7, validation_split=0.2)

