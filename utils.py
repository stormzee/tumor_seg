import os
import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Dropout, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization
from keras.metrics import MeanIoU
from keras.optimizers import Adam
# create custom image loader

def load_image(img_dir, img_list):
    loaded_images = []
    for i, image_name in enumerate(img_list):
        if image_name.split('.')[1] == 'npy':
            image = np.load(img_dir+image_name)
            loaded_images.append(image)
            
    loaded_images = np.array(loaded_images)
    
    
    return (loaded_images)



def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)
    
    # infinite generation of images for each batch
    while True:
        
        batch_start = 0
        batch_end = batch_size
        
        while batch_start < L:
            limit = min(batch_end, L)
            
            X = load_image(img_dir, img_list[batch_start:limit])
            X = X.astype(np.float32)
            Y = load_image(mask_dir, mask_list[batch_start:limit])
            Y = Y.astype(np.float32)
            
            yield (X, Y)
            
            batch_start += batch_size
            batch_end += batch_size
            



# link to useful resource...
# https://stackoverflow.com/questions/71976827/concatenate-layer-requires-inputs-with-matching-shapes-except-for-the-concat-a

def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    #Build the model
        kernel_initializer = kernel_initializer = 'he_uniform'
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS), name='input')
        #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
        # for my system, inputs are gonna be normalized before feeding to the model...
        s = inputs
        #Contraction path
        c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
        c1 = Dropout(0.1)(c1)
        c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
        p1 = MaxPooling3D((2, 2, 2))(c1)
        
        c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
        p2 = MaxPooling3D((2, 2, 2))(c2)
         
        c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
        p3 = MaxPooling3D((2, 2, 2))(c3)
         
        c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
        p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
         
        c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
        
        #Expansive path 
        u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
         
        u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
         
        u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
         
        u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
         
        outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)
         
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        #compile model outside of this function to make it flexible. 
        # model.summary()
        
        return model