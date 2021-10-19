import os 
import glob
import shutil
from sklearn.model_selection import train_test_split
import csv
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

import tensorflow as tf

from utils import split_data, order_test_set , create_generators


from deeplearning_model  import streesigns_model







if __name__=="__main__":
    

    if False:
        path_to_data="F:\\tutorials\\tensorflow\\Train"
        path_to_save_train= "F:\\tutorials\\tensorflow\\training_data\\train"
        path_to_save_val ="F:\\tutorials\\tensorflow\\training_data\\val"

        split_data(path_to_data,path_to_save_train,path_to_save_val)
    if False:
        path_to_images = "F:\\tutorials\\tensorflow\\Test"
        path_to_csv = "F:\\tutorials\\tensorflow\\Test.csv"
        order_test_set(path_to_images, path_to_csv)

    path_to_train= "F:\\tutorials\\tensorflow\\training_data\\train"
    path_to_val =  "F:\\tutorials\\tensorflow\\training_data\\val"
    path_to_test = "F:\\tutorials\\tensorflow\\Test"
    
    batch_size = 64
    epochs =15
    
    train_generator,val_generator,test_generator = create_generators(batch_size,path_to_train,path_to_val,path_to_test)
    
    nbr_classes = train_generator.num_classes
    
    
    
    TEST= True
    
    TRAIN=False
    
    if TRAIN:
        path_to_save_model = './Models'

        ckpt_saver = ModelCheckpoint(
            path_to_save_model,
            monitor="val_accuracy",
            mode = 'max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )

        early_stop = EarlyStopping(monitor="val_accuracy",patience=10)

        model = streesigns_model(nbr_classes)
        
        model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
        
        model.fit(train_generator,
                            epochs=epochs,
                            batch_size = batch_size,
                            validation_data=val_generator,
                            callbacks=[ckpt_saver,early_stop]
        )
    if TEST:
        model = tf.keras.models.load_model('./Models')
        model.summary()

        model.evaluate(val_generator)
        model.evaluate(test_generator)