"""
Code for "The one with the anime (or hentai?)"
@author: Abhishek Thakur
"""

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

training_images = '/home/abhishek/Workspace/anime_hentai/training_images'
validation_images = '/home/abhishek/Workspace/anime_hentai/validation_images'


def conv_net():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model


model = conv_net()
optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


model_file = "weights_cnn_scratch.h5"
highest_acc_model = ModelCheckpoint(model_file, monitor='val_acc', save_best_only=True)

data_generator_train = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=15.,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

data_generator_validation = ImageDataGenerator(rescale=1./255)

train_generator = data_generator_train.flow_from_directory(
        training_images,
        target_size=(224, 224),
        batch_size=32,
        shuffle=True,
        classes=['anime', 'hentai'],
        class_mode='categorical')

validation_generator = data_generator_validation.flow_from_directory(
        validation_images,
        target_size=(224, 224),
        batch_size=32,
        shuffle=True,
        classes=['anime', 'hentai'],
        class_mode='categorical')

model.fit_generator(
        train_generator,
        samples_per_epoch=1500,
        nb_epoch=35,
        validation_data=validation_generator,
        nb_val_samples=500,
        callbacks=[highest_acc_model])



