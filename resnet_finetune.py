"""
Code for "The one with the anime (or hentai?)"
@author: Abhishek Thakur
"""

from keras.layers import AveragePooling2D
from keras.models import Model
from keras.optimizers import SGD

from keras.applications.resnet50 import ResNet50

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


training_images = '/home/abhishek/Workspace/anime_hentai/training_images'
validation_images = '/home/abhishek/Workspace/anime_hentai/validation_images'


model = ResNet50(include_top=False, weights='imagenet',
                 input_tensor=None, input_shape=(224, 224, 3))

output = model.get_layer(index=-3).output
output = AveragePooling2D((5, 5), strides=(5, 5), name='avg_pool')(output)
output = Flatten(name='flatten')(output)
output = Dense(2, activation='softmax', name='predictions')(output)

resnet_model = Model(model.input, output)

optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=True)
resnet_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


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

resnet_model.fit_generator(
        train_generator,
        samples_per_epoch=1500,
        nb_epoch=35,
        validation_data=validation_generator,
        nb_val_samples=500,
        callbacks=[highest_acc_model])



