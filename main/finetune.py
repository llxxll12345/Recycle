import keras
from keras import backend as K
import os
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D, Conv2D, BatchNormalization
from keras.applications import MobileNet
from mymobilenet import MyMobileNetV2
from keras.callbacks import CSVLogger
from keras.applications.mobilenet import preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam
import sys
import math
import argparse

model_path = "model/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"

def generate(batch, size=224):
    """Data generation and augmentation
    # Arguments
        batch: Integer, batch size.
        size: Integer, image size.
    # Returns
        train_generator: train set generator
        validation_generator: validation set generator
        count1: Integer, number of train set.
        count2: Integer, number of test set.
    """
    train_path = "../images/training"
    test_path = "../images/testing"

    traingen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )

    trainflow = traingen.flow_from_directory(
        train_path,
        target_size=(size, size),
        batch_size=batch,
        class_mode='categorical'
    )

    testgen = ImageDataGenerator(rescale=1.0 / 255)

    testflow = testgen.flow_from_directory(
        test_path,
        target_size=(size, size),
        batch_size=batch,
        class_mode='categorical'
    )

    trainlen = sum([len(name) for root, dir, name in os.walk('../images/training')])
    testlen = sum([len(name) for root, dir, name in os.walk('../images/testing')])
    print(trainlen, testlen)
    return trainflow, testflow, trainlen, testlen

def step_decay(epoch):
    initial_lrate = 0.3
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
    return lrate

def myfinetune(num_class, KerasModel, layer_num=-1):
    # Using the keras model or not
    if KerasModel:
        conv_model=MobileNetV2(weights=model_path,include_top=False, input_shape=(224,224,3)) 
        conv_model.save_weights('model/base.h5')
    else:
        net = MyMobileNetV2()
        conv_model=net.MobileNetv2Conv()

    #conv_model = MobileNetv2Conv((224, 224, 3))
    #conv_model.load_weights('model/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5')


    for i in range(144, 155):
        conv_model.layers.pop()
    x = conv_model.output

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Conv2D(1280, (3, 3), padding='same', strides=(2, 2))(x)
    x = BatchNormalization(axis=channel_axis)(x)

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Conv2D(640, (1, 1), padding='same', strides=(1, 1))(x)
    x = BatchNormalization(axis=channel_axis)(x)

    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    #x = Dense(512,activation='relu')(x) 
    #x = Dropout(0.25)(x)
    x = Dense(256,activation='relu')(x) 
    x = Dropout(0.5)(x)
    preds=Dense(num_class,activation='softmax')(x) 
    model=Model(inputs=conv_model.input,outputs=preds)
    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    for layer in model.layers[:layer_num]:
        layer.trainable = False
    for layer in model.layers[layer_num:]:
        layer.trainable = True
    return model


def validate_result(num_class):
    model = myfinetune(num_class, 1, 0)
    model.load_weights('result/weights.h5')
     

def train(batch, epochs, num_classes, size, lay_num, using_keras):
    if not os.path.exists('model'):
        os.makedirs('model')

    trainflow, testflow, trainlen, testlen = generate(batch, size)

    model = myfinetune(num_classes, using_keras, layer_num=lay_num)

    # stop after 30 epcohs without any improvement in accuracy
    earlystop = EarlyStopping(monitor='val_acc', patience=30, verbose=1, mode='auto')

    lrate = LearningRateScheduler(step_decay)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    csv_logger = CSVLogger('result.log')

    result = model.fit_generator(
        trainflow,
        validation_data=testflow,
        steps_per_epoch=trainlen // batch,
        validation_steps=testlen // batch,
        epochs=epochs,
        callbacks=[earlystop, csv_logger]
    )
    model.save_weights('model/weights.h5')

def main(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--c",
        help="The number of classes of dataset.")
  
    parser.add_argument(
        "--s",
        default=224,
        help="The image size of train sample.")
    parser.add_argument(
        "--b",
        default=32,
        help="The number of train samples per batch.")
    parser.add_argument(
        "--e",
        default=300,
        help="The number of train iterations.")

    parser.add_argument(
        "--l",
        default=144,
        help="The number of untrainable layers from the start"
    )

    parser.add_argument(
        "--k",
        default=1,
        help="Using keras model or not"
    )
   
    args = parser.parse_args()

    train(
        int(args.b), 
        int(args.e), 
        int(args.c), 
        int(args.s),
        int(args.l),
        int(args.k)
    )

if __name__ == '__main__':
    main(sys.argv)