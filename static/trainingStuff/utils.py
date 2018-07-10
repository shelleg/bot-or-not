from keras.models import Sequential
from typing import TypeVar
S = TypeVar("Sequential")

class GeneralUtils(object):
    @classmethod
    def list_dir_full_path(cls, dir, shuffle=True):
        import os
        import random
        l = [os.path.join(dir, path) for path in os.listdir(dir)]
        random.shuffle(l)
        return l

    @classmethod
    def read_images(cls, list_of_path, flatten=False):
        from scipy import misc
        import numpy as np
        import os
        images = [misc.imresize(misc.imread(path, flatten=False), (200, 200)) for path in list_of_path]

        images = np.stack(images)
        if flatten:
            images = images.reshape(images.shape + (1,))
        return images

    @classmethod
    def mkdir(cls, path):
        import os
        if not os.path.exists(path):
            os.mkdir(os.path.expanduser(path))
        return True

    @classmethod
    def sync_s3_to_local(cls, bucket, local):
        import boto3
        import os

        cls.mkdir(local)

        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket)

        files = set(os.listdir(local))
        objects = [file.key for file in bucket.objects.all() if file.key not in files]
        # print (objects)

        for obj in objects:
            bucket.download_file(obj, os.path.join(local, obj))


class ModelCreator(object):

    @classmethod
    def built_model(cls):
        from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Softmax, Dropout
        from keras.models import Sequential, save_model
        from keras.losses import categorical_crossentropy, binary_crossentropy
        from keras.optimizers import SGD, adadelta

        model = Sequential()
        #model.add(Flatten(input_shape=(200, 200)))

        model.add(Conv2D(200, (2, 2),input_shape=(200, 200, 3), kernel_initializer="uniform"))
        model.add(Activation("sigmoid"))
        #model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(200, (2, 2), kernel_initializer="uniform"))
        model.add(Activation("sigmoid"))
        model.add(MaxPooling2D((2, 2)))
        #model.add(Dropout(0.5))
        model.add(Conv2D(400, (2, 2), kernel_initializer="uniform"))
        model.add(Activation("sigmoid"))
        model.add(MaxPooling2D((2, 2)))
        #model.add(Dropout(0.5))

        """
        model.add(Conv2D(1000, (2, 2), kernel_initializer="uniform"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(1000, (2, 2), kernel_initializer="uniform"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(1000, (2, 2), kernel_initializer="uniform"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(1000, (2, 2), kernel_initializer="uniform"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(1000, (2, 2), kernel_initializer="uniform"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2)))
        """
        model.add(Flatten())
        model.add(Dense(200, kernel_initializer="uniform"))
        model.add(Activation("sigmoid"))
        #model.add(Dropout(0.5))
        #model.add(Dense(100, kernel_initializer="uniform"))
        #model.add(Activation("sigmoid"))
        #model.add(Dropout(0.5))
        #model.add(Dense(1000, kernel_initializer="uniform"))
        #model.add(Activation("relu"))
        model.add(Dense(9))
        #model.add(Softmax())
        model.compile(loss=categorical_crossentropy, optimizer=SGD(), metrics=["accuracy"])
        return model


