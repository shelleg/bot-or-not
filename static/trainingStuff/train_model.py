from keras.datasets import mnist
import tensorflowjs as tfjs
import os
from keras.utils import to_categorical
import numpy as np

from utils import ModelCreator, GeneralUtils


from pre_process_images import meta, images
#idx = meta[:, -4] == "shaital"
#meta = meta[idx]
#images = images[idx]

"""
idx = np.isin(meta[:, -2], np.array(["2","4","6","8"]))
meta = meta[idx]
images = images[idx]
"""

labels = to_categorical([int(l) - 1 for l in meta[:,-2].tolist()], 2=9)
model = ModelCreator.built_model()



print(model.evaluate(images, labels))
model.fit(images, labels, epochs=500, batch_size=25, shuffle = True, validation_split = 0.2)
exit()




import random
from collections import Counter
sync_s3_to_local('collect-data-for-machine-learning', dataPath)

files = os.listdir(dataPath)
random.shuffle(files)
full_files = [os.path.join(dataPath, file) for file in files]
images = read_images(full_files)

keys = {"sha":0, "name":1, "time":2, "direction":3, "number":4}
files = [file.split(".")[0].split("-") for file in files]
files = np.array(files)
names = files[:,keys["name"]]
nameCounter = Counter(names.tolist())

#print(files.shape)
idx = np.array([nameCounter[key] for key in files[:, keys["name"]]]) > 40
images = images[idx]
files = files[idx]


#print (files.shape)

#images = images[files[:, keys["name"]] == "shaital"]
#files = files[files[:, keys["name"]] == "shaital"]


tags = files[:,keys["direction"]]
names = files[:,keys["name"]]
y = to_categorical(tags)


#print (Counter(names))
print (set(names))
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())


model = built_model()
model.fit(images, y, validation_split=0.1, epochs = 5000, shuffle=True, batch_size=64)



exit()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)



import os
print (os.getcwd())
#exit()
tfjs.converters.save_keras_model(model, "model")
#tfjs.converters.save_keras_model()
#model.fit(x_train, y_train)
#print (x_train.shape)
exit()
