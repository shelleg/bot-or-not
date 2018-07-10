from utils import GeneralUtils
from setup import dataPath, processedImagesPath
from scipy import misc
import shutil
import os
import numpy as np

metaFilePath, imagesFilePath = os.path.join(processedImagesPath, "data.npy"), os.path.join(processedImagesPath, "images.npy")

def save_data():
    import numpy as np
    np.save(metaFilePath, data)
    np.save(imagesFilePath, images)


def load_data():
    import numpy as np
    return np.load(metaFilePath), np.load(imagesFilePath)


def process_data():
    global data, images
    GeneralUtils.sync_s3_to_local('collect-data-for-machine-learning', dataPath)
    full_files = GeneralUtils.list_dir_full_path(dataPath, shuffle=True)
    images = GeneralUtils.read_images(full_files)
    shutil.rmtree(processedImagesPath, ignore_errors=True)
    GeneralUtils.mkdir(processedImagesPath)
    data = np.array([file.split("-") for file in full_files])
    save_data()


if __name__ == "__main__":
    process_data()

meta, images = load_data()
