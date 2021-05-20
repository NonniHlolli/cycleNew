import tensorflow_datasets as tfds
from numpy import asarray
from numpy import vstack


# images is a python dictionary
def load_images(images):
    data_list = list()
    for image in images:
        data_list.append(image['image'].T)
    return asarray(data_list)

# returns dataA(apples) and dataB(oranges)
def load_dataset(dataset):
    dir = 'cycle_gan/'+ dataset
    ds = tfds.load(dir, with_info=False)
    ds = tfds.as_numpy(ds)

    trainA = load_images(ds['trainA'])
    testA = load_images(ds['testA'])
    dataA = vstack((trainA, testA))
    dataA = (dataA - 127.5) / 127.5

    trainB = load_images(ds['trainB'])
    testB = load_images(ds['testB'])
    dataB = vstack((trainB, testB))
    dataB = (dataB - 127.5) / 127.5
    return dataA, dataB
