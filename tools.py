import os
import random
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn
from sklearn import preprocessing

def showImage(image):
    plt.imshow(image, cmap='gray')
    plt.show()

def printErr(info):
    print(info)
    exit(1)

def makedir(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError:
            printErr("Failed creating dir: " + dir)

def get_dataset(folder):
    ######################################################################
    imgs, labels = parse_data(path, 'train', flatten)
    indices = np.random.permutation(labels.shape[0])
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    train_img, train_labels = imgs[train_idx, :], labels[train_idx, :]
    image = train_img[0]
    plt.imshow(image, cmap='gray')
    plt.show()

    val_img, val_labels = imgs[val_idx, :], labels[val_idx, :]
    test = parse_data(path, 't10k', flatten)
    return (train_img, train_labels), (val_img, val_labels), test

#  def get_mnist_dataset(batch_size):
    # Step 1: Read in data
    mnist_folder = 'data/mnist'
    download_mnist(mnist_folder)
    train, val, test = read_mnist(mnist_folder, flatten=False)

    # Step 2: Create datasets and iterator
    train_data = tf.data.Dataset.from_tensor_slices(train)
    train_data = train_data.shuffle(10000) # if you want to shuffle your data
    train_data = train_data.batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices(test)
    test_data = test_data.batch(batch_size)

    return train_data, test_data

    ####################################################################################
    full_dataset = loadImages(folder)
    train_set=full_dataset.head()
    test_set=full_dataset.tail()
    return train_set, test_set
    pass

def loadImage(folder,file):
    #file = "9-0-coplate30.png"
    img = cv2.imread(folder+"/"+file,0)
    img = cv2.resize(img,(28,28))
    image = np.asarray(img, dtype=np.float32)
    image = sklearn.preprocessing.normalize(image)
    image[image >= 0.1] = 1
    label_index = file.split("-")[0]
    #print(image.shape, end="****")
    #plt.imshow(image, cmap='gray')
    #plt.title(label_index)
    #plt.show()    
    if (label_index.isdigit()):
        label_index = ord(label_index) - ord('0')
    else:
        label_index = ord(label_index) - 87
    label = np.zeros(36)
    label[label_index] = 1
#    print(label)
#    plt.imshow(image, cmap='gray')
#    plt.title(label)
    #plt.show()
    return image, label

def full_dataset(folder):
    full_dataset = pd.DataFrame()
#    folder = "/home/afzp99/Documentos/2. DATASET PLACAS/3. Caracteres de Placas/"
    folder = folder[:-1]
    files = os.listdir(folder)
    random.shuffle(files)
    #print(files)
    number_files = len(files) 
#    print(number_files)
    images = []
    labels = []
    for file in files:
        image,label = loadImage(folder, file)
        #print("image shape")
        #print(image.shape)
        #print("label shape")
        #print(label.shape)
        images.append(image)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
#    print("images type after")
#    print(type(images[0]))
#    print("labels type after")
#    print(type(labels[0]))
    #    print(images)
    #    print(labels)
    return images,labels

def get_dataset(folder, train_size, test_size, batch_size):
    images, labels = full_dataset(folder)
    train=(images,labels)
    #print(type(labels))
    #print(type(images))
    #print(type(train))
    #print(images.shape)
    #print(labels.shape)
    print(train[0][0].shape)
    print(train[1].shape)
    test=(images,labels)
    train_data = tf.data.Dataset.from_tensor_slices(train)
    train_data = train_data.batch(batch_size)
    test_data = tf.data.Dataset.from_tensor_slices(test)
    test_data = test_data.batch(batch_size)
    return train_data, test_data

#    return (images,labels),(images,labels)

#loadImages()
#makedir("/tmp/folders/anidadas")
#full_dataset("")
