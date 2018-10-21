import numpy as np
import cv2
#from keras.models import Sequential
# from keras.layers import convolutional, Dense

IMGS_FOLDER = 'data/letras/'

def read_data():
	import os
	img_names = os.listdir(IMGS_FOLDER)
	unique_labels = sorted(list(set([s[0] for s in img_names])))

	classed_names = [[] for _ in range(len(unique_labels))]
	for name in img_names:
		label = ord(name[0]) - 97
		classed_names[label].append(name)

	train_names = []
	train_imgs = []
	train_labels = []
	test_names = []
	test_imgs = []
	test_labels = []

	for i in range(len(classed_names)):
		wall = int(np.floor(len(classed_names[i]) * 0.8))
		a, b = classed_names[i][:wall], classed_names[i][wall:]
		train_names.extend(a)
		train_labels.extend([i for _ in range(len(a))])
		test_names.extend(b)
		test_labels.extend([i for _ in range(len(b))])

	for name in train_names:
		img = cv2.resize(cv2.imread(IMGS_FOLDER + name, 0), (28, 28))
		train_imgs.append(img)

	for name in test_names:
		img = cv2.resize(cv2.imread(IMGS_FOLDER + name, 0), (28, 28))
		test_imgs.append(img)

	train_imgs = np.array(train_imgs).reshape(len(train_imgs), 28, 28, 1)
	train_labels = np.array(train_labels).reshape(-1, 1)
	test_imgs = np.array(test_imgs).reshape(len(test_imgs), 28, 28, 1)
	test_labels = np.array(test_labels).reshape(-1, 1)
	
	#print(train_imgs.shape)
	#print(train_labels.shape)
	#print(test_imgs.shape)
	#print(test_labels.shape)
	return train_imgs, train_labels, test_imgs, test_labels, unique_labels

if __name__ == '__main__':
	from numpy.random import seed
	from tensorflow import set_random_seed
	seed(0)
	set_random_seed(0)
	# 0-0 98% 80 iters 1.0 drop_p -> bad z, bad g

	train_imgs, train_labels, test_imgs, test_labels, unique_labels = read_data()
	input_shape = (train_imgs[0].shape[0], train_imgs[0].shape[1], 1)

	# Parameters
	batch_size = 100
	drop_rate = 0.75
	lr = 0.001
	n_classes = 26
	n_epochs = 80

	from keras.models import Sequential
	from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
	#from keras.optimizers import Adam
	from keras.optimizers import Adadelta

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(drop_rate))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(drop_rate))
	model.add(Dense(n_classes, activation='softmax'))


	#opt = Adam(lr=lr)
	model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])

	import keras
	train_labels = keras.utils.to_categorical(train_labels, n_classes)
	test_labels = keras.utils.to_categorical(test_labels, n_classes)
	model.fit(train_imgs, train_labels, batch_size=batch_size, epochs=n_epochs, verbose=1, validation_data=(test_imgs, test_labels))

	pred_img = cv2.resize(cv2.imread('data/preds/z.jpeg', 0), (28, 28)).reshape(1, 28, 28, 1)
	label = unique_labels.index('z')
	pred = model.predict(pred_img)
	print('logits', pred)
	print('label', label)
	print('pred', np.argmax(pred, axis=1))