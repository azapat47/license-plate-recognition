import numpy as np
import pandas as pd
import cv2
import os
from numpy.random import seed
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import convolutional, Dense
from keras.optimizers import Adadelta
#from keras.optimizers import Adam
import keras

IMGS_FOLDER = 'data/letras/'

class ConvNet(object):

	def __init__(self):
		self.batch_size = 100
		self.drop_rate = 0.75
		self.lr = 0.001
		self.n_classes = 26
		self.n_epochs = 80
		
	def build_graph(self):
		self.model = Sequential()
		self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
		self.model.add(Conv2D(64, (3, 3), activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(self.drop_rate))
		self.model.add(Flatten())
		self.model.add(Dense(128, activation='relu'))
		self.model.add(Dropout(self.drop_rate))
		self.model.add(Dense(self.n_classes, activation='softmax'))

	def train(self):
		pass

	def predict_from_file(self, folder, file):
		image = cv2.imread(folder+"/"+file, 0)
		return predict_image(image)

	def predict_image(self, image):
		image = cv2.resize(image, (28,28)).reshape(1, 28, 28, 1)
		label = unique_labels.index('z')
		pred = self.model.predict(image)
		print('logits', pred)
		print('pred', np.argmax(pred, axis=1))

def read_emnist():
	IMGS_FOLDER = 'data/'
	df_train = pd.read_csv(IMGS_FOLDER + 'emnist-letters-train.csv', index_col=0, header=-1)
	df_test = pd.read_csv(IMGS_FOLDER + 'emnist-letters-test.csv', index_col=0, header=-1)
	train_imgs = df_train.values.reshape(-1, 28, 28, 1)
	test_imgs = df_test.values.reshape(-1, 28, 28, 1)
	train_labels = np.array(df_train.index).reshape(-1, 1) - 1
	test_labels = np.array(df_test.index).reshape(-1, 1) - 1
	unique_labels = sorted([chr(x + 96) for x in set(df_train.index)]) # 96 becasue starts at 1, not at 0
	#print(len(unique_labels))
	#print(unique_labels)
	#print(train_imgs.shape)
	#print(test_imgs.shape)
	#print(train_labels.shape)
	#print(test_labels.shape)
	return train_imgs, train_labels, test_imgs, test_labels, unique_labels
		
# def read_data():
# 	img_names = os.listdir(IMGS_FOLDER)
# 	unique_labels = sorted(list(set([s[0] for s in img_names])))
        
# 	classed_names = [[] for _ in range(len(unique_labels))]
# 	for name in img_names:
# 		label = ord(name[0]) - 97
# 		classed_names[label].append(name)

# 	train_names = []
# 	train_imgs = []
# 	train_labels = []
# 	test_names = []
# 	test_imgs = []
# 	test_labels = []
                
# 	for i in range(len(classed_names)):
# 		wall = int(np.floor(len(classed_names[i]) * 0.8))
# 		a, b = classed_names[i][:wall], classed_names[i][wall:]
# 		train_names.extend(a)
# 		train_labels.extend([i for _ in range(len(a))])
# 		test_names.extend(b)
# 		test_labels.extend([i for _ in range(len(b))])
		
# 	for name in train_names:
# 		img = cv2.resize(cv2.imread(IMGS_FOLDER + name, 0), (28, 28))
# 		train_imgs.append(img)
        
# 	for name in test_names:
# 		img = cv2.resize(cv2.imread(IMGS_FOLDER + name, 0), (28, 28))
# 		test_imgs.append(img)
                                        
# 	train_imgs = np.array(train_imgs).reshape(len(train_imgs), 28, 28, 1)
# 	train_labels = np.array(train_labels).reshape(-1, 1)
# 	test_imgs = np.array(test_imgs).reshape(len(test_imgs), 28, 28, 1)
# 	test_labels = np.array(test_labels).reshape(-1, 1)
					
# 	#print(train_imgs.shape)
# 	#print(train_labels.shape)
# 	#print(test_imgs.shape)
# 	#print(test_labels.shape)
# 	return train_imgs, train_labels, test_imgs, test_labels, unique_labels

if __name__ == '__main__':
	seed(0)
	set_random_seed(0)
	# 0-0 98% 80 iters 1.0 drop_p -> bad z, bad g

	train_imgs, train_labels, test_imgs, test_labels, unique_labels = read_emnist()
	input_shape = (train_imgs[0].shape[0], train_imgs[0].shape[1], 1)
	alpha_convNet = ConvNet()
	alpha_convNet.build_graph()

	alpha_convNet.model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])

	train_labels = keras.utils.to_categorical(train_labels, alpha_convNet.n_classes)
	test_labels = keras.utils.to_categorical(test_labels, alpha_convNet.n_classes)
	alpha_convNet.model.fit(train_imgs, train_labels, batch_size=alpha_convNet.batch_size, epochs=alpha_convNet.n_epochs, verbose=1, validation_data=(test_imgs, test_labels))
	
	preds_folder = "data/preds/"
	prediction = alpha_convNet.predict_from_file(preds_folder,"z.jpeg")
	print(prediction)

