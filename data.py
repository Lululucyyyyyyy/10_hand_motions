import numpy as np
from PIL import Image
import glob

def load_images(globpath, num):
    for i, image in enumerate(globpath):
        with open(image, 'rb') as file:
            img = Image.open(file)
            img = img.resize((224, 224))
            np_img = np.array(img)
            my_list.append(np_img)
        labels.append(num)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

my_list = []
labels = []
load_images(glob.glob("dataset/ok*.JPG"), 0)
load_images(glob.glob("dataset/one*.JPG"), 1)
load_images(glob.glob("dataset/two*.JPG"), 2)
load_images(glob.glob("dataset/three*.JPG"), 3)
load_images(glob.glob("dataset/four*.JPG"), 4)
load_images(glob.glob("dataset/five*.JPG"), 5)
load_images(glob.glob("dataset/down*.JPG"),6 )
load_images(glob.glob("dataset/left*.JPG"), 7)
load_images(glob.glob("dataset/right*.JPG"), 8)
load_images(glob.glob("dataset/thumb*.JPG"), 9)

my_list = np.array(my_list)
print('my_list.shape: ', my_list.shape)
labels = np.array(labels)
print("labels.shape ",labels.shape)


p = np.random.permutation(len(my_list))
X_orig = my_list[p]
Y_orig = labels[p]

length = int(len(my_list) * 0.2)
X_test_orig = X_orig[0:length]
Y_test_orig = Y_orig[0:length]

X_dev_orig = X_orig[(length+1):(length*2)]
Y_dev_orig = Y_orig[(length+1):(length*2)]

X_train_orig = X_orig[(length*2+1):len(labels)]
Y_train_orig = Y_orig[(length*2+1):len(labels)]

X_train = X_train_orig/255
X_dev = X_dev_orig/255
X_test = X_test_orig/255
Y_train = convert_to_one_hot(Y_train_orig, 10).T
Y_test = convert_to_one_hot(Y_test_orig, 10).T
Y_dev = convert_to_one_hot(Y_dev_orig, 10).T
def get_datasets():
	return X_train, X_dev, X_test, Y_train, Y_dev, Y_test