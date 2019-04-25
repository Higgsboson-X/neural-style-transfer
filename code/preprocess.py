import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from PIL import Image
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

import time
import functools

mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False

tf.enable_eager_execution()

CONTENT_DIR = '../images/content/'
STYLE_DIR = '../images/style/'


def load_image(path):

	max_dim = 512
	image = Image.open(path)
	longest = max(image.size)
	scale = max_dim / longest

	image = image.resize((round(image.size[0] * scale), round(image.size[1] * scale)), Image.ANTIALIAS)

	image = kp_image.img_to_array(image)

	image = np.expand_dims(image, axis=0)

	return image


def show_image(img, title=None):

	image = np.squeeze(img, axis=0)
	image = image.astype('uint8')

	# plt.imshow(image)
	if title:
		plt.title(title)
	plt.imshow(image)
	# plt.show()



def preprocess_image(path):

	image = load_image(path)
	image = tf.keras.applications.vgg19.preprocess_input(image)

	return image


def deprocess_image(image):

	x = image.copy()
	if len(x.shape) == 4:
		x = np.squeeze(x, 0)

	if len(x.shape) != 3:
		raise ValueError('Invalid input image.')

	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, :, 2] += 123.68

	x = x[:, :, ::-1]

	x = np.clip(x, 0, 255).astype('uint8')

	return x



def test():

	content_path = CONTENT_DIR + '1.jpg'
	style_path = STYLE_DIR + '1.jpg'

	plt.figure(figsize=(10, 10))

	content = load_image(content_path).astype('uint8')
	style = load_image(style_path).astype('uint8')

	plt.subplot(1, 2, 1)
	show_image(content, 'Content')

	plt.subplot(1, 2, 2)
	show_image(style, 'Style')

	plt.show()


if __name__ == '__main__':

	test()

