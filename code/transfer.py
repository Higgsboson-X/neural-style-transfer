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
import IPython.display

from preprocess import *

CONTENT_LAYERS = ['block5_conv2']
STYLE_LAYERS = [

	'block1_conv1',
	'block2_conv1',
	'block3_conv1',
	'block4_conv1',
	'block5_conv1'

]

NUM_CONTENT_LAYERS = len(CONTENT_LAYERS)
NUM_STYLE_LAYERS = len(STYLE_LAYERS)

def construct_model():

	vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
	vgg.trainable = False

	style_outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS]
	content_outputs = [vgg.get_layer(name).output for name in CONTENT_LAYERS]

	model_outputs = style_outputs + content_outputs

	return models.Model(vgg.input, model_outputs)


def get_content_loss(content, target):

	return tf.reduce_mean(tf.square(content - target))


def gram_matrix(tensor):

	channels = int(tensor.shape[-1])

	a = tf.reshape(tensor, [-1, channels])
	n = tf.shape(a)[0]

	gram = tf.matmul(a, a, transpose_a=True)

	return gram / tf.cast(n, tf.float32)


def get_style_loss(style, gram_target):

	height, width, channels = style.get_shape().as_list()
	gram_style = gram_matrix(style)

	return tf.reduce_mean(tf.square(gram_style - gram_target)) # / (4. * (channels ** 2) * (width * height) ** 2)


def get_features(model, content_path, style_path):

	content_image = preprocess_image(content_path)
	style_image = preprocess_image(style_path)

	content_outputs = model(content_image)
	style_outputs = model(style_image)

	style_features = [layer[0] for layer in style_outputs[:NUM_STYLE_LAYERS]]
	content_features = [layer[0] for layer in content_outputs[NUM_STYLE_LAYERS:]]

	return style_features, content_features


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):

	style_weight, content_weight = loss_weights

	model_outputs = model(init_image)

	style_output_features = model_outputs[:NUM_STYLE_LAYERS]
	content_output_features = model_outputs[NUM_STYLE_LAYERS:]

	style_score = 0
	content_score = 0

	weight_per_style_layer = 1.0 / float(NUM_STYLE_LAYERS)
	for target_style, comb_style in zip(gram_style_features, style_output_features):
		style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

	weight_per_content_layer = 1.0 / float(NUM_CONTENT_LAYERS)
	for target_content, comb_content in zip(content_features, content_output_features):
		# print(comb_content[0].shape, target_content.shape)
		content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

	style_score *= style_weight
	content_score *= content_weight

	loss = style_score + content_score

	return loss, style_score, content_score


def compute_grads(cfg):

	with tf.GradientTape() as tape:
		all_loss = compute_loss(**cfg)

	total_loss = all_loss[0]

	return tape.gradient(total_loss, cfg['init_image']), all_loss


def style_transfer(content_path, style_path, iters=1000, content_weight=1e3, style_weight=1e-2):


	model = construct_model()

	for layer in model.layers:
		layer.trainable = False

	style_features, content_features = get_features(model, content_path, style_path)
	gram_style_features = [gram_matrix(feature) for feature in style_features]

	init_image = preprocess_image(content_path)
	init_image = tfe.Variable(init_image, dtype=tf.float32)

	opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

	best_loss, best_img = float('inf'), None

	loss_weights = (style_weight, content_weight)

	cfg = {

		'model': model,
		'loss_weights': loss_weights,
		'init_image': init_image,
		'gram_style_features': gram_style_features,
		'content_features': content_features

	}

	num_rows = 2
	num_cols = 5

	display_interval = iters / (num_rows * num_cols)

	start_time = time.time()
	global_start = time.time()

	norm_means = np.array([103.939, 116.779, 123.68])
	min_vals = -norm_means
	max_vals = 255 - norm_means

	imgs = []

	for i in range(iters):

		grads, all_loss = compute_grads(cfg)
		loss, style_score, content_score = all_loss

		opt.apply_gradients([(grads, init_image)])

		clipped = tf.clip_by_value(init_image, min_vals, max_vals)

		init_image.assign(clipped)

		end_time = time.time()

		if loss < best_loss:
			best_loss = loss
			best_img = deprocess_image(init_image.numpy())

		if i % display_interval == 0:
			start_time = time.time()

			plot_img = init_image.numpy()
			plot_img = deprocess_image(plot_img)
			imgs.append(plot_img)

			filename = '../results/' + content_path[-5] + '-' + style_path[-5] + '/' + str(i) + '.jpg'
			plt.imshow(plot_img)
			plt.savefig(filename)

			IPython.display.clear_output(wait=True)
			IPython.display.display_png(Image.fromarray(plot_img))

			print('Iteration: {}'.format(i))
			print('Total loss: {:.4e}, '
				  'style loss: {:.4e}, '
				  'content loss: {:.4e}, '
				  'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))

	print('Total time: {:.4f}s'.format(time.time() - global_start))

	IPython.display.clear_output(wait=True)

	plt.figure(figsize=(14, 4))

	for i, img in enumerate(imgs):
		plt.subplot(num_rows, num_cols, i + 1)
		plt.imshow(img)
		plt.xticks([])
		plt.yticks([])

	return best_img, best_loss


def run():

	content_id = 3
	style_id = 7

	content_path = CONTENT_DIR + str(content_id) + '.jpg'
	style_path = STYLE_DIR + str(style_id) + '.jpg'

	image, loss = style_transfer(content_path, style_path, iters=1000, content_weight=1e3, style_weight=1e-2)

	filename = '../results/content[' + str(content_id) + ']-style[' + str(style_id) + '].jpg'

	plt.imshow(image)
	plt.savefig(filename)


if __name__ == '__main__':

	run()


