import tensorflow as tf
import tensorflow.contrib as tf_contrib

weight_init = tf_contrib.layers.xavier_initializer()
weight_regularizer = None


##########################
#Layers
##########################
def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=False, scope='conv0'):
	with tf.variable_scope(scope):
		if pad_type == 'zero':
			x = tf.pad(x, [[0,0], [pad,pad], [pad,pad], [0,0]])
		if pad_type == 'reflect':
			x = tf.pad(x, [[0,0], [pad,pad], [pad,pad], [0,0]], mode='REFLECT')
		x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel,
							strides=stride, kernel_initializer=weight_init,data_format='channels_last',
							kernel_regularizer=weight_regularizer, use_bias=use_bias)
		return x


def deconv(x, channels, kernel=4, stride=2, use_bias=True, scope='deconv_0'):
	with tf.variable_scope(scope):
		x = tf.layers.conv2d_transpose(inputs=x, filters=channels, kernel_size=kernel,
										kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
										strides=stride, padding='SAME', use_bias=use_bias)
		return x

def flatten(x):
	return tf.layers.flatten(x)

###########################
#Residual
###########################
def resblock(x_init, channels, use_bias=True, scope='resblock'):
	with tf.variable_scope(scope):
		#Residual block1
		with tf.variable_scope('res1'):
			x = conv(x_init, channels, kernel=3, stride=1, pad=1, use_bias=use_bias)
			x = instance_norm(x)
			x = relu(x)

		#Residual block2
		with tf.variable_scope('res2'):
			x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias)
			x = instance_norm(x)

		return x+x_init

###########################
#Activation function
###########################
def lrelu(x, alpha=0.2):
	return tf.nn.leaky_relu(x, alpha)

def relu(x):
	return tf.nn.relu(x)

def tanh(x):
	return tf.tanh(x)

###########################
#Instance norm
###########################
def instance_norm(x, scope='instance_norm'):
	return tf_contrib.layers.instance_norm(x,
											epsilon=1e-5,
											center=True, scale=True,
											scope=scope)

#######################
#Loss function
#######################
def discriminator_loss(real, fake):
	#WGAN-GP
	real_loss = 0
	fake_loss = 0

	real_loss = -tf.reduce_mean(real)
	fake_loss = tf.reduce_mean(fake)

	loss = real_loss + fake_loss

	return loss

def generator_loss(fake):
	fake_loss = 0

	fake_loss = -tf.reduce_mean(fake)

	loss = fake_loss

	return loss

def classification_loss(logit, label):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=label))

def L1_loss(real, fake):
	return tf.reduce_mean(tf.abs(real-fake))