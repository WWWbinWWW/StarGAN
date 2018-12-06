from ops import *
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
#import cv2
from PIL import Image

class StarGAN(object):
	def __init__(self, epoch=20, iteration=1000, batch_size=16, n_res=6, n_dis=6, lr=0.0001,
				 adv_weight=1, cls_weight=1, rec_weight=10, ld=10, img_size=128, img_ch=3, ch=16,
				 c_dim = 5, selected_attrs=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'],
				 critic=5):
		self.epoch = epoch
		self.iteration = iteration
		self.batch_size = batch_size
		self.n_res = n_res
		self.n_dis = n_dis
		self.lr = lr
		self.critic = critic

		self.adv_weight = adv_weight
		self.cls_weight = cls_weight
		self.rec_weight = rec_weight
		self.ld = ld

		self.img_size = img_size
		self.img_ch = img_ch
		self.c_dim = c_dim

		self.ch = ch
		self.selected_attrs = selected_attrs
		self.lines = open('list_attr_celeba.txt', 'r').readlines()
		self.idx2attr = {}
		self.attr2idx = {}

		self.train_dataset = []
		self.train_dataset_label = []

	def read_image(self, filename, label):
		#seed = random.randint(0, 2**31-1)
		#x = tf.read_file(filename)
		#x_decode = tf.image.decode_jpeg(x, self.img_ch)
		#178*178
		#x_img = tf.random_crop(x_decode, [178,178], seed=seed)
		#128*128
		#x_img = tf.image.resize_images(x_decode, [self.img_size, self.img_size])
		#x_img = tf.cast(x_img, tf.float32)/127.5-1
		#print(x_img.shape)
		img = Image.open(filename)
		x_img = img.resize([self.img_size, self.img_size])
		x_img = np.array(x_img)

		x_img = x_img / 127.5 - 1


		return x_img[:,:,:3], label

	def augmentation(self, image, aug_size):
		seed = random.randint(0, 2**31-1)
		ori_size = [self.img_size, self.img_size]
		image = tf.image.random_flip_left_right(image, seed=seed)
		image = tf.image.resize_images(image, [aug_size, aug_size])
		image = tf.random_crop(image, ori_size, seed=seed)

		return image

	def processing(self):
		all_attr_name = self.lines[1].split()
		for i, attr_name in enumerate(all_attr_name):
			self.attr2idx[attr_name] = i
			self.idx2attr[i] = attr_name

		lines = self.lines[2:]
		random.seed(1234)
		random.shuffle(lines)

		for i, line in enumerate(lines):
			split = line.split()
			filename = './train/'+split[0]
			values = split[1:]

			label = []
			for attr_name in self.selected_attrs:
				idx = self.attr2idx[attr_name]

				if values[idx] == '1':
					label.append(1)
				else:
					label.append(0)

			self.train_dataset.append(filename)
			self.train_dataset_label.append(label)


	def generator(self, x_init, c, reuse=False, scope='generator'):
		channel = self.ch
		#c的shape为batch_size*c_dim
		#域信息先reshape为batch_size*1*1*c_dim,转换为浮点数
		#对域信息张量扩展，为batch_size*128*128*c_dim
		#将域信息与图像堆叠
		c = tf.cast(tf.reshape(c, shape=[-1,1,1,c.shape[-1]]), tf.float32)
		c = tf.tile(c, [1, x_init.shape[1], x_init.shape[2], 1])
		x = tf.concat([x_init, c], axis=-1)

		with tf.variable_scope(scope, reuse=reuse):
			x = conv(x, channel, kernel=7, stride=1, pad=3, use_bias=False, scope='conv')
			x = instance_norm(x, scope='ins_norm0')
			x = relu(x)

			for i in range(2):
				x = conv(x, channel*2, kernel=4, stride=2, pad=1, use_bias=False, scope='conv_'+str(i))
				x = instance_norm(x, scope='down_ins_norm_'+str(i))
				x = relu(x)

				channel = channel * 2

			for i in range(self.n_res):
				x = resblock(x, channel, use_bias=False, scope='resblock_'+str(i))

			for i in range(2):
				x = deconv(x, channel//2, kernel=4, stride=2, use_bias=False, scope='dconv_'+str(i))
				x = instance_norm(x, scope='up_ins_norm_'+str(i))
				x = relu(x)

				channel = channel // 2

			x = conv(x, channels=3, kernel=7, stride=1, pad=3, use_bias=False, scope='G_logit')
			x = tanh(x)

			return x

	def discriminator(self, x_init, reuse=False, scope='discriminator'):
		with tf.variable_scope(scope, reuse=reuse):
			channel = self.ch

			x = conv(x_init, channel, kernel=4, stride=2, pad=1, use_bias=True, scope='conv')
			x = lrelu(x, 0.01)

			for i in range(1, self.n_dis):
				x = conv(x, channel*2, kernel=4, stride=2, pad=1, use_bias=True, scope='conv_'+str(i))
				x = lrelu(x, 0.01)

				channel = channel * 2

			#128/64=2
			c_kernel = int(self.img_size/(np.power(2, self.n_dis)))

			logit = conv(x, 1, kernel=3, stride=1, pad=1, use_bias=False, scope='D_logit')
			c = conv(x, self.c_dim, c_kernel, stride=1, pad=0, use_bias=False, scope='D_label')
			c = tf.reshape(c, shape=[-1, self.c_dim])

			return logit, c

	def gradient_panalty(self, real, fake, scope='discriminator'):

		alpha = tf.random_uniform([self.batch_size, 1, 1, 1], minval=0., maxval=1.)
		result = alpha*real + (1-alpha)*fake

		logit, _ = self.discriminator(result, reuse=True, scope=scope)

		GP = 0

		grad = tf.gradients(logit, result)[0]
		grad_norm = tf.norm(flatten(grad), axis=1)

		GP = self.ld * tf.reduce_mean(tf.square(grad_norm-1.))

		return GP

	def get_epoch_batch(self, iteration):
		#读取16张图片，用于下个iteration
		j = (iteration//10) % 13750
		#读取16100张图片，用于下个epoch
		#j = epoch % 13
		x_real = []
		x_label = []
		with tf.Session() as sess:
			for i in range(self.batch_size*10):
				x_img, label = self.read_image(self.train_dataset[i+j*160+1600], self.train_dataset_label[i+j*160+1600])
				x_real.append(x_img)
				x_label.append(label)

		x_real = np.array(x_real)
		x_label = np.array(x_label)
		x_real = np.reshape(x_real, [self.batch_size*10, self.img_size, self.img_size, self.img_ch])
		x_label = np.reshape(x_label, [self.batch_size*10, self.c_dim])
		x_label_trg = x_label.copy()
		random.shuffle(x_label_trg)
		#x_label_trg = tf.random_shuffle(x_label)
		print("Read File x_real:{}\t x_label:{}\t x_label_trg:{}\n".format(x_real.shape, x_label.shape, x_label_trg.shape))

		return x_real, x_label, x_label_trg

	def train(self):
		x_real = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_size, self.img_size, self.img_ch], name='x-real')
		x_label = tf.placeholder(tf.float32, shape=[self.batch_size, self.c_dim], name='x-label')
		x_label_trg = tf.placeholder(tf.float32, shape=[self.batch_size, self.c_dim], name='x-label-trg')

		x_fake = self.generator(x_real, x_label_trg)
		x_rec = self.generator(x_fake, x_label, reuse=True)

		real_logit, real_cls = self.discriminator(x_real)
		fake_logit, fake_cls = self.discriminator(x_fake, reuse=True)

		GP = self.gradient_panalty(x_real, x_fake)

		g_adv_loss = generator_loss(fake_logit)
		g_cls_loss = classification_loss(logit=fake_cls, label=x_label_trg)
		g_rec_loss = L1_loss(x_real, x_rec)

		d_adv_loss = discriminator_loss(real_logit, fake_logit) + GP
		d_cls_loss = classification_loss(logit=real_cls, label=x_label)

		self.g_loss = g_adv_loss + self.cls_weight*g_cls_loss + self.rec_weight*g_rec_loss
		self.d_loss = d_adv_loss + self.cls_weight*d_cls_loss

		t_vars = tf.trainable_variables()
		G_vars = [var for var in t_vars if 'generator' in var.name]
		D_vars = [var for var in t_vars if 'discriminator' in var.name]

		self.g_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.g_loss, var_list=G_vars)
		self.d_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.d_loss, var_list=D_vars)

		self.saver = tf.train.Saver(max_to_keep=2)

		self.processing()
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
		config = tf.ConfigProto(gpu_options=gpu_options)
		config.gpu_options.allow_growth=True

		with tf.Session(config=config) as sess:
			#tf.global_variables_initializer().run()
			self.saver.restore(sess, tf.train.latest_checkpoint('./Model'))
			for epoch in range(self.epoch):
				for i in range(self.iteration):
					if i % 10 == 0:
						#每10次读一次数据集160张图片
						x_r, x_l, x_l_t = self.get_epoch_batch(i+epoch*self.iteration)
						print("Training {} iterations".format(i+epoch*self.iteration))

					sess.run(self.d_optimizer, feed_dict={x_real: x_r[(i%10)*16:(i%10)*16+16], x_label: x_l[(i%10)*16:(i%10)*16+16],
														 x_label_trg: x_l_t[(i%10)*16:(i%10)*16+16]})
					if i % self.critic == 0:
						sess.run(self.g_optimizer, feed_dict={x_real: x_r[(i%10)*16:(i%10)*16+16], x_label: x_l[(i%10)*16:(i%10)*16+16],
														 x_label_trg: x_l_t[(i%10)*16:(i%10)*16+16]})
					if i % 100 == 0:
						self.saver.save(sess,'./Model/StarGAN_model', global_step=i+epoch*self.iteration+200)
					

def main(argv=None):
	tf.reset_default_graph()
	model = StarGAN()
	model.train()

if __name__ == '__main__':
	tf.app.run()
