import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers.convolutional import Convolution1D
from keras.layers import GlobalAveragePooling1D, Input, Dense, Activation, Lambda
import numpy as np
import sys

class Attention(Layer):
	def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
		self.supports_masking = True
		assert op in {'attsum', 'attmean'}
		assert activation in {None, 'tanh'}
		self.op = op
		self.activation = activation
		self.init_stdev = init_stdev
		super(Attention, self).__init__(**kwargs)

	def build(self, input_shape):
		init_val_v = (np.random.randn(input_shape[2]) * self.init_stdev).astype(K.floatx())
		self.att_v = K.variable(init_val_v, name='att_v')
		init_val_W = (np.random.randn(input_shape[2], input_shape[2]) * self.init_stdev).astype(K.floatx())
		self.att_W = K.variable(init_val_W, name='att_W')
		self.trainable_weights = [self.att_v, self.att_W]
	
	def call(self, x, mask=None):
		y = K.dot(x, self.att_W)
		if not self.activation:
			weights = K.theano.tensor.tensordot(self.att_v, y, axes=[0, 2])
		elif self.activation == 'tanh':
			weights = K.theano.tensor.tensordot(self.att_v, K.tanh(y), axes=[0, 2])
		weights = K.softmax(weights)
		out = x * K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1])
		if self.op == 'attsum':
			out = out.sum(axis=1)
		elif self.op == 'attmean':
			out = out.sum(axis=1) / mask.sum(axis=1, keepdims=True)
		return K.cast(out, K.floatx())

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[2])
	
	def compute_mask(self, x, mask):
		return None
	
	def get_config(self):
		config = {'op': self.op, 'activation': self.activation, 'init_stdev': self.init_stdev}
		base_config = super(Attention, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class Conv1DWithMasking(Convolution1D):
	def __init__(self, **kwargs):
		self.supports_masking = True
		super(Conv1DWithMasking, self).__init__(**kwargs)
	
	def compute_mask(self, x, mask):
		return mask
	
################################################################################################	
class MeanOverTime(Layer):
	def __init__(self, mask_zero=True, **kwargs):
		self.mask_zero = mask_zero
		self.supports_masking = True
		super(MeanOverTime, self).__init__(**kwargs)

	def call(self, x, mask=None):	#A
		if self.mask_zero:
			return K.cast(K.sum(x, axis=1) / K.sum(mask, axis=1, keepdims=True), K.floatx())
		else:
			return K.mean(x, axis=1)

# 	def call(self, x, mask=None):	#B
# 		if mask is not None:
# 			mask = K.cast(mask, 'float32')
# 			s = mask.sum(axis=1, keepdims=True)
# 			if K.equal(s, K.zeros_like(s)):
# 				return K.mean(x, axis=1)
# 			else:
# 				return K.cast(x.sum(axis=1) / mask.sum(axis=1, keepdims=True), K.floatx())
# 		else:
# 			return K.mean(x, axis=1)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[2])
		#return (input_shape[0], input_shape[-1])
	
	def compute_mask(self, x, mask):
		return None
	
	def get_config(self):
		config = {'mask_zero': self.mask_zero}
		base_config = super(MeanOverTime, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	#from_config


# https://github.com/fchollet/keras/issues/2151
# http://stackoverflow.com/questions/36428323/lstm-followed-by-mean-pooling/36524166

# "supports both theano and tensorflow"
class TemporalMeanPooling(Layer):
	'''
	This is a custom Keras layer. This pooling layer accepts the temporal
	sequence output by a recurrent layer and performs temporal pooling,
	looking at only the non-masked portion of the sequence. The pooling
	layer converts the entire variable-length hidden vector sequence
	into a single hidden vector, and then feeds its output to the Dense
	layer.
	
	input shape: (nb_samples, nb_timesteps, nb_features)
	output shape: (nb_samples, nb_features)
	'''
	def __init__(self, **kwargs):
		super(TemporalMeanPooling, self).__init__(**kwargs)
		self.supports_masking = True
		self.input_spec = [InputSpec(ndim=3)]

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[2])

	def call(self, x, mask=None): #mask: (nb_samples, nb_timesteps)
		if mask is None:
			mask = K.mean(K.ones_like(x), axis=-1)
		ssum = K.sum(x,axis=-2) #(nb_samples, np_features)
		mask = K.cast(mask,K.floatx())
		rcnt = K.sum(mask,axis=-1,keepdims=True) #(nb_samples)
		return ssum/rcnt
		#return rcnt

	def compute_mask(self, input, mask):
		return None
	
class MeanPool(Layer):
	def __init__(self, **kwargs):
		self.supports_masking = True
		super(MeanPool, self).__init__(**kwargs)

	def compute_mask(self, input, input_mask=None):
		# do not pass the mask to the next layers
		return None

	def call(self, x, mask=None):
		if mask is not None:
			# mask (batch, time)
			mask = K.cast(mask, K.floatx())
			# mask (batch, time, 'x')
			mask = mask.dimshuffle(0, 1, 'x')
			# to make the masked values in x be equal to zero
			x = x * mask
		return K.sum(x, axis=1) / K.sum(mask, axis=1)

	def compute_output_shape(self, input_shape):
		# remove temporal dimension
		return input_shape[0], input_shape[2]

class GlobalMeanPooling(Layer):
	'''Global average pooling operation for temporal data.
	# Input shape
		3D tensor with shape: `(samples, steps, features)`.
	# Output shape
		2D tensor with shape: `(samples, features)`.
	'''
	def __init__(self, **kwargs):
		super(GlobalMeanPooling, self).__init__(**kwargs)
		self.input_spec = [InputSpec(ndim=3)]
		self.supports_masking = True

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[2])

	def compute_mask(self, input, input_mask=None):
		if type(input_mask) is list:
			return [None] * len(input_mask)
		else:
			return None

	def call(self, x, mask=None):
		if mask == None:
			return K.mean(x, axis=1)
		else:
			mask = K.cast(mask, K.floatx())
			# Expand count so we can do a broadcast division
			count = K.sum(mask, axis=1)
			count = K.expand_dims(count, axis=1)
			if K.backend() == 'tensorflow':  # tensorflow shape command dosn't work
				mask = K.repeat_elements(K.expand_dims(mask), K.int_shape(x)[2], 2)
			else:  # theano
				mask = K.repeat_elements(K.expand_dims(mask), K.shape(x)[2], 2)
			x = x * mask  # zero out everything in the mask
			avg = K.sum(x, axis=1) / (count + 1e-5)
			return K.cast(avg, x.dtype)
		
################################################################

from recurrentshop import RecurrentModel
from keras.layers import add, concatenate, multiply
from keras import initializers

def RWA(input_dim, output_dim):
	x = Input((input_dim, ))
	h_tm1 = Input((output_dim, ))
	n_tm1 = Input((output_dim, ))
	d_tm1 = Input((output_dim, ))
	
	x_h = concatenate([x, h_tm1])
	
	u = Dense(output_dim)(x)
	g = Dense(output_dim, activation='tanh')(x_h)

	a = Dense(output_dim, use_bias=False)(x_h)
	e_a = Lambda(lambda x: K.exp(x))(a)

	z = multiply([u, g])
	nt = add([n_tm1, multiply([z, e_a])])
	dt = add([d_tm1, e_a])
	dt = Lambda(lambda x: 1.0 / x)(dt)
	ht = multiply([nt, dt])
	ht = Activation('tanh')(ht)

	return RecurrentModel(input=x, output=ht,
                          initial_states=[h_tm1, n_tm1, d_tm1],
                          final_states=[ht, nt, dt],
                          state_initializer=[initializers.random_normal(stddev=1.0)])
