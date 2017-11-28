import numpy as np
import logging
from keras.layers.wrappers import Bidirectional

logger = logging.getLogger(__name__)

def create_model(args, initial_mean_value, overal_maxlen, vocab):
	
	import keras.backend as K
	
	from keras import layers
	from keras.layers import *
	
	from deepats.my_layers import Attention, Conv1DWithMasking, MeanOverTime, TemporalMeanPooling, MeanPool, GlobalMeanPooling
	
	from keras.models import Sequential, Model
	from keras.initializers import Constant


	###############################################################################################################################
	## Create Model
	#
	
	vocab_size = len(vocab)
	
	dropout_W = 0.5		# default=0.5
	dropout_U = 0.1		# default=0.1
	
	cnn_border_mode='same'
	if initial_mean_value.ndim == 0:
		initial_mean_value = np.expand_dims(initial_mean_value, axis=1)
	num_outputs = len(initial_mean_value)
	
	if args.model_type == 'cls':
		raise NotImplementedError
	
	elif args.model_type == 'rwa':
		logger.info('Building a RWA model')
		
		from deepats.rwa import RWA
# 		from deepats.RWACell import RWACell as RWA
		
		model = Sequential()
		model.add(Embedding(vocab_size, args.emb_dim))
		
		for i in range(args.stack-1):
			model.add(LSTM(args.rnn_dim, return_sequences=True, dropout=dropout_W, recurrent_dropout=dropout_U))
			model.add(Dropout(args.dropout_prob))
			
		model.add(RWA(args.rnn_dim))
		#model.add(Bidirectional(RWA(args.rnn_dim), merge_mode='ave'))# {'sum', 'mul', 'concat', 'ave'***, None}
		
		model.add(Dropout(args.dropout_prob))
		
		bias_value = (np.log(initial_mean_value) - np.log(1 - initial_mean_value)).astype(K.floatx())
		model.add(Dense(num_outputs, bias_initializer=Constant(value=bias_value)))
		
		#model.add(Activation('sigmoid'))
		model.add(Activation('tanh'))
		model.emb_index = 0
	
	elif args.model_type == 'regp':
		logger.info('Building an LSTM REGRESSION model with POOLING')
		
		POOL=2 #2
		
		if POOL==1:
			mask_zero=False
		else:
			mask_zero=True
		model = Sequential()
		model.add(Embedding(vocab_size, args.emb_dim, mask_zero=mask_zero))
		
		for i in range(args.stack):
			model.add(LSTM(args.rnn_dim, return_sequences=True, dropout=dropout_W, recurrent_dropout=dropout_U))
			model.add(Dropout(args.dropout_prob))
		
		## MEAN POOLING.
		if POOL==1:
			model.add(GlobalAveragePooling1D())
		elif POOL==2:
			model.add(MeanOverTime())#A/B
		elif POOL==3:
			model.add(TemporalMeanPooling())
		elif POOL==4:
			model.add(MeanPool())
		elif POOL==5:
			model.add(GlobalMeanPooling())
		
		bias_value = (np.log(initial_mean_value) - np.log(1 - initial_mean_value)).astype(K.floatx())
		model.add(Dense(num_outputs, bias_initializer=Constant(value=bias_value)))
		
		model.add(Activation('sigmoid'))
		#model.add(Activation('tanh'))
		model.emb_index = 0
		
	elif args.model_type == 'regp_ORIG':
		logger.info('Building a REGRESSION model with POOLING')
		model = Sequential()
		model.add(Embedding(vocab_size, args.emb_dim, mask_zero=True))
		if args.cnn_dim > 0:
			model.add(Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1))
		if args.rnn_dim > 0:
			model.add(LSTM(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U))
		if args.dropout_prob > 0:
			model.add(Dropout(args.dropout_prob))
		if args.aggregation == 'mot':
			model.add(MeanOverTime(mask_zero=True))
		elif args.aggregation.startswith('att'):
			model.add(Attention(op=args.aggregation, activation='tanh', init_stdev=0.01))
		model.add(Dense(num_outputs))
		if not args.skip_init_bias:
			bias_value = (np.log(initial_mean_value) - np.log(1 - initial_mean_value)).astype(K.floatx())
			model.layers[-1].b.set_value(bias_value)
		model.add(Activation('sigmoid'))
		model.emb_index = 0
		
	
	logger.info('  Done')
	
	###############################################################################################################################
	## Initialize embeddings if requested
	#

	if args.emb_path:
		from w2vEmbReader import W2VEmbReader as EmbReader
		logger.info('Initializing lookup table')
		emb_reader = EmbReader(args.emb_path, emb_dim=args.emb_dim)
		emb_reader.load_embeddings(vocab)
		emb_wts = emb_reader.get_emb_matrix_given_vocab(vocab, model.layers[model.emb_index].get_weights()[0])
		wts = model.layers[model.emb_index].get_weights()
		wts[0] = emb_wts
		model.layers[model.emb_index].set_weights(wts)
		logger.info('  Done')
	
	return model
