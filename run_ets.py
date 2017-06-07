#!/usr/bin/env python

import os
import sys
import argparse
import logging
import numpy as np
import scipy
from time import time
import deepats.utils as U
import pickle as pk

from deepats.ets_evaluator import Evaluator
import deepats.ets_reader as dataset

from keras.models import model_from_yaml

logger = logging.getLogger(__name__)

##################################
## set theano/tensorflow in:
## ~/.keras/keras.json
##################################

## kappa loss
import keras.backend as K
#import theano.tensor as T

def kappa_metric(t,x):
	u = 0.5 * K.sum(K.square(x - t))
	v = K.dot(K.transpose(x), t - K.mean(t))
	return v / (v + u)

# theano
def kappa_loss(t,x):
	u = K.sum(K.square(x - t))
	v = K.dot(K.squeeze(x,1), K.squeeze(t - K.mean(t),1))##v = T.tensordot(x, t - K.mean(t))
	return u / (2*v + u)# =1-qwk


###############################################################################################################################
## Parse arguments
#
def run(argv=None):

	parser = argparse.ArgumentParser()
	parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")
	parser.add_argument("-p", "--prompt", dest="prompt_id", type=int, metavar='<int>', required=False, help="Promp ID for ASAP dataset. '0' means all prompts.")
	parser.add_argument("-t", "--type", dest="model_type", type=str, metavar='<str>', default='regp', help="Model type (reg|regp|breg|bregp) (default=regp)")
	parser.add_argument("-u", "--rec-unit", dest="recurrent_unit", type=str, metavar='<str>', default='lstm', help="Recurrent unit type (lstm|gru|simple) (default=lstm)")
	parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='rmsprop', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
	parser.add_argument("-l", "--loss", dest="loss", type=str, metavar='<str>', default='mse', help="Loss function (mse|mae) (default=mse)")
	parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=50, help="Embeddings dimension (default=50)")
	parser.add_argument("-c", "--cnndim", dest="cnn_dim", type=int, metavar='<int>', default=0, help="CNN output dimension. '0' means no CNN layer (default=0)")
	parser.add_argument("-w", "--cnnwin", dest="cnn_window_size", type=int, metavar='<int>', default=3, help="CNN window size. (default=3)")
	parser.add_argument("-r", "--rnndim", dest="rnn_dim", type=int, metavar='<int>', default=300, help="RNN dimension. '0' means no RNN layer (default=300)")
	parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size (default=32)")
	parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")
	parser.add_argument("--aggregation", dest="aggregation", type=str, metavar='<str>', default='mot', help="The aggregation method for regp and bregp types (mot|attsum|attmean) (default=mot)")
	parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.5, help="The dropout probability. To disable, give a negative number (default=0.5)")
	parser.add_argument("--vocab-path", dest="vocab_path", type=str, metavar='<str>', help="(Optional) The path to the existing vocab file (*.pkl)")
	parser.add_argument("--skip-init-bias", dest="skip_init_bias", action='store_true', help="Skip initialization of the last layer bias")
	parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file (Word2Vec format)")
	parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=100, help="Number of epochs (default=50)")
	parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
	parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")
	## dsv
	parser.add_argument("--min-word-freq", dest="min_word_freq", type=int, metavar='<int>', default=2, help="Min word frequency")
	parser.add_argument("--stack", dest="stack", type=int, metavar='<int>', default=1, help="how deep to stack core RNN")
	parser.add_argument("--skip-emb-preload", dest="skip_emb_preload", action='store_true', help="Skip preloading embeddings")
	parser.add_argument("--tokenize-old", dest="tokenize_old", action='store_true', help="use old tokenizer")
	
	parser.add_argument("-ar", "--abs-root", dest="abs_root", type=str, metavar='<str>', required=False, help="Abs path to root directory")
	parser.add_argument("-ad", "--abs-data", dest="abs_data", type=str, metavar='<str>', required=False, help="Abs path to data directory")
	parser.add_argument("-ao", "--abs-out", dest="abs_out", type=str, metavar='<str>', required=False, help="Abs path to output directory")
	parser.add_argument("-dp", "--data-path", dest="data_path", type=str, metavar='<str>', required=False, help="Abs path to output directory")
	##
	
	if argv is None:
		args = parser.parse_args()
	else:
		args = parser.parse_args(argv)
	
	out_dir = args.abs_out
	U.mkdir_p( os.path.join(out_dir, 'preds'))
	U.set_logger(out_dir)
	U.print_args(args)
	
	assert args.model_type in {'reg', 'regp', 'breg', 'bregp', 'rwa'}
	assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}
	assert args.loss in {'mse', 'mae', 'kappa', 'soft_kappa'}
	assert args.recurrent_unit in {'lstm', 'gru', 'simple', 'rwa'}
	assert args.aggregation in {'mot', 'attsum', 'attmean'}
	
	if args.seed > 0:
		RANDSEED = args.seed
	else:
		RANDSEED = np.random.randint(10000)
	np.random.seed(RANDSEED)
	
	
	#######################
	
	#from deepats.util import GPUtils as GPU
	import GPUtil as GPU
	mem = GPU.avail_mem()
	logger.info('AVAIL GPU MEM == %.4f' % mem)
# 	if mem < 0.05:
# 		return None
	###############################################################################################################################
	## Prepare data
	#
	
	emb_words = None
	if not args.skip_emb_preload:#if args.emb_path:
		from deepats.w2vEmbReader import W2VEmbReader as EmbReader
		logger.info('Loading embedding vocabulary...')
		emb_reader = EmbReader(args.emb_path, emb_dim=args.emb_dim)
		emb_words = emb_reader.load_words()
		
	train_df, dev_df, test_df, vocab, overal_maxlen, qwks = dataset.get_data(args.data_path, emb_words=emb_words, seed=RANDSEED)
	vocab_size = len(vocab)
	
	train_x = train_df['text'].values;	train_y = train_df['y'].values
	dev_x = dev_df['text'].values; 		dev_y = dev_df['y'].values
	test_x = test_df['text'].values;	test_y = test_df['y'].values

	# Dump vocab

	abs_vocab_file = os.path.join(out_dir, 'vocab.pkl')
	with open(os.path.join(out_dir, 'vocab.pkl'), 'wb') as vocab_file:
		pk.dump(vocab, vocab_file)
	
	if args.recurrent_unit == 'rwa':
		setattr(args, 'model_type', 'rwa')
		
	# Pad sequences for mini-batch processing
	from keras.preprocessing import sequence
	
	if args.model_type in {'breg', 'bregp', 'rwa'}:
		assert args.rnn_dim > 0
		train_x = sequence.pad_sequences(train_x, maxlen=overal_maxlen)
		dev_x = sequence.pad_sequences(dev_x, maxlen=overal_maxlen)
		test_x = sequence.pad_sequences(test_x, maxlen=overal_maxlen)
	else:
		train_x = sequence.pad_sequences(train_x)
		dev_x = sequence.pad_sequences(dev_x)
		test_x = sequence.pad_sequences(test_x)
	
	###############################################################################################################################
	## Some statistics
	
# 	train_y = np.array(train_y, dtype=K.floatx())
# 	dev_y = np.array(dev_y, dtype=K.floatx())
# 	test_y = np.array(test_y, dtype=K.floatx())
	
	bincounts, mfs_list = U.bincounts(train_y)
	with open(os.path.join(out_dir,'bincounts.txt'), 'w') as output_file:
		for bincount in bincounts:
			output_file.write(str(bincount) + '\n')
	
	train_mean = train_y.mean(axis=0)
	train_std = train_y.std(axis=0)
	dev_mean = dev_y.mean(axis=0)
	dev_std = dev_y.std(axis=0)
	test_mean = test_y.mean(axis=0)
	test_std = test_y.std(axis=0)
	
	logger.info('Statistics:')
	logger.info('  TEST KAPPAS (float, int)= \033[92m%.4f (%.4f)\033[0m ' % (qwks[1], qwks[0]))
	logger.info('  RANDSEED =   ' + str(RANDSEED))
	logger.info('  train_x shape: ' + str(np.array(train_x).shape))
	logger.info('  dev_x shape:   ' + str(np.array(dev_x).shape))
	logger.info('  test_x shape:  ' + str(np.array(test_x).shape))
	logger.info('  train_y shape: ' + str(train_y.shape))
	logger.info('  dev_y shape:   ' + str(dev_y.shape))
	logger.info('  test_y shape:  ' + str(test_y.shape))
	logger.info('  train_y mean: %s, stdev: %s, MFC: %s' % (str(train_mean), str(train_std), str(mfs_list)))
	logger.info('  overal_maxlen:  ' + str(overal_maxlen))
	
	###############################################################################################################################
	## Optimizaer algorithm
	#
	
	from deepats.optimizers import get_optimizer
	#optimizer = get_optimizer(args)
	
	from keras import optimizers
	
	## RMS-PROP
	
	#optimizer = optimizers.RMSprop(lr=0.00075, rho=0.9, clipnorm=1)
	#optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, clipnorm=1)
	optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6, clipnorm=10)
	#optimizer = optimizers.RMSprop(lr=0.0018, rho=0.88, epsilon=1e-6, clipnorm=10)
		
	#optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, clipnorm=10)
	#optimizer = optimizers.RMSprop(lr=0.004, rho=0.85, epsilon=1e-6, clipnorm=10)# best 2.1 (RWA)
	#optimizer = optimizers.RMSprop(lr=0.0025, rho=0.8, epsilon=1e-8, clipnorm=10) # best 2.1 (RWA)
	#optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, clipnorm=10) # best 2.3 (RWA)
	#optimizer = optimizers.RMSprop(lr=0.0025, rho=0.88, epsilon=1e-8, clipnorm=10) # best 2.3 (RWA)
	#optimizer = optimizers.RMSprop(lr=0.004, rho=0.85, epsilon=1e-8, clipnorm=10) # best 2.10 (RWA)
	
	## OTHER METHODS
	#optimizer = optimizers.Adam(lr=0.0018, clipnorm=5)
	#optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1)
	#optimizer = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-06, clipnorm=10)
	
	#optimizer = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=10)
	#optimizer = optimizers.SGD(lr=0.05, momentum=0, decay=0.0, nesterov=False, clipnorm=10)
	#optimizer = optimizers.Adagrad(lr=0.03, epsilon=1e-08, clipnorm=10)
	#optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06, clipnorm=10)
	
	###############################################################################################################################
	## Building model
	#
	if args.loss == 'mse':
		loss = 'mean_squared_error'
		metric = kappa_metric; metric_name = 'kappa_metric'
	elif args.loss == 'mae':
		loss = 'mean_absolute_error'
		metric = kappa_metric; metric_name = 'kappa_metric'
	elif args.loss == 'kappa':
		loss = kappa_loss
		metric = kappa_metric; metric_name = 'kappa_metric'
	
	########################################################
	
	from deepats.models import create_model
	model = create_model(args, train_y.mean(axis=0), overal_maxlen, vocab)
	
	############################################
	'''
	# test yaml serialization/de-serialization
	yaml = model.to_yaml()
	print yaml
	from deepats.my_layers import MeanOverTime
	from deepats.rwa import RWA
	model = model_from_yaml(yaml, custom_objects={'MeanOverTime': MeanOverTime, 'RWA':RWA})
	'''
	############################################
	
	model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
	
	print(model.summary())
	
	###############################################################################################################################
	## Plotting model
	#
# 	from keras.utils.visualize_util import plot
# 	plot(model, to_file = os.path.join(out_dir,'model.png'))
	
	###############################################################################################################################
	## Save model architecture
	#
	
	logger.info('Saving model architecture')
	with open(os.path.join(out_dir, 'model_arch.json'), 'w') as arch:
		arch.write(model.to_json(indent=2))
	logger.info('  Done')
		
	###############################################################################################################################
	## Evaluator
	#
	evl = Evaluator(dataset, args.prompt_id, out_dir, dev_x, test_x, dev_df, test_df)
	
	###############################################################################################################################
	## Training
	#
	
	logger.info('----------------------------------------------------------------')
	logger.info('Initial Evaluation:'); evl.evaluate(model, -1, print_info=True)
	
	total_train_time = 0
	total_eval_time = 0
	
	for ii in range(args.epochs):
		# Training
		t0 = time()
		train_history = model.fit(train_x, train_y, batch_size=args.batch_size, epochs=1, verbose=0)
		tr_time = time() - t0
		total_train_time += tr_time
		
		# Evaluate
		t0 = time()
		evl.evaluate(model, ii)
		evl_time = time() - t0
		total_eval_time += evl_time
		
		# Print information
		train_loss = train_history.history['loss'][0]
		train_metric = train_history.history[metric_name][0]
		logger.info('Epoch %d, train: %is, evaluation: %is' % (ii, tr_time, evl_time))
		logger.info('[Train] loss: %.4f, metric: %.4f' % (train_loss, train_metric))
		evl.print_info()
	
	###############################################################################################################################
	## Summary of the results
	#
	
	logger.info('Training:   %i seconds in total' % total_train_time)
	logger.info('Evaluation: %i seconds in total' % total_eval_time)
	
	evl.print_final_info()

if __name__ == "__main__":
	
	prompt = '61891';
	# 57452 54147 61693 61915 70086* 61875 
	# 55433 61247 61352* 54735 61923* 
	# 55052 62037 61891*
	
	# 55417* : RANDSEED=9830 300x300 test-qwk= 0.8065 (0.7818) VS TEST KAPPAS (float, int)= 0.8094 (0.7753)
	# 55403* : RANDSEED=8636 300x300 optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6, clipnorm=10)
	# 61891* : 50x100
	#################
	
	dataroot = '/home/david/data/ats/ets'
	datapath = os.path.join(dataroot, prompt)
	
	deepatsroot = '/home/david/code/python/DeepATS'
	outroot = os.path.join(deepatsroot, 'output')
	
	args = '-o output'
	argv = args.split()
	
	argv.append('--prompt'); argv.append(prompt)
	
	#argv.append('--batch-size'); argv.append('32')
	argv.append('--batch-size'); argv.append('64')
	#argv.append('--batch-size'); argv.append('128')
	
	
	argv.append('--loss'); argv.append('kappa')
	#argv.append('--loss'); argv.append('soft_kappa')
	#argv.append('--loss'); argv.append('mse')
	
	argv.append('--emb'); argv.append('/home/david/data/embed/glove.6B.50d.txt')
	#argv.append('--emb'); argv.append('/home/david/data/embed/glove.6B.100d.txt'); argv.append('--embdim'); argv.append('100');
	#argv.append('--emb'); argv.append('/home/david/data/embed/glove.6B.200d.txt'); argv.append('--embdim'); argv.append('200');
	#argv.append('--emb'); argv.append('/home/david/data/embed/glove.6B.300d.txt'); argv.append('--embdim'); argv.append('300');
	
	#argv.append('--emb'); argv.append('/home/david/data/embed/lexvec.commoncrawl.300d.W.pos.neg3.txt'); argv.append('--embdim'); argv.append('300');
	
	#argv.append('--emb'); argv.append('/home/david/data/embed/fasttext.sg.100d.txt'); argv.append('--embdim'); argv.append('100');##BEST
	#argv.append('--emb'); argv.append('/home/david/data/embed/fasttext.sg.200d.m1.txt'); argv.append('--embdim'); argv.append('200');
	#argv.append('--emb'); argv.append('/home/david/data/embed/fasttext.sg.200d.m2.txt'); argv.append('--embdim'); argv.append('200');
	#argv.append('--emb'); argv.append('/home/david/data/embed/fasttext.cb.200d.m2.txt'); argv.append('--embdim'); argv.append('200');
	#argv.append('--emb'); argv.append('/home/david/data/embed/fasttext.6033.200d.txt'); argv.append('--embdim'); argv.append('200');
	
	#argv.append('--emb'); argv.append('/home/david/data/embed/sswe.w5.100d.txt'); argv.append('--embdim'); argv.append('100');
	
	#argv.append('--vocab-size'); argv.append('2560')
	
	argv.append('--rec-unit'); argv.append('rwa')
	#argv.append('--stack'); argv.append('2')
	
	#argv.append('--cnndim'); argv.append('64')
	
	#argv.append('--rnndim'); argv.append('167')
	#argv.append('--rnndim'); argv.append('200')
	#argv.append('--rnndim'); argv.append('250')
	argv.append('--rnndim'); argv.append('100')
		
	#argv.append('--dropout'); argv.append('0.46')
	argv.append('--dropout'); argv.append('0.5')

	#argv.append('--aggregation'); argv.append('attsum')
	#argv.append('--aggregation'); argv.append('attmean')
	
	#argv.append('--type'); argv.append('bregp'); argv.append('--skip-init-bias')
	
	#argv.append('--algorithm'); argv.append('sgd')
	#argv.append('--algorithm'); argv.append('adagrad')
	
	#argv.append('--seed'); argv.append('0')
	argv.append('--seed'); argv.append('4357638')
	
	#argv.append('--skip-emb-preload')
	#argv.append('--tokenize-old')
	argv.append('--min-word-freq'); argv.append('2')
	
	argv.append('--abs-root'); argv.append(deepatsroot)
	argv.append('--abs-data'); argv.append(dataroot)
	argv.append('--abs-out'); argv.append(outroot)
	argv.append('--data-path'); argv.append(datapath)
	
	argv.append('--epochs'); argv.append('100')
	
	run(argv)











