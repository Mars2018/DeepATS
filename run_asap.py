#!/usr/bin/env python

##################################
## set theano/tensorflow in:
## ~/.keras/keras.json
##################################

import argparse
import logging
import numpy as np
import scipy
from time import time
import sys
import deepats.utils as U
import pickle as pk
from deepats import asap_reader

from keras.models import model_from_yaml

logger = logging.getLogger(__name__)


###############################################
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
	parser.add_argument("-tr", "--train", dest="train_path", type=str, metavar='<str>', required=True, help="The path to the training set")
	parser.add_argument("-tu", "--tune", dest="dev_path", type=str, metavar='<str>', required=True, help="The path to the development set")
	parser.add_argument("-ts", "--test", dest="test_path", type=str, metavar='<str>', required=True, help="The path to the test set")
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
	parser.add_argument("--stack", dest="stack", type=float, metavar='<float>', default=1, help="how deep to stack core RNN")
	parser.add_argument("--skip-emb-preload", dest="skip_emb_preload", action='store_true', help="Skip preloading embeddings")
	parser.add_argument("--tokenize-old", dest="tokenize_old", action='store_true', help="use old tokenizer")
	
	parser.add_argument("-ar", "--abs-root", dest="abs_root", type=str, metavar='<str>', required=False, help="Abs path to root directory")
	parser.add_argument("-ad", "--abs-data", dest="abs_data", type=str, metavar='<str>', required=False, help="Abs path to data directory")
	parser.add_argument("-ao", "--abs-out", dest="abs_out", type=str, metavar='<str>', required=False, help="Abs path to output directory")
	##
	
	if argv is None:
		args = parser.parse_args()
	else:
		args = parser.parse_args(argv)
		
	setattr(args, 'abs_emb_path', args.abs_root + args.emb_path)
	
	out_dir = args.out_dir_path
	
	U.mkdir_p(out_dir + '/preds')
	U.set_logger(out_dir)
	U.print_args(args)
	
	assert args.model_type in {'reg', 'regp', 'breg', 'bregp', 'rwa'}
	assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}
	assert args.loss in {'mse', 'mae', 'kappa', 'soft_kappa'}
	assert args.recurrent_unit in {'lstm', 'gru', 'simple', 'rwa'}
	assert args.aggregation in {'mot', 'attsum', 'attmean'}
	
	if args.seed > 0:
		np.random.seed(args.seed)
	
	if args.tokenize_old:
		asap_reader.token = 0
		logger.info('using OLD tokenizer!')
		
	if args.prompt_id>=0:
		from deepats.asap_evaluator import Evaluator
		import deepats.asap_reader as dataset
	else:
		raise NotImplementedError
	
	###############################################################################################################################
	## Prepare data
	#
	
	emb_words = None
	if not args.skip_emb_preload:#if args.emb_path:
		from deepats.w2vEmbReader import W2VEmbReader as EmbReader
		logger.info('Loading embedding vocabulary...')
		emb_reader = EmbReader(args.emb_path, emb_dim=args.emb_dim)
		emb_words = emb_reader.load_words()
	
	from keras.preprocessing import sequence
	
	# data_x is a list of lists
	(train_x, train_y, train_pmt), (dev_x, dev_y, dev_pmt), (test_x, test_y, test_pmt), vocab, vocab_size, overal_maxlen, num_outputs = dataset.get_data(
		(args.train_path, args.dev_path, args.test_path), 
		args.prompt_id, 
		args.vocab_size, 
		args.maxlen, 
		tokenize_text=True, 
		to_lower=True, 
		sort_by_len=False, 
		vocab_path=args.vocab_path, 
		min_word_freq=args.min_word_freq, 
		emb_words=emb_words)
	
	# Dump vocab
	with open(out_dir + '/vocab.pkl', 'wb') as vocab_file:
		pk.dump(vocab, vocab_file)
	
	if args.recurrent_unit == 'rwa':
		setattr(args, 'model_type', 'rwa')
		
	# Pad sequences for mini-batch processing
	if args.model_type in {'breg', 'bregp', 'rwa'}:
		assert args.rnn_dim > 0
		train_x = sequence.pad_sequences(train_x, maxlen=overal_maxlen)
		dev_x = sequence.pad_sequences(dev_x, maxlen=overal_maxlen)
		test_x = sequence.pad_sequences(test_x, maxlen=overal_maxlen)
		#assert args.recurrent_unit == 'lstm'
	else:
		train_x = sequence.pad_sequences(train_x)
		dev_x = sequence.pad_sequences(dev_x)
		test_x = sequence.pad_sequences(test_x)
	
	###############################################################################################################################
	## Some statistics
	#
	
	import keras.backend as K
	
	train_y = np.array(train_y, dtype=K.floatx())
	dev_y = np.array(dev_y, dtype=K.floatx())
	test_y = np.array(test_y, dtype=K.floatx())
	
	if args.prompt_id:
		train_pmt = np.array(train_pmt, dtype='int32')
		dev_pmt = np.array(dev_pmt, dtype='int32')
		test_pmt = np.array(test_pmt, dtype='int32')
	
	bincounts, mfs_list = U.bincounts(train_y)
	with open('%s/bincounts.txt' % out_dir, 'w') as output_file:
		for bincount in bincounts:
			output_file.write(str(bincount) + '\n')
	
	train_mean = train_y.mean(axis=0)
	train_std = train_y.std(axis=0)
	dev_mean = dev_y.mean(axis=0)
	dev_std = dev_y.std(axis=0)
	test_mean = test_y.mean(axis=0)
	test_std = test_y.std(axis=0)
	
	logger.info('Statistics:')
	
	logger.info('  train_x shape: ' + str(np.array(train_x).shape))
	logger.info('  dev_x shape:   ' + str(np.array(dev_x).shape))
	logger.info('  test_x shape:  ' + str(np.array(test_x).shape))
	
	logger.info('  train_y shape: ' + str(train_y.shape))
	logger.info('  dev_y shape:   ' + str(dev_y.shape))
	logger.info('  test_y shape:  ' + str(test_y.shape))
	
	logger.info('  train_y mean: %s, stdev: %s, MFC: %s' % (str(train_mean), str(train_std), str(mfs_list)))
	
	logger.info('  overal_maxlen:  ' + str(overal_maxlen))
	
	# We need the dev and test sets in the original scale for evaluation
	dev_y_org = dev_y.astype(dataset.get_ref_dtype())
	test_y_org = test_y.astype(dataset.get_ref_dtype())
	train_y_org = train_y.astype(dataset.get_ref_dtype())
	
	# Convert scores to boundary of [0 1] for training and evaluation (loss calculation)
	train_y = dataset.get_model_friendly_scores(train_y, train_pmt)
	dev_y = dataset.get_model_friendly_scores(dev_y, dev_pmt)
	test_y = dataset.get_model_friendly_scores(test_y, test_pmt)
	
	###############################################################################################################################
	## Optimizaer algorithm
	#
	
	from deepats.optimizers import get_optimizer
	#optimizer = get_optimizer(args)
	
	from keras import optimizers
	
	## RMS-PROP
	
	#optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6, clipnorm=10)
	optimizer = optimizers.RMSprop(lr=0.0018, rho=0.88, epsilon=1e-6, clipnorm=10)
		
	#optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, clipnorm=10)
	#optimizer = optimizers.RMSprop(lr=0.004, rho=0.85, epsilon=1e-6, clipnorm=10)# best 2.1 (RWA)
	#optimizer = optimizers.RMSprop(lr=0.0025, rho=0.8, epsilon=1e-8, clipnorm=10) # best 2.1 (RWA)
	#optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, clipnorm=10) # best 2.3 (RWA)
	#optimizer = optimizers.RMSprop(lr=0.0025, rho=0.88, epsilon=1e-8, clipnorm=10) # best 2.3 (RWA)
	#optimizer = optimizers.RMSprop(lr=0.004, rho=0.85, epsilon=1e-8, clipnorm=10) # best 2.10 (RWA)
	
	## OTHER METHODS
	#optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=10)
	#optimizer = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-06, clipnorm=10)
	
	#optimizer = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=10)
	#optimizer = optimizers.SGD(lr=0.05, momentum=0, decay=0.0, nesterov=False, clipnorm=10)
	#optimizer = optimizers.Adagrad(lr=0.03, epsilon=1e-08, clipnorm=10)
	#optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06, clipnorm=10)
	
	###############################################################################################################################
	## Building model
	#
	
	from deepats.models import create_model
	N=1; L=0
	if args.loss == 'mse':
		loss = 'mean_squared_error'
		#metric = 'mean_absolute_error'; metric_name = metric
		metric = kappa_metric; metric_name = 'kappa_metric'
	elif args.loss == 'mae':
		loss = 'mean_absolute_error'
		#metric = 'mean_squared_error'; metric_name = metric
		metric = kappa_metric; metric_name = 'kappa_metric'
	elif args.loss == 'kappa':
		loss = kappa_loss
		metric = kappa_metric; metric_name = 'kappa_metric'

	########################################################
	
	if N>1:
		train_y_hot = np.eye(N)[train_y_org-L].astype('float32')
		train_y = train_y_hot
	
	########################################################
	
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
# 	plot(model, to_file = out_dir + '/model.png')
	
	###############################################################################################################################
	## Save model architecture
	#
	
	logger.info('Saving model architecture')
	with open(out_dir + '/model_arch.json', 'w') as arch:
		arch.write(model.to_json(indent=2))
	logger.info('  Done')
		
	###############################################################################################################################
	## Evaluator
	#
	
	evl = Evaluator(dataset, args.prompt_id, out_dir, dev_x, test_x, dev_y, test_y, dev_y_org, test_y_org, N, L)
	
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
	
	#data = 'asap1'; asap_reader.asap_ranges = asap_reader.asap1_ranges
	data = 'asap2'; asap_reader.asap_ranges = asap_reader.asap2_ranges
	
	prompt = '10';
	fold = '0'
	
	####################################################################################################
		
	#dataroot = 'data/'+data+'/fold_'+fold+'/'
	asap_reader.set_score_range(data)
	
	deepatsroot = '/home/david/code/python/deepats/'
	outroot = deepatsroot + 'output/'
	
	asaproot = '/home/david/data/ats/asap/'
	dataroot = asaproot + data + '/fold_' +fold+ '/'
	
	args = '-tr '+dataroot+'train.tsv -tu '+dataroot+'dev.tsv -ts '+dataroot+'test.tsv -o output'
	argv = args.split()
	
	argv.append('--prompt'); argv.append(prompt)
	
	#argv.append('--batch-size'); argv.append('64')
	argv.append('--batch-size'); argv.append('64')
	
	
	argv.append('--loss'); argv.append('kappa')
	#argv.append('--loss'); argv.append('soft_kappa')
	#argv.append('--loss'); argv.append('mse')
	
	#argv.append('--emb'); argv.append('/home/david/data/embed/glove.6B.50d.txt')
	#argv.append('--emb'); argv.append('/home/david/data/embed/glove.6B.100d.txt'); argv.append('--embdim'); argv.append('100');
	#argv.append('--emb'); argv.append('/home/david/data/embed/glove.6B.200d.txt'); argv.append('--embdim'); argv.append('200');
	argv.append('--emb'); argv.append('/home/david/data/embed/glove.6B.300d.txt'); argv.append('--embdim'); argv.append('300');
	
	#argv.append('--emb'); argv.append('/home/david/data/embed/lexvec.commoncrawl.300d.W.pos.neg3.txt'); argv.append('--embdim'); argv.append('300');
	
	#argv.append('--emb'); argv.append('/home/david/data/embed/fasttext.sg.100d.txt'); argv.append('--embdim'); argv.append('100');##BEST
	#argv.append('--emb'); argv.append('/home/david/data/embed/fasttext.sg.200d.m1.txt'); argv.append('--embdim'); argv.append('200');
	#argv.append('--emb'); argv.append('/home/david/data/embed/fasttext.sg.200d.m2.txt'); argv.append('--embdim'); argv.append('200');
	#argv.append('--emb'); argv.append('/home/david/data/embed/fasttext.cb.200d.m2.txt'); argv.append('--embdim'); argv.append('200');
	#argv.append('--emb'); argv.append('/home/david/data/embed/fasttext.6033.200d.txt'); argv.append('--embdim'); argv.append('200');
	
	#argv.append('--emb'); argv.append('/home/david/data/embed/sswe.w5.100d.txt'); argv.append('--embdim'); argv.append('100');
	
	#argv.append('--vocab-size'); argv.append('2560')
	
	argv.append('--rec-unit'); argv.append('rwa')
	#argv.append('--stack'); argv.append('3')
	
	#argv.append('--cnndim'); argv.append('64')
	#argv.append('--rnndim'); argv.append('167')
	argv.append('--rnndim'); argv.append('167')
	
	#argv.append('--stack'); argv.append('-3.25')
	
	#argv.append('--dropout'); argv.append('0.46')
	argv.append('--dropout'); argv.append('0.5')

	#argv.append('--aggregation'); argv.append('attsum')
	#argv.append('--aggregation'); argv.append('attmean')
	
	#argv.append('--type'); argv.append('bregp'); argv.append('--skip-init-bias')
	
	#argv.append('--algorithm'); argv.append('sgd')
	#argv.append('--algorithm'); argv.append('adagrad')
	
	#argv.append('--seed'); argv.append('629692')
	
	#argv.append('--skip-emb-preload')
	#argv.append('--tokenize-old')
	argv.append('--min-word-freq'); argv.append('2')
	
	argv.append('--abs-root'); argv.append(deepatsroot)
	argv.append('--abs-data'); argv.append(dataroot)
	argv.append('--abs-out'); argv.append(outroot)
	
	argv.append('--epochs'); argv.append('100')
	
	run(argv)











