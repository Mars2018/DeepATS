#!/usr/bin/env python

import os
import sys
import argparse
import logging
import numpy as np
import scipy
from time import time
import pickle as pk

from keras.models import model_from_yaml
from keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint

from sklearn.metrics import precision_recall_curve, average_precision_score
from tabulate import tabulate

import deepats.utils as U
from deepats.ets_evaluator import Evaluator

# from deepats import ets_reader
from nlp.readers import ets_reader
from nlp.readers.essay_reader import EssaySetReader


REGEX_NUM = r'^[0-9]*\t[0-9]\t[0-9]\t[0-9]\t(?!\s*$).+'
REGEX_MODE = r'^[0-9]*\tm\tm\tm\t(?!\s*$).+'

logger = logging.getLogger(__name__)

##################################
## set theano/tensorflow in:
## ~/.keras/keras.json
##################################

## kappa loss
import keras.backend as K
#import theano.tensor as T

# numpy sort by col
def sortrows(x,col=0,asc=True):
    n=-1
    if asc:
        n=1
    x=x*n
    return n*x[x[:,col].argsort()]
	
def stats((y,p),n=None):
	pre, rec, _ = precision_recall_curve(y, p)
	pos = int(y.sum())
	tot = len(y)
	if n:
		pos = int(pos * float(n)/tot)
		tot = int(n)
	
	M = []
	for i in range(20):
		r = 1.0 - i*.05
		if r<0.5:
			break
		j = np.argmax(rec<=r)
		pj,rj = pre[j], rec[j]
		tp = pos*rj
		fp = fp = tp/pj - tp
		fn = pos-tp
		tn = tot-tp-fp-fn
		M.append([pj,rj,int(round(fp)),int(round(fn)),int(round(tp)),int(round(tn))])
		if pj==1:
			break
	
	M.insert(0,['Prec','Rec','FP','FN','TP','TN'])
	return M

def down_sample(y,p,q):
	x = np.vstack((y,p)).T
	x = sortrows(x,col=1,asc=False)
	y=x[:,0]
	p=x[:,1]
	f = np.squeeze(np.where(y==1))
	a1,b = y.sum(),len(y)
	a2 = round(b*q)
	yy,pp = y,p
	if a1-a2>0:
		ff = np.random.choice(f,int(a1-a2),replace=False)
		yy=np.delete(y,ff)
		pp=np.delete(p,ff)
	return (yy,pp)

def down_sample_bootstrap(y,p,q,n=1000):
	V = []
	X = []
	for i in np.arange(n):
		(yy,pp) = down_sample(y,p,q)
		V.append(average_precision_score(yy,pp))
		X.append((yy,pp))
	v = np.mean(V)
	f = np.argmax(abs(v-V)==np.min(abs(v-V)))
	return X[f]

def print_table(y, p, q, n=1000):
	print('\n{0}% Off-Mode (sample size= {1}):'.format(int(q*100),n))
	print tabulate(stats(down_sample_bootstrap(y,p,q),n), headers="firstrow", floatfmt='.3f')
		
def print_tables(y, p, Q=[0.1,0.05,0.01], n=1000):
	print_table(y, p, q, n=n)

# 	print tabulate(stats((y,p),1000), headers="firstrow", floatfmt='.3f')
# 	print tabulate(stats(down_sample(y,p,0.01),1000), headers="firstrow", floatfmt='.3f')
# 	print tabulate(stats(down_sample_bootstrap(y,p,0.01),1000), headers="firstrow", floatfmt='.3f')


# def stats(y,p,T):
# 	x = np.vstack((y,p)).T
# 	x = sortrows(x,col=1,asc=False)
# 	y=x[:,0]
# 	p=x[:,1]
# 	t=T[0]
# 	yy=(p>t).astype('float')
# 	m = confusion_matrix(y, yy)
# 	fp,fn,tp,tn = m[0,1],m[1,0],m[0,0],m[1,1]
	
def kappa(t,x):
	u = 0.5 * K.sum(K.square(x - t))
	v = K.dot(K.transpose(x), t - K.mean(t))
	return v / (v + u)

## theano
def kappa_loss(t,x):
	u = K.sum(K.square(x - t))
	v = K.dot(K.squeeze(x,1), K.squeeze(t - K.mean(t),1))##v = T.tensordot(x, t - K.mean(t))
	return u / (2*v + u)# =1-qwk

# numpy kappa
def qwk(t,x):
	u = 0.5 * np.sum(np.square(x - t))
	v = np.dot(np.transpose(x), t - np.mean(t))
	return v / (v + u)

###############################################################################################################################
## Parse arguments
#
def run(argv=None):

	parser = argparse.ArgumentParser()
	parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")
	parser.add_argument("-p", "--prompt", dest="prompt_id", type=str, metavar='<str>', required=False, help="Promp ID for ASAP dataset. '0' means all prompts.")
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
	parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=0, help="Random seed (default=1234)")
	parser.add_argument("--mode", dest="run_mode", type=str, metavar='<str>', default='train', help="run mode")
	
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
	
	
	pid = args.prompt_id
	mode = args.run_mode
	
	#######################
	
	#from deepats.util import GPUtils as GPU
# 	import GPUtil as GPU
# 	mem = GPU.avail_mem()
# 	logger.info('AVAIL GPU MEM == %.4f' % mem)
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
	
	vocab_path = None
	abs_vocab_file = os.path.join(out_dir, 'vocab.pkl')
	if mode=='test':
		vocab_path = abs_vocab_file
	
	train_df, dev_df, test_df, vocab, overal_maxlen = ets_reader.get_mode_data(	args.data_path,
																				dev_split=0.1,
																				emb_words=emb_words,
																				vocab_path=vocab_path,
																				seed=RANDSEED)
	
	train_x = train_df['text'].values;	train_y = train_df['yint'].values.astype('float32')
	dev_x = dev_df['text'].values; 		dev_y = dev_df['yint'].values.astype('float32')
	test_x = test_df['text'].values;	test_y = test_df['yint'].values.astype('float32')
	
	# Dump vocab
	if mode=='train':
		with open(os.path.join(out_dir, 'vocab.pkl'), 'wb') as vocab_file:
			pk.dump(vocab, vocab_file)
	
	if args.recurrent_unit == 'rwa':
		setattr(args, 'model_type', 'rwa')
	if args.recurrent_unit == 'lstm':
		setattr(args, 'model_type', 'regp')
		
	# Pad sequences for mini-batch processing
	from keras.preprocessing import sequence
	
	train_x = sequence.pad_sequences(train_x, maxlen=overal_maxlen)
	dev_x = sequence.pad_sequences(dev_x, maxlen=overal_maxlen)
	test_x = sequence.pad_sequences(test_x, maxlen=overal_maxlen)

	
	###############################################################################################################################
	## Some statistics
	
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
	logger.info('  PROMPT_ID\t= ' + U.b_green(args.prompt_id))
	logger.info('  RANDSEED\t= ' + U.b_green(str(RANDSEED)))
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
	
	from keras import optimizers
	from deepats.optimizers import get_optimizer
	#optimizer = get_optimizer(args)
	
# 	optimizer = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=10)#***RWA***
	
	optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1)

# 	optimizer = optimizers.Nadam(lr=0.001, clipnorm=10)
# 	optimizer = optimizers.Nadam(lr=0.002, clipnorm=1)
	
# 	optimizer = optimizers.RMSprop(lr=0.0015, rho=0.9, epsilon=1e-8, clipnorm=10)
# 	optimizer = optimizers.RMSprop(lr=0.003, rho=0.88, epsilon=1e-6, clipnorm=10)
# 	optimizer = optimizers.RMSprop(lr=0.0025, rho=0.8, epsilon=1e-8, clipnorm=10) # best 2.1 (RWA)

	
	## OTHER METHODS
	#optimizer = optimizers.Adam(lr=0.0018, clipnorm=5)
	#optimizer = optimizers.Nadam(lr=0.002)
	#optimizer = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=10)
	#optimizer = optimizers.SGD(lr=0.05, momentum=0, decay=0.0, nesterov=False, clipnorm=10)
	#optimizer = optimizers.Adagrad(lr=0.03, epsilon=1e-08, clipnorm=10)
	
	###############################################################################################################################
	## Building model
	from deepats.models import create_model
	
	#loss = kappa_loss
	#metrics = [kappa,'mean_squared_error']
	
	if args.loss == 'mse':
		loss = 'mean_squared_error'
		metrics = ['acc']
# 		metrics = [kappa]
		monitor='val_kappa'
	elif args.loss == 'kappa':
		loss = kappa_loss
		metrics = [kappa]
		monitor='val_kappa'
	
	model = create_model(args, train_y.mean(axis=0), overal_maxlen, vocab)
	model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
	print(model.summary())
	
	###############################################################################################################################
	## Callbacks
	callbacks=[]
	
	##############################
	''' Evaluate test_kappa '''
	
	from sklearn.metrics import roc_auc_score as auc, average_precision_score
	def map(y_true, y_prob):
		return average_precision_score(y_true, y_prob)
	
	class Eval(Callback):
	    def __init__(self, x, y, funcs, prefix='test', batch_size=128):
	        super(Eval, self).__init__()
	        self.x=x
	        self.y=y
	        self.funcs = funcs
	        self.prefix = prefix
	        self.batch_size = batch_size
	        self.epoch = 0
	        
	    def on_epoch_end(self, epoch, logs={}):
	    	self.epoch+=1
	    	p = np.asarray(self.model.predict(self.x, batch_size=self.batch_size).squeeze())
	    	for func in self.funcs:
	    		f = func(self.y, p)
	    		name = '{}_{}'.format(self.prefix, func.__name__)
	    		logs[name] = f
	    		print(' - {0}: {1:0.4f}'.format(name,f))
	    		#sys.stdout.write(' - {0}: {1:0.4f} '.format(name,f))
	
	eval = Eval(dev_x, dev_df['yint'].values, [map], 'val'); callbacks.append(eval)
	monitor = 'val_map'
	
# 	eval = Eval(test_x, test_df['yint'].values, [qwk,auc], 'test'); callbacks.append(eval)
	eval = Eval(test_x, test_df['yint'].values, [map,qwk], 'test'); callbacks.append(eval)
# 	monitor = 'test_map'

	##############################
	''' ModelCheckpoint '''
	
	wt_path= os.path.join(out_dir, 'weights.{}.hdf5'.format(pid))
	checkpt = ModelCheckpoint(wt_path, monitor=monitor, verbose=1, save_best_only=True, mode='max')
	callbacks.append(checkpt)
	
	##############################
	''' PR Curve '''
	from sklearn.metrics import precision_recall_curve
	import matplotlib.pyplot as plt
	
	class PR(object):
	    def __init__(self, model, checkpoint, x, y, prefix='test', batch_size=128):
	        self.model=model
	        self.checkpoint=checkpoint
	        self.x=x
	        self.y=y
	        self.prefix = prefix
	        self.batch_size = batch_size
	        
	    def predict(self):
	    	self.model.load_weights(self.checkpoint.filepath)
	    	self.p = np.asarray(self.model.predict(self.x, batch_size=self.batch_size).squeeze())
	        
	    def pr_curve(self, y, p, s=''):
	    	aps = average_precision_score(y, p)
	    	precision, recall, _ = precision_recall_curve(y, p)
	    	name = '{}_{}'.format(self.prefix, 'pr_curve')
	    	plt.figure()
	    	plt.step(recall, precision, color='b', alpha=0.2, where='post')
	    	plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
	    	plt.xlabel('Recall')
	    	plt.ylabel('Precision')
	    	plt.ylim([0.0, 1.05])
	    	plt.xlim([0.0, 1.0])
	    	plt.title('PR curve (mode={1}): {2}, AUC={0:0.4f}'.format(aps, pid, s))
	    
	    def run_sample(self, q, n=1000):
	    	(y,p) = down_sample_bootstrap(self.y, self.p, q, n)
	    	## draw curve
	    	self.pr_curve(y, p, s='{0}% off-mode'.format(int(q*100)))
	    	## make table
	    	print('\nMode={2}, {0}% off-mode (#samples={1}):'.format(int(q*100), n, pid))
	    	print tabulate(stats((y,p),n), headers="firstrow", floatfmt='.3f')
	    	
	    
	    def run(self, Q=[0.1, 0.01]):
	    	self.predict()
	    	for q in Q:
	    		self.run_sample(q)
	    	return self.y,self.p
		
	pr = PR(model, checkpt, test_x, test_df['yint'].values, 'test')
	    		
	##############################
	''' LRplateau '''
	
	class LRplateau(ReduceLROnPlateau):
	    def __init__(self, monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=1e-4, cooldown=0, min_lr=0,checkpoint=None):
	        super(LRplateau, self).__init__(monitor,factor,patience,verbose,mode,epsilon,cooldown,min_lr)
	        self.checkpoint=checkpoint
	        
	    def on_lr_reduce(self, epoch):
	    	if self.checkpoint:
	    		if self.verbose > 0:
	    			print('Epoch {}: loading wts from {}.\n'.format(epoch, self.checkpoint.filepath))
	    		self.model.load_weights(self.checkpoint.filepath)
	    		
	    def on_epoch_end(self, epoch, logs=None):
	        logs = logs or {}
	        logs['lr'] = K.get_value(self.model.optimizer.lr)
	        current = logs.get(self.monitor)
	        if current is None:
	            warnings.warn('Learning Rate Plateau Reducing requires %s available!' %
	                          self.monitor, RuntimeWarning)
	        else:
	            if self.in_cooldown():
	                self.cooldown_counter -= 1
	                self.wait = 0
	
	            if self.monitor_op(current, self.best):
	                self.best = current
	                self.wait = 0
	            elif not self.in_cooldown():
	                if self.wait >= self.patience:
	                    old_lr = float(K.get_value(self.model.optimizer.lr))
	                    if old_lr > self.min_lr + self.lr_epsilon:
	                        new_lr = old_lr * self.factor
	                        new_lr = max(new_lr, self.min_lr)
	                        K.set_value(self.model.optimizer.lr, new_lr)
	                        if self.verbose > 0:
	                            print('\nEpoch {0}: reducing learning rate to {1:0.4g}.'.format(epoch, new_lr))
	                        self.cooldown_counter = self.cooldown
	                        self.wait = 0
	                        self.on_lr_reduce(epoch)
	                self.wait += 1
	
	reduce_lr = LRplateau(	monitor=monitor,
							mode='max',
							patience=3,
							factor=0.33,
							min_lr=0.00001,
							verbose=1,
							checkpoint=checkpt)
	
	callbacks.append(reduce_lr)
	

	###############################################################################################################################
	## Training
	if mode=='train':
		model.fit(	train_x,train_y,
					validation_data=(dev_x, dev_df['yint'].values),
					batch_size=args.batch_size, 
					epochs=args.epochs,
					callbacks=callbacks,
					verbose=1)
	
	## Evaluate ###############################################
	y,p = pr.run(Q=[0.2,0.1,0.05])
	return y,p
	
	
	###############################################################################################################################

def make_dataset(	mode, ids,
					data_dir='/home/david/data/ets1b/2016',
					out_dir = '/home/david/data/ets1b/2016/off_mode',
# 					out_dir = '/home/david/data/ets1b/2016/temp',
					vocab_file='vocab_n250.txt',
					Nnum = 2000,
					Nmode = 1000,
					train_test_split = 0.8,
					train_test_overlap=True
				):
	
	out_dir = os.path.join(out_dir, mode)
	vocab_file = os.path.join(data_dir, vocab_file)
	essay_files = []

	for id in ids:
	    idstr = '{}'.format(id)
	    essay_file = os.path.join(data_dir, idstr, idstr + '.txt.clean.tok')
	    essay_files.append(essay_file)
	
	word_vocab, char_vocab, max_word_length = load_vocab(vocab_file)
	num_reader = EssaySetReader(essay_files, word_vocab, char_vocab, chunk_size=4000, regex=REGEX_NUM)
	mode_reader = EssaySetReader(essay_files, word_vocab, char_vocab, chunk_size=5000, regex=REGEX_MODE)

	text = []
	train_ids = []
	test_ids = []
	
	def add_samples(reader, label, N):
		n = 0
		for r in reader.record_stream(stop=True):
			n += 1
			if n>N:
				break
			id = r[0]
			txt = r[2]
			text.append('{}\t{}\t{}'.format(id, label, txt))
			if np.random.rand() < train_test_split:
				train_ids.append(id)
			else:
				test_ids.append(id)
		return n
	
	nm = add_samples(mode_reader, 1, Nmode)
	if nm<Nmode:
		Nnum = Nnum*nm/Nmode
	add_samples(num_reader, 0, Nnum)
	print('n={}'.format(len(text)))
	
	random.shuffle(text)
	train_ids.sort()
	test_ids.sort()
	
	def write_to_file(lines, dir, name):
		with open(os.path.join(dir,name), 'w') as f:
			for line in lines:
				f.write(str(line) + '\n')
	
	write_to_file(text, out_dir, 'text.txt')
	write_to_file(train_ids, out_dir, 'train_ids.txt')
	write_to_file(test_ids, out_dir, 'test_ids.txt')

if __name__ == "__main__":
	make_data = False
	
	# 	MAP
# 	prompt = 'arg'# 0.925(1/4) [100x300] pool=2
# 	prompt = 'exp'# 
# 	prompt = 'inf'# 
# 	prompt = 'nar'# 
	prompt = 'opi'# 0.95 (1/4) [100x100] pool=3(2)

# 	make_data = True
	
	run_mode = 'train'
# 	run_mode = 'test'

	######################################################
	
	if make_data and run_mode=='train':
		if prompt == 'arg':
			make_dataset('arg',
						[54145, 55050, 55401, 55413, 56342, 56523, 57244, 57456, 61088, 61697, 61703, 61711, 61861, 67556],
						vocab_file='vocab_n250.txt', Nnum = 4000, Nmode = 1000
						)
		if prompt == 'exp':
			make_dataset('exp',
						[54135,54151,54741,55064,55074,55373,55405,55419,57517,61659,61665,
						61673,61681,61719,61897,61913,61921,61929,61985,61993,70088],
						vocab_file='vocab_n250.txt', Nnum = 8000, Nmode = 2000
						)
		if prompt == 'inf':
			make_dataset('inf',
						[54183, 54703, 55381, 55427, 55435, 56356, 56525, 57186, 57236, 57440, 57472, 57513, 61332, 61342, 62003],
						vocab_file='vocab_n250.txt', Nnum = 4000, Nmode = 1000
						)
		if prompt == 'nar':
			make_dataset('nar',
						[54191,54209,54697,55385,55393,55415,55425,55431,56354,56379,56529,57392,57398,
						57446,57482,61334,61452,61689,61727,61835,61843,61881,61937,62033,62047,62051],
						vocab_file='vocab_n250.txt', Nnum = 4000, Nmode = 1000
						)
		if prompt == 'opi':
			make_dataset('opi',
		 				[54201,55387,56365,57190,57478,61426,61442,61851,62059],
		 				vocab_file='vocab_n250.txt', Nnum = 4000, Nmode = 1000
		 				)
 	
# 	sys.exit()
	######################################################
	
	dataroot = '/home/david/data/ets1b/2016/off_mode'
	datapath = os.path.join(dataroot, prompt)
	
	deepatsroot = '/home/david/code/python/DeepATS'
	outroot = os.path.join(deepatsroot, 'output')
	
	args = '-o output'
	argv = args.split()
	
	argv.append('--prompt'); argv.append(prompt)
	argv.append('--mode'); argv.append(run_mode)
	
# 	argv.append('--batch-size'); argv.append('32')
	argv.append('--batch-size'); argv.append('64')
# 	argv.append('--batch-size'); argv.append('128')
	
	argv.append('--loss'); argv.append('mse')
# 	argv.append('--loss'); argv.append('kappa')
	
# 	argv.append('--rec-unit'); argv.append('rwa')
# 	argv.append('--stack'); argv.append('2')
	
# 	argv.append('--emb'); argv.append('/home/david/data/embed/glove.6B.50d.txt'); argv.append('--embdim'); argv.append('50');
	argv.append('--emb'); argv.append('/home/david/data/embed/glove.6B.100d.txt'); argv.append('--embdim'); argv.append('100');
# 	argv.append('--emb'); argv.append('/home/david/data/embed/glove.6B.200d.txt'); argv.append('--embdim'); argv.append('200');
# 	argv.append('--emb'); argv.append('/home/david/data/embed/glove.6B.300d.txt'); argv.append('--embdim'); argv.append('300');
	
	#argv.append('--vocab-size'); argv.append('2560')
	#argv.append('--cnndim'); argv.append('64')
	
	#argv.append('--rnndim'); argv.append('167')
	#argv.append('--rnndim'); argv.append('200')
	#argv.append('--rnndim'); argv.append('250')
	argv.append('--rnndim'); argv.append('300')
		
	#argv.append('--dropout'); argv.append('0.46')
	argv.append('--dropout'); argv.append('0.5')

	#argv.append('--aggregation'); argv.append('attsum')
	#argv.append('--aggregation'); argv.append('attmean')
	
	#argv.append('--type'); argv.append('bregp'); argv.append('--skip-init-bias')
	
	#argv.append('--algorithm'); argv.append('sgd')
	#argv.append('--algorithm'); argv.append('adagrad')
	
# 	argv.append('--seed'); argv.append('1831')
	
	#argv.append('--skip-emb-preload')
	#argv.append('--tokenize-old')
	argv.append('--min-word-freq'); argv.append('2')
	
	argv.append('--abs-root'); argv.append(deepatsroot)
	argv.append('--abs-data'); argv.append(dataroot)
	argv.append('--abs-out'); argv.append(outroot)
	argv.append('--data-path'); argv.append(datapath)
	
	argv.append('--epochs'); argv.append('100')
	
	y,p = run(argv)

