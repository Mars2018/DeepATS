import os
import sys
import random
import re
import numpy as np
import pandas as pd
import keras.backend as K
import codecs
import nltk
import logging
import pickle as pk
from sklearn.cross_validation import train_test_split

from my_kappa_calculator import quadratic_weighted_kappa as qwk

logger = logging.getLogger(__name__)
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'

def set_score_range(data_set):
	global asap_ranges
	asap_ranges = range_dict[data_set]

token = 1

def get_ref_dtype():
	return ref_scores_dtype

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

def tokenize_NEW(string):
	tokens = tokenizer.tokenize(string)
	for index, token in enumerate(tokens):
		if token == '@' and (index+1) < len(tokens):
			tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
			tokens.pop(index)
	return tokens

def tokenize_OLD(string):
	tokens = nltk.word_tokenize(string)
	for index, token in enumerate(tokens):
		if token == '@' and (index+1) < len(tokens):
			tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
			tokens.pop(index)
	return tokens

def tokenize(string):
	if token > 0:
		return tokenize_NEW(string)
	return tokenize_OLD(string)

def is_number(token):
	return bool(num_regex.match(token))

def load_vocab(vocab_path):
	logger.info('Loading vocabulary from: ' + vocab_path)
	with open(vocab_path, 'rb') as vocab_file:
		vocab = pk.load(vocab_file)
	return vocab

def create_vocab(lines, tokenize_text, to_lower, min_word_freq, emb_words=None, maxlen=0):
	logger.info('Creating vocabulary...')
	if maxlen > 0:
		logger.info('  Removing sequences with more than ' + str(maxlen) + ' words')
	total_words, unique_words = 0, 0
	word_freqs = {}

	for content in lines:
		if to_lower:
			content = content.lower()
		if tokenize_text:
			content = tokenize(content)
		else:
			content = content.split()
		if maxlen > 0 and len(content) > maxlen:
			continue
		for word in content:
			try:
				word_freqs[word] += 1
			except KeyError:
				unique_words += 1
				word_freqs[word] = 1
			total_words += 1
					
	logger.info('  %i total words, %i unique words' % (total_words, unique_words))
	##
	#for key in sorted(word_freqs):
	#	print "%s\t%d" % (key, word_freqs[key])
	##
	import operator
	sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)

	# Choose vocab size automatically by removing infrequent words (freq < min_word_freq)
	vocab_size = 0
	for word, freq in sorted_word_freqs:
		if freq >= min_word_freq:
			vocab_size += 1
				
	vocab = {'<pad>':0, '<unk>':1, '<num>':2}
	vcb_len = len(vocab)
	index = vcb_len
	for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
		if emb_words != None:
			if word not in emb_words: continue
		vocab[word] = index
		index += 1
	return vocab


def tokenize_dataset(lines, vocab, tokenize_text, to_lower):
	data = []
	num_hit, unk_hit, total = 0., 0., 0.

	for content in lines:
		if to_lower:
			content = content.lower()
		if tokenize_text:
			content = tokenize(content)
		else:
			content = content.split()
		indices = []
		for word in content:
			if is_number(word):
				indices.append(vocab['<num>'])
				num_hit += 1
			elif word in vocab:
				indices.append(vocab[word])
			else:
				indices.append(vocab['<unk>'])
				unk_hit += 1
			total += 1
		data.append(indices)
					
	logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))
	return data

def qwok(t,y):
	mu = t.mean()
	t = t-mu
	y = y-mu
	k = 2*t.dot(y)/(t.dot(t) + y.dot(y))
	return k

def get_data(data_path, 
			dev_split=0.12, 
			tokenize_text=True, 
			to_lower=True, 
			vocab_path=None, 
			min_word_freq=2, 
			emb_words=None,
			seed=1234
			):
	
	train_id_file = os.path.join(data_path, 'train_ids.txt')
	test_id_file = os.path.join(data_path, 'test_ids.txt')
	text_file = os.path.join(data_path, 'text.txt')
	
	df = pd.read_csv(text_file, sep="\t", header = None, encoding='utf-8').sort_values(by=0)
	df.columns = ['id','yint','text']
	train_ids = np.genfromtxt(train_id_file, dtype=np.int32)
	
	df_test = pd.read_csv(test_id_file, sep="\t", header = None).sort_values(by=0)
	test_ids = df_test[0].values.astype('int32')
	
	yint_test = df_test[1].values.astype('int32')
	y_test = df_test[2].values.astype('float32')
	
	df2 = df.loc[df['id'].isin(test_ids)]
	t_test = df2['yint'].values.astype('float32')
	
	# compute 2 kappas
	k1 = qwk(t_test.astype('int32'), yint_test, df['yint'].min(), df['yint'].max())
	k2 = qwok(t_test, y_test)
	
     # stratified train/dev split
	df2 = df.loc[df['id'].isin(train_ids)]
	y = df2.pop('yint')
	X_train, X_dev, y_train, y_dev = train_test_split( df2, y, stratify=y, test_size=dev_split, random_state=seed)
	train_ids = X_train['id'].values
	dev_ids = X_dev['id'].values
	train_ids.sort(); dev_ids.sort(); test_ids.sort()
	
	'''
	# get frequency counts 
	from scipy.stats import itemfreq
	itemfreq(y_train)
	itemfreq(y_dev)
	'''
	
# 	# random dev split
# 	n = int(dev_split*len(train_ids))
# 	random.shuffle(train_ids)
# 	dev_ids = train_ids[:n]
# 	train_ids = train_ids[n:]
	
	# add normalized scores
	y = df['yint'].values.astype(K.floatx())
	ymin = y.min().astype('int32'); ymax = y.max().astype('int32')
	yy = (y-ymin)/(ymax-ymin)
	df.insert(1, column='y', value=yy)
	
	train_df = df.loc[df['id'].isin(train_ids)]
	dev_df = df.loc[df['id'].isin(dev_ids)]
	test_df = df.loc[df['id'].isin(test_ids)]
	
	train_df.ymin = ymin; train_df.ymax = ymax
	dev_df.ymin = ymin; dev_df.ymax = ymax
	test_df.ymin = ymin; test_df.ymax = ymax
	
	vocab = create_vocab(train_df['text'].values, tokenize_text, to_lower, min_word_freq, emb_words)
	logger.info('  Vocab size: %i' % (len(vocab)))
	
	pd.options.mode.chained_assignment = None
	train_df.loc[:,'text'] = tokenize_dataset(train_df['text'].values, vocab, tokenize_text, to_lower)
	dev_df.loc[:,'text'] = tokenize_dataset(dev_df['text'].values, vocab, tokenize_text, to_lower)
	test_df.loc[:,'text'] = tokenize_dataset(test_df['text'].values, vocab, tokenize_text, to_lower)
	
	train_maxlen = train_df['text'].map(len).max()
	dev_maxlen = dev_df['text'].map(len).max()
	test_maxlen = test_df['text'].map(len).max()
	overal_maxlen = max(train_maxlen, dev_maxlen, test_maxlen)
	
	return train_df, dev_df, test_df, vocab, overal_maxlen, (k1,k2)

if __name__ == '__main__':
	from w2vEmbReader import W2VEmbReader as EmbReader
	emb_reader = EmbReader('/home/david/data/embed/glove.6B.50d.txt', emb_dim=50)
	emb_words = emb_reader.load_words()
	
	train_df, dev_df, test_df, vocab, overal_maxlen, qwks = get_data('/home/david/data/ats/ets/54147', emb_words=emb_words)
	print(qwks)
	
	print('Done.')
	