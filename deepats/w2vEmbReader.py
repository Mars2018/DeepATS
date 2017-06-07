import sys
import codecs
import logging
import numpy as np
import string

logger = logging.getLogger(__name__)

class W2VEmbReader:
	def __init__(self, emb_path, emb_dim=None):
		#logger.info('Loading embeddings from: ' + emb_path)
		emb_path = emb_path.format(emb_dim)
		self.emb_path=emb_path
		self.emb_dim_request=emb_dim
		self.has_header=False
		self.words = None
		with codecs.open(emb_path, 'r', encoding='utf8') as emb_file:
			tokens = emb_file.next().split()
			if len(tokens) == 2:
				try:
					int(tokens[0])
					int(tokens[1])
					self.has_header = True
				except ValueError:
					pass
		#self.load_embeddings()
	
	def load_embeddings(self, vocab=None):
		logger.info('Loading embeddings from: ' + self.emb_path)
		self.words = None
		if vocab != None:
			self.words = set(vocab.keys())
		if self.has_header:
			with codecs.open(self.emb_path, 'r', encoding='utf8') as emb_file:
				tokens = emb_file.next().split()
				assert len(tokens) == 2, 'The first line in W2V embeddings must be the pair (vocab_size, emb_dim)'
				self.vocab_size = int(tokens[0])
				self.emb_dim = int(tokens[1])
				#print self.vocab_size, self.emb_dim
				assert self.emb_dim == self.emb_dim_request, 'The embeddings dimension does not match with the requested dimension'
				self.embeddings = {}
				counter = 0
				for line in emb_file:
					word = string.split(line,maxsplit=1)[0]
					if not self.check_word(word): continue
					tokens = line.split()
					assert len(tokens) == self.emb_dim + 1, 'The number of dimensions does not match the header info'
					#word = tokens[0]
					vec = tokens[1:]
					self.embeddings[word] = vec
					counter += 1
			if self.words is None:
				assert counter == self.vocab_size, 'Vocab size does not match the header info'
			else:
				self.vocab_size = counter
		else:
			with codecs.open(self.emb_path, 'r', encoding='utf8') as emb_file:
				self.vocab_size = 0
				self.emb_dim = -1
				self.embeddings = {}
				for line in emb_file:
					word = string.split(line,maxsplit=1)[0]
					if not self.check_word(word): continue
					tokens = line.split()
					if self.emb_dim == -1:
						self.emb_dim = len(tokens) - 1
						assert self.emb_dim == self.emb_dim_request, 'The embeddings dimension does not match with the requested dimension'
					else:
						assert len(tokens) == self.emb_dim + 1, 'The number of dimensions does not match the header info'
					#word = tokens[0]
					vec = tokens[1:]
					self.embeddings[word] = vec
					self.vocab_size += 1
		
		logger.info('  #vectors: %i, #dimensions: %i' % (self.vocab_size, self.emb_dim))
	
	def load_words(self):
		logger.info('Loading words only from: ' + self.emb_path)
		with codecs.open(self.emb_path, 'r', encoding='utf8') as emb_file:
			if self.has_header:
				tokens = emb_file.next().split()
				assert len(tokens) == 2, 'The first line in W2V embeddings must be the pair (vocab_size, emb_dim)'
				self.vocab_size = int(tokens[0])
				logger.info('  #words: %i' % self.vocab_size)
			counter = 0
			self.words = set()
			for line in emb_file:
				#tokens = line.split()
				#word = tokens[0]
				word = string.split(line,maxsplit=1)[0]
				self.words.add(word)
				counter += 1
		assert counter == self.vocab_size, 'Vocab size does not match the header info'
		return self.words
	
	def check_word(self, word):
		if self.words == None:
			return True
		if word in self.words:
			return True
		return False
	
	def get_emb_given_word(self, word):
		try:
			return self.embeddings[word]
		except KeyError:
			return None
	
	def get_emb_matrix_given_vocab(self, vocab, emb_matrix):
		counter = 0.
		for word, index in vocab.iteritems():
			try:
				emb_matrix[index] = self.embeddings[word]
				counter += 1
			except KeyError:
				pass
		logger.info('%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(vocab), 100*counter/len(vocab)))
		return emb_matrix
	
	def get_emb_dim(self):
		return self.emb_dim
	
	
	
	
