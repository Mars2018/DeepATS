import os
from scipy.stats import pearsonr, spearmanr, kendalltau
import logging
import numpy as np
from deepats.my_kappa_calculator import quadratic_weighted_kappa as qwk

logger = logging.getLogger(__name__)

class Evaluator():
	
	def __init__(self, dataset, prompt_id, out_dir, dev_x, test_x, dev_df, test_df, model_id=''):
		self.dataset = dataset
		self.prompt_id = prompt_id
		self.model_id = model_id
		self.out_dir = out_dir
		self.cur_epoch = -1
		self.dev_x, self.test_x = dev_x, test_x
		self.dev_y, self.test_y = dev_df['y'].values, test_df['y'].values
		self.dev_yint, self.test_yint = dev_df['yint'].values, test_df['yint'].values
		self.ymin, self.ymax = dev_df.ymin, dev_df.ymax
		self.stats = [-1,-1,-1,-1,-1]
		self.best_dev = self.stats
		self.best_test = self.stats
		self.batch_size = 200
				
		self.comp_idx = 0 # 0=relaxed-qwk, 1=int-qwk
	
	def dump_predictions(self, dev_pred, test_pred, epoch):
		np.savetxt(os.path.join(self.out_dir, '/preds/dev_pred_' + str(epoch) + '.txt'), dev_pred, fmt='%.8f')
		np.savetxt(os.path.join(self.out_dir, '/preds/test_pred_' + str(epoch) + '.txt'), test_pred, fmt='%.8f')
	
	def calc_qwk(self, dev_pred, test_pred): # Kappa only supports integer values
		dev_pred_int = np.rint(dev_pred).astype('int32')
		test_pred_int = np.rint(test_pred).astype('int32')
		dev_qwk = qwk(self.dev_yint, dev_pred_int, self.ymin, self.ymax)
		test_qwk = qwk(self.test_yint, test_pred_int, self.ymin, self.ymax)
		return dev_qwk, test_qwk
	
	def rescale(self, x, lo, hi):
		return np.clip(x*(hi-lo)+lo, lo, hi)
	
	def evaluate(self, model, epoch, print_info=False):
		self.cur_epoch = epoch
		self.flip = False
		
		self.dev_loss, self.dev_metric = model.evaluate(self.dev_x, self.dev_y, batch_size=self.batch_size, verbose=2)
		self.test_loss, self.test_metric = model.evaluate(self.test_x, self.test_y, batch_size=self.batch_size, verbose=0)
		
		self.dev_pred = model.predict(self.dev_x, batch_size=self.batch_size).squeeze()
		self.test_pred = model.predict(self.test_x, batch_size=self.batch_size).squeeze()
		
		self.dev_pred = self.rescale(self.dev_pred, self.ymin, self.ymax)
		self.test_pred = self.rescale(self.test_pred, self.ymin, self.ymax)
		self.dev_qwk, self.test_qwk = self.calc_qwk(self.dev_pred, self.test_pred)
		#self.dump_predictions(self.dev_pred, self.test_pred, epoch)
		
		self.stats = [self.dev_metric, self.dev_qwk, self.test_metric, self.test_qwk, epoch]
		
		i = self.comp_idx
		j = i+2
		if self.stats[i] > self.best_dev[i]:
			self.best_dev = self.stats
			model.save_weights(os.path.join(self.out_dir, 'best_model_weights.h5'), overwrite=True)
			self.flip = True
		if self.stats[j] > self.best_test[j]:
			self.best_test = self.stats
			self.flip = True

		#################################################################################################
		
		if print_info:
			self.print_info()
	
	def output_info(self):
		stats = self.stats
		print('[CURRENT  %s]	epoch: %i, dev-qwk= %.4f (%.4f), test-qwk= %.4f (%.4f)' % (self.model_id, self.cur_epoch, stats[0], stats[1], stats[2], stats[3]))
		if self.flip:
			stats = self.best_dev
			print('[BEST DEV %s]	epoch: %i, dev-qwk= %.4f (%.4f), test-qwk= %.4f (%.4f)' % (self.model_id, stats[4], stats[0], stats[1], stats[2], stats[3]))
			stats = self.best_test
			print('[BEST TEST %s]	epoch: %i, dev-qwk= %.4f (%.4f), test-qwk= %.4f (%.4f)' % (self.model_id, stats[4], stats[0], stats[1], stats[2], stats[3]))
			self.flip = False
	
	def print_info(self):
		stats = self.stats
		logger.info('[CURRENT %s]	epoch: %i, dev-qwk= %.4f (%.4f), test-qwk= %.4f (%.4f)' % (self.model_id, self.cur_epoch, stats[0], stats[1], stats[2], stats[3]))
		stats = self.best_dev
		logger.info('[BEST DEV %s]	epoch: %i, dev-qwk= \033[92m%.4f (%.4f)\033[0m, test-qwk= %.4f (%.4f)' % (self.model_id, stats[4], stats[0], stats[1], stats[2], stats[3]))
		stats = self.best_test
		logger.info('[BEST TEST %s]	epoch: %i, dev-qwk= %.4f (%.4f), test-qwk= \033[92m%.4f (%.4f)\033[0m' % (self.model_id, stats[4], stats[0], stats[1], stats[2], stats[3]))
		logger.info('----------------------------------------------------------------')
	
	def print_final_info(self):
		logger.info('----------------------------------------------------------------')
		stats = self.best_dev
		logger.info('[BEST DEV %s]	epoch: %i, dev-qwk= \033[92m%.4f (%.4f)\033[0m, test-qwk= %.4f (%.4f)' % (self.model_id, stats[4], stats[0], stats[1], stats[2], stats[3]))
		stats = self.best_test
		logger.info('[BEST TEST %s]	epoch: %i, dev-qwk= %.4f (%.4f), test-qwk= \033[92m%.4f (%.4f)\033[0m' % (self.model_id, stats[4], stats[0], stats[1], stats[2], stats[3]))
