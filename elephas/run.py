from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, loguniform, normal, lognormal
from elephas.hyperparam import HyperParamModel
from pyspark import SparkContext, SparkConf

def data():
    from keras.utils import np_utils
    from keras.preprocessing import sequence
    import keras.backend as K
    import numpy as np
    
    import pickle as pk
    import nea.asap_reader as dataset
    from nea.w2vEmbReader import W2VEmbReader as EmbReader
    from nea.config import get_args
    
    import logging
    logger = logging.getLogger(__name__)
    
    args = get_args()
    
    if args.seed > 0:
        np.random.seed(args.seed)
        
    emb_reader = EmbReader(args.abs_emb_path, emb_dim=args.emb_dim)
    emb_words = emb_reader.load_words()
    
    dataset.set_score_range(args.data_set)
    (train_x, train_y, train_pmt), (dev_x, dev_y, dev_pmt), (test_x, test_y, test_pmt), vocab, vocab_size, overal_maxlen, num_outputs = dataset.get_data((args.train_path, args.dev_path, args.test_path), args.prompt_id, args.vocab_size, args.maxlen, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=args.vocab_path, min_word_freq=args.min_word_freq, emb_words=emb_words)
    
    abs_vocab_file = args.abs_out_path + 'vocab.pkl'
    with open(abs_vocab_file, 'wb') as vocab_file:
        pk.dump(vocab, vocab_file)
    
    train_x = sequence.pad_sequences(train_x, maxlen=overal_maxlen)
    dev_x = sequence.pad_sequences(dev_x, maxlen=overal_maxlen)
    test_x = sequence.pad_sequences(test_x, maxlen=overal_maxlen)
    
    train_y = np.array(train_y, dtype=K.floatx())
    dev_y = np.array(dev_y, dtype=K.floatx())
    test_y = np.array(test_y, dtype=K.floatx())
    
    if args.prompt_id:
        train_pmt = np.array(train_pmt, dtype='int32')
        dev_pmt = np.array(dev_pmt, dtype='int32')
        test_pmt = np.array(test_pmt, dtype='int32')
    
    dev_y_org = dev_y.astype(dataset.get_ref_dtype())
    test_y_org = test_y.astype(dataset.get_ref_dtype())

    train_y = dataset.get_model_friendly_scores(train_y, train_pmt)
    dev_y = dataset.get_model_friendly_scores(dev_y, dev_pmt)
    test_y = dataset.get_model_friendly_scores(test_y, test_pmt)
    
    return train_x, train_y, dev_x, dev_y, test_x, test_y, dev_y_org, test_y_org, overal_maxlen


def model(train_x, train_y, dev_x, dev_y, test_x, test_y, dev_y_org, test_y_org, overal_maxlen):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, GlobalAveragePooling1D
    from keras.layers.embeddings import Embedding
    from keras.layers.recurrent import LSTM
    from keras import optimizers
    import keras.backend as K
    
    import pickle as pk
    import numpy as np
    from nea.optimizers import get_optimizer
    from nea.asap_evaluator import Evaluator
    import nea.asap_reader as dataset
    from nea.config import get_args
    from nea.my_layers import MeanOverTime
    from nea.rwa import RWA
    
    import string
    import random
    def random_id(size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))
    
    import time
    ms = int(round(time.time() * 1000))
    rand_seed = ms % (2**32 - 1)
    random.seed(rand_seed)
       
    args = get_args()
    model_id = random_id()
    
    def kappa_metric(t,x):
        u = 0.5 * K.sum(K.square(x - t))
        v = K.dot(K.transpose(x), t - K.mean(t))
        return v / (v + u)
    
    def kappa_loss(t,x):
        u = K.sum(K.square(x - t))
        v = K.dot(K.squeeze(x,1), K.squeeze(t - K.mean(t),1))
        return u / (2*v + u)
    
    lr = {{lognormal(-3 * 2.3, .8)}}
    lr = lr*2
    rho = {{normal(.875, .04)}}
    clipnorm={{uniform(1, 15)}}
    eps=1e-6
    
    opt = optimizers.RMSprop(lr=lr,
                            rho=rho,
                            clipnorm=clipnorm,
                            epsilon=eps
                            )
    loss = kappa_loss
    metric = kappa_metric
    dataset.set_score_range(args.data_set)
    evl = Evaluator(dataset, args.prompt_id, args.abs_out_path, dev_x, test_x, dev_y, test_y, dev_y_org, test_y_org, model_id=model_id)
    
    abs_vocab_file = args.abs_out_path + 'vocab.pkl'
    with open(abs_vocab_file, 'rb') as vocab_file:
        vocab = pk.load(vocab_file)
    
    train_y_mean = train_y.mean(axis=0)
    if train_y_mean.ndim == 0:
        train_y_mean = np.expand_dims(train_y_mean, axis=1)
    num_outputs = len(train_y_mean)
    
    mask_zero=False
    
    emb_dim = {{choice([50, 100, 200, 300])}}
    rnn_dim = {{uniform(50, 300)}}
    rnn_dim = int(rnn_dim)
                   
    model = Sequential()
    model.add(Embedding(args.vocab_size, emb_dim, mask_zero=mask_zero))
    model.add(RWA(rnn_dim))
    model.add(Dense(num_outputs))
    if not args.skip_init_bias:
        bias_value = (np.log(train_y_mean) - np.log(1 - train_y_mean)).astype(K.floatx())
        model.layers[-1].bias.set_value(bias_value)
    model.add(Activation('tanh'))
    model.emb_index = 0
    
    emb_path = 'embed/glove.6B.{}d.txt'.format(emb_dim)
    abs_emb_path = args.abs_root + emb_path
    
    from nea.w2vEmbReader import W2VEmbReader as EmbReader
    emb_reader = EmbReader(abs_emb_path, emb_dim=emb_dim)
    emb_reader.load_embeddings(vocab)
    emb_wts = emb_reader.get_emb_matrix_given_vocab(vocab, model.layers[model.emb_index].get_weights()[0])
    wts = model.layers[model.emb_index].get_weights()
    wts[0] = emb_wts
    model.layers[model.emb_index].set_weights(wts)

    model.compile(loss=loss, optimizer=opt, metrics=[metric])
    model_yaml = model.to_yaml()
    print('model_id: %s' % (model_id))
    print(model_yaml)
    print('optimizer:    lr= %.4f, rho= %.4f, clipnorm= %.4f, epsilon= %.4f' % (lr, rho, clipnorm, eps))
    
    print('PARAMS\t\
    %s\t\
    lr= %.4f\t\
    rho= %.4f\t\
    clip= %.4f\t\
    emb= %.4f\t\
    rnn= %.4f' % (model_id, lr, rho, clipnorm, emb_dim, rnn_dim))
    
    for i in range(args.epochs):
        train_history = model.fit(train_x, train_y, batch_size=args.batch_size, epochs=1, verbose=0)
        evl.evaluate(model, i)
        evl.output_info()
        if i>5 and evl.dev_metric<0.4:
            break
        if i>10 and evl.dev_metric<0.5:
            break
        if i>15 and evl.dev_metric<0.6:
            break

    best_dev_kappa = evl.best_dev
    best_test_kappa = evl.best_test

    print('Test kappa:', best_dev_kappa)
    return {'loss': 1-best_dev_kappa, 'status': STATUS_OK, 'model': model.to_yaml(), 'weights': pk.dumps(model.get_weights())}


# Create Spark context
conf = SparkConf().setAppName('Elephas_Hyperparameter_Optimization').setMaster('local[8]')
sc = SparkContext(conf=conf)

# Define hyper-parameter model and run optimization.
hyperparam_model = HyperParamModel(sc, num_workers=8)
best_model = hyperparam_model.minimize(model=model, data=data, max_evals=100)
best_model_yaml = best_model.to_yaml()
print(best_model_yaml)

