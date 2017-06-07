from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, loguniform, normal, lognormal
from elephas import hyperparam
from pyspark import SparkContext, SparkConf

def data():
    from keras.utils import np_utils
    from keras.preprocessing import sequence
    import keras.backend as K
    import numpy as np
    import pickle as pk
    import os
    
    from deepats.w2vEmbReader import W2VEmbReader as EmbReader
    import deepats.ets_reader as dataset
    from deepats.ets_config import get_args
    
    args = get_args()
        
    emb_reader = EmbReader(args.emb_path, args.emb_dim)
    emb_words = emb_reader.load_words()
    
    train_df, dev_df, test_df, vocab, overal_maxlen, qwks = dataset.get_data(args.data_path, emb_words=emb_words, seed=args.seed)
    
    train_x = train_df['text'].values;    train_y = train_df['y'].values
    dev_x = dev_df['text'].values;         dev_y = dev_df['y'].values
    test_x = test_df['text'].values;    test_y = test_df['y'].values
    
    abs_vocab_file = os.path.join(args.abs_out, 'vocab.pkl')
    with open(abs_vocab_file, 'wb') as vocab_file:
        pk.dump(vocab, vocab_file)
        
    train_x = sequence.pad_sequences(train_x, maxlen=overal_maxlen)
    dev_x = sequence.pad_sequences(dev_x, maxlen=overal_maxlen)
    test_x = sequence.pad_sequences(test_x, maxlen=overal_maxlen)
    
    return train_x, train_y, dev_x, dev_y, test_x, test_y, overal_maxlen, qwks


def model(train_x, train_y, dev_x, dev_y, test_x, test_y, overal_maxlen, qwks):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, GlobalAveragePooling1D
    from keras.layers.embeddings import Embedding
    from keras.layers.recurrent import LSTM
    from keras.initializers import Constant
    from keras import optimizers
    import keras.backend as K
    from deepats.my_layers import MeanOverTime
    from deepats.rwa import RWA    
    import pickle as pk
    import numpy as np
    import string
    import random
    import os
    from deepats.optimizers import get_optimizer
    
    from deepats.ets_evaluator import Evaluator
    import deepats.ets_reader as dataset
    from deepats.ets_config import get_args
    import GPUtil

    def random_id(size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))
    
    def kappa_metric(t,x):
        u = 0.5 * K.sum(K.square(x - t))
        v = K.dot(K.transpose(x), t - K.mean(t))
        return v / (v + u)
    
    def kappa_loss(t,x):
        u = K.sum(K.square(x - t))
        v = K.dot(K.squeeze(x,1), K.squeeze(t - K.mean(t),1))
        return u / (2*v + u)
    
    import time
    ms = int(round(time.time() * 1000))
    rand_seed = ms % (2**32 - 1)
    random.seed(rand_seed)

    args = get_args()
    model_id = random_id()
    
    abs_vocab_file = os.path.join(args.abs_out, 'vocab.pkl')
    with open(abs_vocab_file, 'rb') as vocab_file:
        vocab = pk.load(vocab_file)
    vocab_size = len(vocab)
    
    acts = ['tanh','relu','hard_sigmoid']
    emb_dim = {{choice([50, 100, 200, 300])}}
    rnn_dim = {{uniform(50, 500)}}; rnn_dim = int(rnn_dim)
    rec_act = {{choice([0,1,2])}}; rec_act = acts[rec_act]
    dropout = {{uniform(0.2, 0.95)}}
    
    epochs = args.epochs
    n_emb = vocab_size * emb_dim
    n_rwa = (903 + 2*rnn_dim)*rnn_dim
    n_tot = n_emb + n_rwa + rnn_dim + 1
    
    lr = {{lognormal(-3 * 2.3, .8)}}; lr=1.5*lr
    rho = {{normal(.875, .04)}}
    clipnorm={{uniform(1, 15)}}
    eps={{loguniform(-8*2.3, -5*2.3)}}
    
    opt = optimizers.RMSprop(lr=lr,
                            rho=rho,
                            clipnorm=clipnorm,
                            epsilon=eps)
    loss = kappa_loss
    metric = kappa_metric
    
    evl = Evaluator(dataset, args.prompt_id, args.abs_out, dev_x, test_x, dev_df, test_df, model_id=model_id)
    
    train_y_mean = train_y.mean(axis=0)
    if train_y_mean.ndim == 0:
        train_y_mean = np.expand_dims(train_y_mean, axis=1)
    num_outputs = len(train_y_mean)
    
    mask_zero=False
                   
    model = Sequential()
    model.add(Embedding(vocab_size, emb_dim, mask_zero=mask_zero))
    model.add(RWA(rnn_dim, recurrent_activation=rec_act))
    model.add(Dropout(dropout))
    bias_value = (np.log(train_y_mean) - np.log(1 - train_y_mean)).astype(K.floatx())
    model.add(Dense(num_outputs, bias_initializer=Constant(value=bias_value)))
    model.add(Activation('tanh'))
    model.emb_index = 0
    
    from deepats.w2vEmbReader import W2VEmbReader as EmbReader
    emb_reader = EmbReader(args.emb_path, emb_dim)
    emb_reader.load_embeddings(vocab)
    emb_wts = emb_reader.get_emb_matrix_given_vocab(vocab, model.layers[model.emb_index].get_weights()[0])
    wts = model.layers[model.emb_index].get_weights()
    wts[0] = emb_wts
    model.layers[model.emb_index].set_weights(wts)

    model.compile(loss=loss, optimizer=opt, metrics=[metric])
    model_yaml = model.to_yaml()
    
    import GPUtil
    if GPUtil.avail_mem() < 0.1:
        return {'loss': 1, 'status': STATUS_OK, 'model': '', 'weights': None}
    
    print('model_id: %s' % (model_id))
    print(model_yaml)
    print('PARAMS\t\
    %s\t\
    lr= %.4f\t\
    rho= %.4f\t\
    clip= %.4f\t\
    eps= %.4f\t\
    embDim= %.4f\t\
    rnnDim= %.4f\t\
    drop= %.4f\t\
    recAct= %s' % (model_id, lr, rho, clipnorm, np.log(eps)/2.3, emb_dim, rnn_dim, dropout, rec_act))
    
    for i in range(epochs):
        train_history = model.fit(train_x, train_y, batch_size=args.batch_size, epochs=1, verbose=0)
        evl.evaluate(model, i)
        evl.output_info()
        
        p = evl.stats[3]/qwks[0]
        if i>10 and p<0.9:
            break

    i = evl.comp_idx; j=i+2
    best_dev_kappa = evl.best_dev[i]
    best_test_kappa = evl.best_dev[j]

    print('Test kappa:', best_dev_kappa)
    return {'loss': 1-best_dev_kappa, 'status': STATUS_OK, 'model': model.to_yaml(), 'weights': pk.dumps(model.get_weights())}


# Create Spark context
conf = SparkConf().setAppName('Elephas_Hyperparameter_Optimization').setMaster('local[8]')
sc = SparkContext(conf=conf)

# Define hyper-parameter model and run optimization.
hyperparam_model = hyperparam.HyperParamModel(sc, num_workers=6)
best_model = hyperparam_model.minimize(model=model, data=data, max_evals=100)
best_model_yaml = best_model.to_yaml()
print(best_model_yaml)

