import os
import sys
import pandas as pd
import numpy as np
import glob
import re
from shutil import copyfile
#from numpy.ma.core import ids

'''
sudo mount -t cifs -o username=dvaughn //spshare.miat.co/Users /media/david/spshare
'''

class Configuration(object):
    """Dump stuff here"""
CONFIG = Configuration()

CONFIG.item = '57436' # '61354'

CONFIG.score_rgx = 'scores[0-9][0-9]*.txt'

CONFIG.derek_folders = ['mod2_11bw',
                        'mod2_12bw',
                        'mod2_13bw',
                        #'mod2_9'
                        #'mod2_10',
                        #'mod2_11',
                        ]

CONFIG.source_scores_path = '/media/david/spshare/djustice/ETS_2017/%s/scores' % CONFIG.derek_folders[1]

CONFIG.derek_data_path = '/media/david/spshare/djustice/brief_writes_strat__17_02_09'
CONFIG.tara_data_path = '/media/david/spshare/tgilmore/ETS_Output/train'

CONFIG.ets_data_path = '/home/david/data/ats/ets'
CONFIG.tmp_data_path = '/home/david/temp'
CONFIG.source_data_path = CONFIG.derek_data_path

def extract_item_id(s):
    #re.findall(r"\D(\d*)\D", s)
    try:
        return re.findall(r"scores(\d*)_*[0-9]*.txt", s)[0]
    except IndexError:
        return None

def score_files(path=CONFIG.source_scores_path):
    files = glob.glob(os.path.join(path, CONFIG.score_rgx))
    files.sort()
    return files

def score_file(item, path=CONFIG.source_scores_path):
    files = glob.glob(os.path.join(CONFIG.source_scores_path, 'scores' + item +'.txt'))
    if len(files)==1:
        return files[0]
    return None

def data_file(item, path=CONFIG.source_data_path):
    files = glob.glob(os.path.join(path, item +'.txt'))
    if len(files)==1:
        return files[0]
    return None

def tmp_file(id):
    return os.path.join(CONFIG.tmp_data_path, id +'.txt')

def clean_data_file(fin):
    lines = open(fin).read().splitlines()
    fout = tmp_file('xxx')
    fo = open(fout, "w+")
    for line in lines:
        fo.write(line.strip().replace('"','') + '\n')
    fo.close()
    return fout

def get_test_ids(file):
    d = pd.read_csv(file, sep="\t", header = None).sort_values(by=0)
    data = d.loc[d[3]==2]#[0].as_matrix()
    return data
    
def get_all_ids(file):
    f = clean_data_file(file)
    d = pd.read_csv(f, sep="\t", header = None).sort_values(by=0)
    ids = d[0].as_matrix()
    return ids

def get_data_file(score_file):
    item = extract_item_id(score_file)
    
    test_data = get_test_ids(score_file)
    test_ids = test_data[0].values
    test_yint = test_data[1].values
    test_y = test_data[2].values
    
    dfiles = []
    df = data_file(item, path= CONFIG.tara_data_path)
    if df: dfiles.append(df)
    df = data_file(item, path= CONFIG.derek_data_path)
    if df: dfiles.append(df)
        
    for dfile in dfiles:
        #print(dfile)
        all_ids = get_all_ids(dfile)
        x = np.in1d(test_ids, all_ids)
        x = test_ids[~x]
        if len(x)==0:
            return (test_data, all_ids, dfile)
    
    #return '!!\t'+item+'\t%d' % len(x)
    return None
    
def check(path=CONFIG.source_scores_path):
    files = score_files(path=path)
    for sf in files:
        df = get_data_file(sf)
        print('\t' + df)
    
def check_all():
    for df in CONFIG.derek_folders:
        print(df)
        scores_path = '/media/david/spshare/djustice/ETS_2017/%s/scores' % df
        check(scores_path)
    print('Done')
    
def split_data(sf):
    ret = get_data_file(sf)
    if ret is None:
        print('!!!\t' + sf)
        return
    item = extract_item_id(sf)
    
    test_data = ret[0]
    test_ids = test_data[0].values
    test_yint = test_data[1].values
    test_y = test_data[2].values
    
    all_ids = ret[1]
    df = ret[2]
    train_ids = np.setdiff1d(all_ids, test_ids)
    
    dir = os.path.join(CONFIG.ets_data_path, item)
    if not os.path.exists(dir): os.makedirs(dir)
    
    df = clean_data_file(df)
    dst = os.path.join(dir, 'text.txt')
    copyfile(df, dst)
    
    dst = os.path.join(dir, 'train_ids.txt')
    with open(dst, 'w') as f:
        f.writelines("%d\n" % l for l in train_ids)
    
    dst = os.path.join(dir, 'test_ids.txt')
    with open(dst, 'w') as f:
        #f.writelines("%d\t%d\t%.4f\n" % l for l in test_ids)
        for i in range(len(test_ids)):
            f.write("%d\t%d\t%.4f\n" % (test_ids[i], test_yint[i], test_y[i]))
    
def make_folds(path=CONFIG.source_scores_path):
    files = score_files(path=path)
    for sf in files:
        print(sf)
        split_data(sf)
        #sys.exit()
        
def make_all_folds():
    for df in CONFIG.derek_folders:
        scores_path = '/media/david/spshare/djustice/ETS_2017/%s/scores' % df
        make_folds(scores_path)
    print('Done')

if __name__ == '__main__':
    #check()
    #check_all()
    make_all_folds()