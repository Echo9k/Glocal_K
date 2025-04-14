from tensorflow.keras.utils import get_file
import numpy as np
import pandas as pd
import h5py
from scipy.sparse import csc_matrix

def load_data_100k(path='./', delimiter='\t'):
    train = np.loadtxt(path + 'movielens_100k_u1.base', skiprows=0, delimiter=delimiter).astype('int32')
    test = np.loadtxt(path + 'movielens_100k_u1.test', skiprows=0, delimiter=delimiter).astype('int32')
    total = np.concatenate((train, test), axis=0)

    n_u = np.unique(total[:, 0]).size  # num of users
    n_m = np.unique(total[:, 1]).size  # num of movies
    n_train = train.shape[0]  # num of training ratings
    n_test = test.shape[0]  # num of test ratings

    train_r = np.zeros((n_m, n_u), dtype='float32')
    test_r = np.zeros((n_m, n_u), dtype='float32')

    for i in range(n_train):
        train_r[train[i, 1] - 1, train[i, 0] - 1] = train[i, 2]

    for i in range(n_test):
        test_r[test[i, 1] - 1, test[i, 0] - 1] = test[i, 2]

    train_m = (train_r > 1e-12).astype('float32')  # masks indicating non-zero entries
    test_m = (test_r > 1e-12).astype('float32')

    print('Data matrix loaded')
    print('Number of users: {}'.format(n_u))
    print('Number of movies: {}'.format(n_m))
    print('Number of training ratings: {}'.format(n_train))
    print('Number of test ratings: {}'.format(n_test))

    return n_m, n_u, train_r, train_m, test_r, test_m

def load_data_1m(path='./', delimiter='::', frac=0.1, seed=1234):
    print('Reading data...')
    data = np.loadtxt(path + 'movielens_1m_dataset.dat', skiprows=0, delimiter=delimiter).astype('int32')

    n_u = np.unique(data[:, 0]).size  # num of users
    n_m = np.unique(data[:, 1]).size  # num of movies
    n_r = data.shape[0]  # num of ratings

    udict = {u: i for i, u in enumerate(np.unique(data[:, 0]))}
    mdict = {m: i for i, m in enumerate(np.unique(data[:, 1]))}

    np.random.seed(seed)
    idx = np.arange(n_r)
    np.random.shuffle(idx)

    train_r = np.zeros((n_m, n_u), dtype='float32')
    test_r = np.zeros((n_m, n_u), dtype='float32')

    for i in range(n_r):
        u_id = data[idx[i], 0]
        m_id = data[idx[i], 1]
        r = data[idx[i], 2]

        if i < int(frac * n_r):
            test_r[mdict[m_id], udict[u_id]] = r
        else:
            train_r[mdict[m_id], udict[u_id]] = r

    train_m = (train_r > 1e-12).astype('float32')  # masks indicating non-zero entries
    test_m = (test_r > 1e-12).astype('float32')

    print('Data matrix loaded')
    print('Number of users: {}'.format(n_u))
    print('Number of movies: {}'.format(n_m))
    print('Number of training ratings: {}'.format(n_r - int(frac * n_r)))
    print('Number of test ratings: {}'.format(int(frac * n_r)))

    return n_m, n_u, train_r, train_m, test_r, test_m

def load_matlab_file(path_file, name_field):
    db = h5py.File(path_file, 'r')
    ds = db[name_field]

    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        out = np.asarray(ds).astype(np.float32).T

    db.close()
    return out

def load_data_monti(path='./'):
    M = load_matlab_file(path + 'douban_monti_dataset.mat', 'M')
    Otraining = load_matlab_file(path + 'douban_monti_dataset.mat', 'Otraining') * M
    Otest = load_matlab_file(path + 'douban_monti_dataset.mat', 'Otest') * M

    n_u = M.shape[0]  # num of users
    n_m = M.shape[1]  # num of movies
    n_train = Otraining[np.where(Otraining)].size  # num of training ratings
    n_test = Otest[np.where(Otest)].size  # num of test ratings

    train_r = Otraining.T
    test_r = Otest.T

    train_m = (train_r > 1e-12).astype('float32')  # masks indicating non-zero entries
    test_m = (test_r > 1e-12).astype('float32')

    print('Data matrix loaded')
    print('Number of users: {}'.format(n_u))
    print('Number of movies: {}'.format(n_m))
    print('Number of training ratings: {}'.format(n_train))
    print('Number of test ratings: {}'.format(n_test))

    return n_m, n_u, train_r, train_m, test_r, test_m