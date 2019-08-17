#!/usr/bin/env python

from ctypes import *
from os import path

import numpy as np

libpmf = CDLL(path.join(path.dirname(__file__),'pmf_py.so.1'))

class Configs(Structure):
    _names = ['solver_type', 'k', 'threads', 'maxiter', 'maxinneriter',
              'lambda', 'rho', 'eps', 'eta0', 'betaup', 'betadown', 'lrate_method',
              'num_blocks', 'do_predict', 'verbose', 'do_nmf']
    _types = [c_int, c_int, c_int, c_int, c_int, 
              c_double, c_double, c_double, c_double, c_double, c_double, c_int,
              c_int, c_int, c_int, c_int]
    _fields_ = list(zip(_names, _types))

def fill_prototype(f, restype, argtypes): 
    f.restype = restype
    f.argtypes = argtypes

fill_prototype(libpmf.training_option, c_char_p, [])
fill_prototype(libpmf.parse_training_command_line, Configs, [c_char_p])
fill_prototype(libpmf.pmf_train, None, [c_int, c_int, c_long, POINTER(c_int), POINTER(c_int), POINTER(c_double), POINTER(Configs), POINTER(c_double), POINTER(c_double)])

def train(row=None, col=None, dat=None, m=None, n=None, config_str=''):
    configs = libpmf.parse_training_command_line(config_str.encode('ascii'))
    row = np.array(row, dtype=np.int32, copy=False)
    col = np.array(col, dtype=np.int32, copy=False)
    dat = np.array(dat, dtype=np.float64, copy=False)
    if row.max() >= m or col.max() >= n:
        raise RuntimeError()
    u = np.zeros((configs.k, m), dtype=np.float64)
    v = np.zeros((configs.k, n), dtype=np.float64)
    nnz = len(row)
    libpmf.pmf_train(m, n, nnz, row.ctypes.data_as(POINTER(c_int)),
                                col.ctypes.data_as(POINTER(c_int)),
                                dat.ctypes.data_as(POINTER(c_double)), configs, 
                                u.ctypes.data_as(POINTER(c_double)),
                                v.ctypes.data_as(POINTER(c_double)))
    return u.T, v.T
