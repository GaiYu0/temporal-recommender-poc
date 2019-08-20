import argparse
import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import libpmf
from generate import generate

def partition(src, dst, p_train, p_val):
    n_train = int(p_train * len(src))
    n_val = int(p_val * len(src))
    permutation = npr.permutation(len(src))
    idx_train = permutation[:n_train]
    idx_val = permutation[n_train : n_train + n_val]
    idx_test = permutation[n_train + n_val:]
    return [src[idx_train], dst[idx_train]], \
           [src[idx_val], dst[idx_val]], \
           [src[idx_test], dst[idx_test]]

def rmse(h_user, h_item, y, uid=None, iid=None):
    u = h_user if uid is None else h_user[uid]
    i = h_item if iid is None else h_item[iid]
    return np.sqrt(np.mean(np.square(np.sum(u * i, 1) - y)))

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--n-ctgrs', type=int, default=100)
parser.add_argument('--n-factors', type=int, default=100)
parser.add_argument('--n-items-per-ctgr', type=int, default=100)
parser.add_argument('--n-items-per-sess', type=int, default=10)
parser.add_argument('--n-sesses', type=int, default=10)
parser.add_argument('--n-users', type=int, default=10000)
parser.add_argument('--p-div', type=float, default=0)
parser.add_argument('--p-train', type=float, default=0.1)
parser.add_argument('--p-val', type=float, default=0.5)
parser.add_argument('--temporal', action='store_true')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()
args.n_items = args.n_ctgrs * args.n_items_per_ctgr

if args.verbose:
    print(args)

[uid, iid], \
[sess_uids, sess_iids] = generate(args.n_users,
                                  args.n_ctgrs,
                                  args.n_items_per_ctgr,
                                  args.n_sesses,
                                  args.n_items_per_sess,
                                  args.p_div)

if args.verbose:
    print('# ratings: %d' % len(uid))

if args.temporal:
    uid = np.hstack([x + i * args.n_users for i, x in enumerate(sess_uids)])
    iid = np.hstack(sess_iids)

[uid_train, iid_train], \
[uid_val, iid_val], \
[uid_test, iid_test] = partition(uid,
                                 iid,
                                 args.p_train,
                                 args.p_val)

h_user, h_item = libpmf.train(uid_train,
                              iid_train,
                              np.ones(len(uid)),
                              args.n_sesses * args.n_users,
                              args.n_items,
                              '-k %d -l %f' % (args.n_factors, args.alpha))

rmse_train = rmse(h_user, h_item, np.ones(len(uid_train)), uid_train, iid_train)
rmse_val = rmse(h_user, h_item, np.ones(len(uid_val)), uid_val, iid_val)
rmse_test = rmse(h_user, h_item, np.ones(len(uid_test)), uid_test, iid_test)
print('%.3e | %.3e | %.3e' % (rmse_train, rmse_val, rmse_test))

if args.temporal:
    def softmax(x, axis):
        exp = np.exp(x - np.max(x, axis, keepdims=True))
        return exp / np.sum(exp, axis, keepdims=True)

    def temporal_rmse(h_user, h_item, uid, iid, y):
        u = h_user[uid]
        i = h_item[iid]
        p = np.expand_dims(softmax(np.sum(u * i, 2), 1), 2)
        return rmse(np.sum(p * u, 1), np.squeeze(i), np.ones(len(uid)))

    u = h_user / (1e-3 + npl.norm(h_user, 2, 1, keepdims=True))
    u = np.stack(np.vsplit(u, args.n_sesses), 1)
    i = h_item / (1e-3 + npl.norm(h_item, 2, 1, keepdims=True))
    i = np.expand_dims(i, 1)
    rmse_train = temporal_rmse(u, i, uid_train % args.n_sesses, iid_train, np.ones(len(uid_train)))
    rmse_val = temporal_rmse(u, i, uid_val % args.n_sesses, iid_val, np.ones(len(uid_val)))
    rmse_test = temporal_rmse(u, i, uid_test % args.n_sesses, iid_test, np.ones(len(iid_test)))
    print('%.3e | %.3e | %.3e' % (rmse_train, rmse_val, rmse_test))
