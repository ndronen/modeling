#!/usr/bin/env python

import sys
import argparse
import os.path
import cPickle
from itertools import product

import theano
import pylearn2
from pylearn2.config import yaml_parse

import numpy as np
from numpy.random import multivariate_normal as mvnormal
from numpy.random import uniform
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from sklearn.covariance import MinCovDet, EmpiricalCovariance
from sklearn.decomposition import PCA

###########################################################################
# This class was useful for simulating data sets while developing
# this script. 
# means = [[0, 0]]
# cov = [[2, 1], [1, 2]]
# n = 5000
# mvn = PMeansMultivariateNormal(n, means, cov)
# X = mvn.generate()
# X.shape
# np.savetxt(X, file='simulated.csv')
###########################################################################
class PMeansMultivariateNormal(object):
    def __init__(self, means, cov, size):
        self.__dict__.update(locals())
        del self.self
        #self.n = n
        #self.means = means
        #self.cov = cov

    def generate(self):
        return mvnormal(self.means, self.cov, self.size)
        '''
        X = np.empty(shape=(self.n*len(self.means), 2))
        for i, mean in enumerate(self.means):
            idx = range(i*self.n, i*self.n+self.n)
            x, y = mvnormal(mean, self.cov, self.n).T
            X[idx, 0] = x
            X[idx, 1] = y
        return X
        '''

def reconstruction_error(a, b):
    return ((a - b)**2).sum(axis=1)

def train_autoencoder(dataset_path, nvis=2, nhid=2, act_enc=None, act_dec=None):
    yaml = open('outliers.yaml', 'r').read()
    if act_enc is None:
        act_enc = 'null'
    else:
        act_enc = "'" + act_enc + "'"

    if act_dec is None:
        act_dec = 'null'
    else:
        act_dec = "'" + act_dec + "'"

    params = {
        'dataset_path': dataset_path,
        'nvis': nvis,
        'nhid': nhid,
        'act_enc': act_enc,
        'act_dec': act_dec,
        'learning_rate': 0.05,
        'save_path': 'outliers.pkl'
    }
    
    yaml = yaml % (params)

    train = yaml_parse.load(yaml)
    train.main_loop()
    
    pkl = open('outliers.pkl')
    return cPickle.load(pkl)

class NullTransformer(object):
    def fit(self, X):
        pass

    def fit_transform(self, X):
        return X

    def transform(self, X):
        return X

def main():
    parser = argparse.ArgumentParser(
        description='Plot outlier-like distances for a 2-dimensional dataset')
    parser.add_argument(
        'dataset', type=argparse.FileType('r'),
        help='a CSV file containing the dataset')
    parser.add_argument(
        '--plot', type=str, choices=['train', 'grid'], default='grid',
        help='plot the dataset or a grid evenly distributed over its span')
    parser.add_argument(
        '--plotdims', type=int, choices=[2, 3], default=2,
        help='the number of dimensions to plot')

    args = parser.parse_args()

    X = np.loadtxt(args.dataset, delimiter=',')
    fig = plt.figure()

    xformer = NullTransformer()

    if X.shape[1] > 2:
        xformer = PCA(n_components=2)
        X = xformer.fit_transform(X)

    if args.plotdims == 2:
        plt.scatter(X[:, 0], X[:, 1], s=60, linewidth='0')
    else:
        plt.scatter(X[:, 0], X[:, 1])
    plt.show(block=False)

    path_to_script = os.path.realpath(__file__)
    dir_of_script = os.path.dirname(path_to_script)
    dataset_path = dir_of_script + '/outliers.npy'
    np.save(dataset_path, X)
    
    ###########################################################################
    # Train autoencoder with the n samples until convergence.  Run
    # evenly distributed samples through the autoencoder and compute
    # their reconstruction error.
    ###########################################################################

    maxseq_orig = np.max(X)
    minseq_orig = np.min(X)
    seqrange = np.abs(maxseq_orig - minseq_orig)
    maxseq = maxseq_orig + 0.5 * seqrange
    minseq = minseq_orig - 0.5 * seqrange
    print("minseq", minseq, "maxseq", maxseq)
    if args.plot == 'grid':
        seq = np.linspace(minseq, maxseq, num=50, endpoint=True)
        Xplot = np.array([_ for _ in product(seq, seq)])
    else:
        Xplot = X

    robust_cov = MinCovDet().fit(X)
    robust_md = robust_cov.mahalanobis(Xplot)

    empirical_cov = EmpiricalCovariance().fit(X)
    empirical_md = empirical_cov.mahalanobis(Xplot)

    # Assume Xplot is at least 2-dimensional.
    if Xplot.shape[1] > 2:
        Xplot2d = bh_sne(Xplot)
    else:
        Xplot2d = Xplot

    robust_md01 = robust_md - np.nanmin(robust_md)
    robust_md01 = robust_md01 / np.nanmax(robust_md01)

    empirical_md01 = empirical_md - np.nanmin(empirical_md)
    empirical_md01 = empirical_md01 / np.nanmax(empirical_md01)

    fig = plt.figure()
    if args.plotdims == 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(Xplot2d[:, 0], Xplot2d[:, 1], 
            cmap=plt.cm.jet, c=robust_md01, s=60, linewidth='0')
    else:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_trisurf(Xplot2d[:, 0], Xplot2d[:, 1], robust_md01,
            cmap=plt.cm.jet, color=robust_md01)
        ax.set_zlabel('Mahalanobis distance')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Mahalanobis distance (robust covariance)')

    fig = plt.figure()
    if args.plotdims == 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(Xplot2d[:, 0], Xplot2d[:, 1], 
            cmap=plt.cm.jet, c=empirical_md01, s=60, linewidth='0')
    else:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_trisurf(Xplot2d[:, 0], Xplot2d[:, 1], empirical_md01,
            cmap=plt.cm.jet, color=empirical_md01)
        ax.set_zlabel('Mahalanobis distance')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Mahalanobis distance (empirical covariance)')
    
    enc_dec = [
        # tanh encoder, linear decoder
        ['tanh', 'linear'],
        # sigmoid encoder, linear decoder
        ['sigmoid', 'linear'],
        #######################################################################
        # The reconstruction error of the autoencoders trained with the
        # remaining commented-out pairs don't seem to match Mahalanobis
        # distance very well.  Feel free to uncomment them to see for
        # yourself.
        # linear encoder, linear decoder
        # ['linear', 'linear'],
        # tanh encoder, tanh decoder
        # ['tanh', 'tanh'],
        # tanh encoder, sigmoid decoder
        # ['tanh', 'sigmoid'],
        # sigmoid encoder, tanh decoder
        # ['sigmoid', 'tanh'],
        # sigmoid encoder, sigmoid decoder
        # ['sigmoid', 'sigmoid']
        #######################################################################
    ]
    
    for i, act in enumerate(enc_dec):
        enc, dec = act
        if dec == 'linear':
            dec = None
        model = train_autoencoder(dataset_path,
            act_enc=enc, act_dec=dec, nvis=X.shape[1], nhid=16)
        
        Xshared = theano.shared(
            np.asarray(Xplot, dtype=theano.config.floatX), borrow=True)
        f = theano.function([], outputs=model.reconstruct(Xshared))
        fit = f()
        error = reconstruction_error(Xplot, fit)

        error01 = error - np.nanmin(error)
        error01 = error01 / np.nanmax(error01)
        
        fig = plt.figure()
        if args.plotdims == 2:
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(Xplot2d[:, 0], Xplot2d[:, 1],
                cmap=plt.cm.jet, c=error, s=60, linewidth='0')
        else:
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.plot_trisurf(Xplot2d[:, 0], Xplot2d[:, 1], error,
                cmap=plt.cm.jet, color=error01)
            ax.set_zlabel('Reconstruction error')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        encdec_type = ', '.join(act) 
        ax.set_title('Reconstruction error (' + encdec_type + ')')

        print("Correlation of robust MD and reconstruction error (" +
            str(encdec_type) + ") " + str(pearsonr(robust_md, error)))
        print("Correlation of empirical MD and reconstruction error (" +
            str(encdec_type) + ") " + str(pearsonr(empirical_md, error)))

    print("Correlation of robust MD and empirical MD " +
        str(pearsonr(robust_md, empirical_md)))

    os.remove(dataset_path)
    os.remove('outliers.pkl')

    plt.show(block=True)

if __name__ == '__main__':
    sys.exit(main())
