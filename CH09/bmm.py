#! /usr/bin/env python
# ! -*- coding=utf-8 -*-
# Project:  Lihang
# Filename: bmm
# Date: 10/24/18
# Author: 😏 <smirk dot cao at gmail dot com>
import numpy as np
import argparse
import logging


class BMM(object):
    def __init__(self,
                 n_components=2,
                 max_iter=100,
                 tol=1e-3,
                 mu=None,
                 alpha=None,
                 verbose=False):
        """

        :param n_components:
        :param max_iter:
        :param tol:
        :param mu: 这里可能要看一下和书中的p,q对应关系, p=0.6, q=0.7, 对应了X=1 X=0的情况, 所以[0.7, 0.6]
        :param alpha:
        :param verbose:
        """
        # k
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        # N
        self.m = 0
        self.n = 0
        self.gamma = None
        self.n_iter_ = 0
        self.mu = mu
        self.alpha = alpha
        self.X = None
        self.label = None
        self.verbose = verbose

    def fit(self,
            X,
            y=None):
        k = self.n_components
        # self.m, self.n = X.shape
        self.m = X.shape[0]
        N = self.m
        # 如果n > 1 , 分解成n个模型训练, 结果再拼回来
        if y is not None:
            self.label = y
        # X: (N, 2)
        self.X = np.eye(k)[X]
        # gamma: (N, k), 样本对子模型的响应度gamma_jk, 按j求和应该是1
        if self.gamma is None:
            self.gamma = np.full((N, k), 1/k)

        # alpha: (k) , 子模型对混合模型的贡献, 求和为1
        if self.alpha is None:
            self.alpha = 0.5
        self.alpha = np.stack((1 - self.alpha, self.alpha), axis=-1)

        # mu: (k, 2) 2是为了做矩阵乘法, 相对for loop效率应该会高, 这里todo: benchmark
        if self.mu is None:
            self.mu = np.full(k, 1/k)

        self.mu = np.stack((1 - self.mu, self.mu), axis=-1)

        for i in range(self.max_iter):
            # print(self.alpha, "\n", self.mu[:, 0], "\n", self.gamma, "\n")
            self.do_e_step()
            self.do_m_step()
            if self.verbose:
                logger.info("alpha %s" % self.alpha)
                logger.info(self.mu)
            if self.is_convergence():
                break

    def is_convergence(self):

        return False

    def density(self):
        # Bernoulli Distribution Density
        rst = np.dot(self.X, self.mu.T)
        return rst

    def do_e_step(self):
        # 更新gamma
        self.gamma = self.density() * self.alpha
        z = np.sum(self.gamma, axis=1).reshape(-1, 1)
        self.gamma /= z
        return self

    def do_m_step(self):
        nk = np.sum(self.gamma, axis=0)
        # update mu, 注意这里X[:, 1] 中的1 来自X的取值为1的意思.
        # p,q 对应的是A=1, B=1, A=0, C=1, 注意这里用样本刷
        self.mu = np.sum(self.gamma * self.X[:, 1].reshape(-1, 1), axis=0) / nk
        self.mu = np.stack((1 - self.mu, self.mu), axis=-1)

        # update alpha
        self.alpha = nk / self.m
        return self

    def predict(self, X):
        pass

    def sample(self,
               n_samples=1):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data file")
    args = vars(ap.parse_args())

else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="path to input data")
    args = vars(ap.parse_args())
