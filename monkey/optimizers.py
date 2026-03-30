# monkey/optimizers.py
import numpy as np
from . import globals

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, key, param, grad):
        if globals.useNumpy:
            return param - self.lr * grad
        else:
            # scalar
            return param - self.lr * grad


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = {}

    def step(self, key, param, grad):
        if key not in self.m:
            self.m[key] = 0.0 if not globals.useNumpy else np.zeros_like(param)
            self.v[key] = 0.0 if not globals.useNumpy else np.zeros_like(param)
            self.t[key] = 0

        self.t[key] += 1

        if globals.useNumpy:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t[key])
            v_hat = self.v[key] / (1 - self.beta2 ** self.t[key])
            return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        else:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t[key])
            v_hat = self.v[key] / (1 - self.beta2 ** self.t[key])
            return param - self.lr * m_hat / (v_hat ** 0.5 + self.eps)


class RMSProp:
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v = {}

    def step(self, key, param, grad):
        if key not in self.v:
            self.v[key] = 0.0 if not globals.useNumpy else np.zeros_like(param)

        if globals.useNumpy:
            self.v[key] = self.beta * self.v[key] + (1 - self.beta) * (grad ** 2)
            return param - self.lr * grad / (np.sqrt(self.v[key]) + self.eps)
        else:
            self.v[key] = self.beta * self.v[key] + (1 - self.beta) * (grad ** 2)
            return param - self.lr * grad / (self.v[key] ** 0.5 + self.eps)


class AdaGrad:
    def __init__(self, lr=0.01, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.g2 = {}

    def step(self, key, param, grad):
        if key not in self.g2:
            self.g2[key] = 0.0 if not globals.useNumpy else np.zeros_like(param)

        if globals.useNumpy:
            self.g2[key] += grad ** 2
            return param - self.lr * grad / (np.sqrt(self.g2[key]) + self.eps)
        else:
            self.g2[key] += grad ** 2
            return param - self.lr * grad / (self.g2[key] ** 0.5 + self.eps)