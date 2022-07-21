# encoding=utf-8

import cv2
import numpy as np

import scipy
from scipy.stats import multivariate_normal


## random blur kernel filtering
####################
# isotropic gaussian kernels, identical to 'fspecial('gaussian',hsize,sigma)' in matlab
####################

def isotropic_gaussian_kernel_matlab(l, sigma, tensor=False):
    """
    :param l:
    :param sigma:
    :param tensor:
    :return:
    """
    center = [(l - 1.0) / 2.0, (l - 1.0) / 2.0]
    [x, y] = np.meshgrid(np.arange(-center[1], center[1] + 1), np.arange(-center[0], center[0] + 1))
    arg = -(x * x + y * y) / (2 * sigma * sigma)
    k = np.exp(arg)

    k[k < scipy.finfo(float).eps * k.max()] = 0

    ## ----- normalize to [0, 1], sum=1
    sum_k = k.sum()
    if sum_k != 0:
        k = k / sum_k

    return torch.FloatTensor(k) if tensor else k


def random_isotropic_gaussian_kernel(l=21,
                                     sig_min=0.2,
                                     sig_max=4.0,
                                     tensor=False):
    """
    :param l:
    :param sig_min:
    :param sig_max:
    :param tensor:
    :return:
    """
    x = np.random.random() * (sig_max - sig_min) + sig_min
    k = isotropic_gaussian_kernel_matlab(l, x, tensor=tensor)
    return k, np.array([x, x, 0])


####################
# random/stable ani/isotropic gaussian kernel batch generation
####################

####################
# anisotropic gaussian kernels, identical to 'mvnpdf(X,mu,sigma)' in matlab
# due to /np.sqrt((2*np.pi)**2 * sig1*sig2), `sig1=sig2=8` != `sigma=8` in matlab
# rotation matrix [[cos, -sin],[sin, cos]]
####################

def anisotropic_gaussian_kernel_matlab(l,
                                       sig1,
                                       sig2,
                                       theta,
                                       tensor=False):
    """
    :param l:
    :param sig1:
    :param sig2:
    :param theta:
    :param tensor:
    :return:
    """
    # mean = [0, 0]
    # v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    # V = np.array([[v[0], v[1]], [v[1], -v[0]]]) # [[cos, sin], [sin, -cos]]
    # D = np.array([[sig1, 0], [0, sig2]])
    # cov = np.dot(np.dot(V, D), V) # VD(V^-1), V=V^-1

    cov11 = sig1 * np.cos(theta) ** 2 + sig2 * np.sin(theta) ** 2
    cov22 = sig1 * np.sin(theta) ** 2 + sig2 * np.cos(theta) ** 2
    cov21 = (sig1 - sig2) * np.cos(theta) * np.sin(theta)
    cov = np.array([[cov11, cov21], [cov21, cov22]])

    center = l / 2.0 - 0.5
    x, y = np.mgrid[-center:-center + l:1, -center:-center + l:1]
    pos = np.dstack((y, x))
    k = scipy.stats.multivariate_normal.pdf(pos, mean=[0, 0], cov=cov)

    k[k < scipy.finfo(float).eps * k.max()] = 0

    # normalize the kernel
    sum_k = k.sum()
    if sum_k != 0:
        k = k / sum_k

    return torch.FloatTensor(k) if tensor else k


def random_anisotropic_gaussian_kernel(l=15,
                                       sig_min=0.2,
                                       sig_max=4.0,
                                       tensor=False):
    """
    :param l:
    :param sig_min:
    :param sig_max:
    :param tensor:
    :return:
    """
    sig1 = sig_min + (sig_max - sig_min) * np.random.rand()
    sig2 = sig_min + (sig1 - sig_min) * np.random.rand()
    theta = np.pi * np.random.rand()

    k = anisotropic_gaussian_kernel_matlab(l=l,
                                           sig1=sig1,
                                           sig2=sig2,
                                           theta=theta,
                                           tensor=tensor)

    return k, np.array([sig1, sig2, theta])


def random_gaussian_kernel(l=21,
                           sig_min=0.2,
                           sig_max=4.0,
                           rate_iso=1.0,
                           tensor=False):
    """
    :param l:
    :param sig_min:
    :param sig_max:
    :param rate_iso: iso gauss kernel rate
    :param tensor:
    :return:
    """
    if np.random.random() < rate_iso:
        return random_isotropic_gaussian_kernel(l=l,
                                                sig_min=sig_min,
                                                sig_max=sig_max,
                                                tensor=tensor)
    else:
        return random_anisotropic_gaussian_kernel(l=l,
                                                  sig_min=sig_min,
                                                  sig_max=sig_max,
                                                  tensor=tensor)