# encoding=utf-8

import os

import cv2
import matplotlib
import numpy as np
import scipy

matplotlib.use('PS')
import matplotlib.pyplot as plt
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
    k = multivariate_normal.pdf(pos, mean=[0, 0], cov=cov)

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


def plot_kernel(out_k_np, save_path, gt_k_np=None):
    """
    :param out_k_np:
    :param save_path:
    :param gt_k_np:
    :return:
    """
    plt.clf()

    if gt_k_np is None:
        ax = plt.subplot(111)
        im = ax.imshow(out_k_np, vmin=out_k_np.min(), vmax=out_k_np.max())
        plt.colorbar(im, ax=ax)
    else:
        ax = plt.subplot(121)
        im = ax.imshow(gt_k_np, vmin=gt_k_np.min(), vmax=gt_k_np.max())
        plt.colorbar(im, ax=ax)
        ax.set_title('GT Kernel')

        ax = plt.subplot(122)
        im = ax.imshow(out_k_np, vmin=gt_k_np.min(), vmax=gt_k_np.max())
        plt.colorbar(im, ax=ax)
        ax.set_title('Kernel PSNR: {:.2f}'
                     .format(calculate_kernel_psnr(out_k_np, gt_k_np)))

    plt.show()
    plt.savefig(save_path)


def draw_kernel(kernel_img, save_kernel_path, ratio=15):
    """
    :param kernel_img:
    :param save_kernel_path:
    :return:
    """
    kernel_img = cv2.normalize(kernel_img, None, 0, 255, cv2.NORM_MINMAX)
    kernel_img = kernel_img.round()
    kernel_img = kernel_img.astype(np.uint8)

    k_h, k_w = kernel_img.shape
    if ratio > 0:
        ## ----- INTER_NEAREST
        kernel_img = cv2.resize(kernel_img,
                                (int(k_w * ratio), int(k_h * ratio)),
                                cv2.INTER_AREA)

    cv2.imwrite(save_kernel_path, kernel_img)
    print("[Info]: {:s} saved.".format(save_kernel_path))


def test_blur():
    """
    :return:
    """
    img_name = "test_plate.jpg"
    img_dir = "E:/PyProjs/TestExperiments"
    img_path = os.path.abspath(img_dir + "/" + img_name)
    if not os.path.isfile(img_path):
        print("[Err]: invalid image path: {:s}".format(img_path))
        exit(-1)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h, w, c = img.shape
    if max(w, h) > 1000:
        img = cv2.resize(img, (int(w // 2), int(h // 2)), cv2.INTER_AREA)

    for i in range(10):
        ## ----- generate random blurring kernel
        k_size = np.random.randint(3, 10)  # [3, 7]
        kernel, sigma = random_gaussian_kernel(l=k_size,
                                               sig_min=0.5,
                                               sig_max=7,
                                               rate_iso=0.2,
                                               tensor=False)

        ## ----- apply blurring
        blurred_img = cv2.filter2D(img, -1, kernel)
        ## -----

        kernel_img = kernel.copy()
        kernel_img = (kernel_img * 255.0).round()
        kernel_img = kernel_img.astype(np.uint8)

        # k_h, k_w = kernel_img.shape
        # if kernel_img.shape[0] < 10:
        #     kernel_img = cv2.resize(kernel_img, (k_w * 10, k_h * 10), cv2.INTER_AREA)
        # elif 10 < kernel_img.shape[0] < 20:
        #     kernel_img = cv2.resize(kernel_img, (k_w * 10, k_h * 5), cv2.INTER_AREA)
        # elif 20 < kernel_img.shape[0] < 30:
        #     kernel_img = cv2.resize(kernel_img, (k_w * 10, k_h * 3), cv2.INTER_AREA)

        win_name = "blurring, kernel_size: {:d}".format(k_size)

        # cv2.imshow(win_name, blurred_img)
        # cv2.waitKey()

        save_img_path = "./blurred_{:d}_{:d}.png".format(i, k_size)
        cv2.imwrite(save_img_path, blurred_img)

        # cv2.imshow("kernel size: {:d}".format(k_size), kernel_img)
        save_kernel_path = "./kernel_{:d}_{:d}.png".format(i, k_size)
        # plot_kernel(kernel_img, save_kernel_path)  ## PIL
        draw_kernel(kernel_img, save_kernel_path)  ## OpenCV

    print("Done.")


if __name__ == "__main__":
    test_blur()
    print("Done.")
