# encoding=utf-8

import os

import cv2
import numpy as np
from TestBlur import random_gaussian_kernel


def random_sharpening(img, low=3, high=50, weight=0.5, threshold=10):
    """USM sharpening. borrowed from real-ESRGAN
    Input image: I; Blurry image: B.
    1. K = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * K + (1 - Mask) * I
    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int): residual threshold
    """
    img = img.astype(np.float32)
    img = cv2.normalize(img, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    img = np.clip(img, 0.0, 1.0)

    # blur = cv2.GaussianBlur(img, (radius, radius), 0)

    ## ----- generate random blurring kernel
    radius = np.random.randint(low, high + 1)
    k_size = radius * 2 + 1  # [3, 7]
    k_size = k_size if k_size >= 3 else 3
    kernel, sigma = random_gaussian_kernel(l=k_size,
                                           sig_min=0.5,
                                           sig_max=7,
                                           rate_iso=0.2,
                                           tensor=False)

    ## ----- apply blurring
    blur = cv2.filter2D(img, -1, kernel)
    ## -----

    residual = img - blur
    mask = np.abs(residual) * 255.0 > threshold
    mask = mask.astype(np.float32)

    kernel, sigma = random_gaussian_kernel(l=k_size,
                                           sig_min=0.5,
                                           sig_max=7,
                                           rate_iso=0.2,
                                           tensor=False)
    soft_mask = cv2.filter2D(mask, -1, kernel)
    # soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    K = img + weight * residual
    K = np.clip(K, 0.0, 1.0)

    img_sharp = soft_mask * K + (1 - soft_mask) * img
    img_sharp *= 255.0
    img_sharp = np.clip(img_sharp, 0.0, 255.0)
    img_sharp = img_sharp.astype(np.uint8)

    return img_sharp, k_size


def test_sharpening():
    """
    :return:
    """
    img_name = "test_plate.jpg"
    img_dir = "E:/PyProjs/TestExperiments/data"
    save_dir = "E:/PyProjs/TestExperiments/result"
    img_path = img_dir + "/" + img_name
    if not os.path.isfile(img_path):
        print("[Err]: invalid image path: {:s}".format(img_path))

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h, w, c = img.shape
    if max(w, h) > 1000:
        img = cv2.resize(img, (int(w // 2), int(h // 2)), cv2.INTER_AREA)

    ## ----- Run the testing
    img_sharp, k_size = random_sharpening(img, low=10, high=50)
    ## -----

    print("kernel size: {:d}".format(k_size))
    cv2.imshow("Sharpening", img_sharp)
    cv2.waitKey()

    save_img_path = save_dir + "/sharpen_k_size{:d}.jpg" \
        .format(k_size)
    cv2.imwrite(save_img_path, img_sharp)


if __name__ == "__main__":
    test_sharpening()
    print("Done.")
