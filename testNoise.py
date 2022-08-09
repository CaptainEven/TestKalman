# encoding=utf-8

import os
import cv2
import numpy as np
from scipy.linalg import orth


def random_gauss_noise(img,
                       low=2,
                       high=25):
    """
    :param img:
    :param low:
    :param high:
    :return:
    """
    img = img.astype(np.float32)
    img = cv2.normalize(img, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    img = np.clip(img, 0.0, 1.0)

    noise_level = np.random.randint(low, high)
    rand_num = np.random.rand()
    if rand_num > 0.6:  # add color Gaussian noise
        img += np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
    elif rand_num < 0.4:  # add grayscale Gaussian noise
        img += np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:  # add  noise
        L = high / 255.0
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    img *= 255
    img = np.clip(img, 0.0, 255)
    img = img.astype(np.uint8)

    return img

def random_poisson_noise(img):
    """
    :param img:
    :return:
    """
    img = img.astype(np.float32)
    img = cv2.normalize(img, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    img = np.clip(img, 0.0, 1.0)

    vals = 10 ** (2 * np.random.random() + 2.0)  # [2, 4]
    if np.random.random() < 0.5:
        img = np.random.poisson(img * vals).astype(np.float32) / vals
    else:
        img_gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        img_gray = np.clip((img_gray * 255.0).round(), 0, 255) / 255.
        noise_gray = np.random.poisson(img_gray * vals).astype(np.float32) / vals - img_gray
        img += noise_gray[:, :, np.newaxis]
    img = np.clip(img, 0.0, 1.0)

    img *= 255
    img = np.clip(img, 0.0, 255)
    img = img.astype(np.uint8)

    return img

def random_speckle_noise(img, low=5, high=25):
    """
    :param img:
    :param low:
    :param high:
    :return:
    """
    img = img.astype(np.float32)
    img = cv2.normalize(img, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    img = np.clip(img, 0.0, 1.0)

    noise_level = np.random.randint(low, high)
    rand_num = np.random.random()
    if rand_num > 0.6:
        img += img \
               * np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
    elif rand_num < 0.4:
        img += img \
               * np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:
        L = high / 255.0
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += img * np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]).astype(np.float32)

    img *= 255
    img = np.clip(img, 0.0, 255)
    img = img.astype(np.uint8)

    return img

def test_noise():
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
    noise_type = np.random.randint(1, 4)
    if noise_type == 1:
        img_noise = random_gauss_noise(img, low=2, high=10)
    elif noise_type == 2:
        img_noise = random_poisson_noise(img)
    elif noise_type == 3:
        img_noise = random_speckle_noise(img, low=5, high=10)
    ## -----

    cv2.imshow("Noise", img_noise)
    cv2.waitKey()

    if noise_type == 1:
        save_img_path = save_dir + "/noise_gauss.jpg"
        print("Noise type: gauss")

    elif noise_type == 2:
        save_img_path = save_dir + "/noise_poisson.jpg"
        print("Noise type: poisson")

    elif noise_type == 3:
        save_img_path = save_dir + "/noise_speckle.jpg"
        print("Noise type: speckle")

    cv2.imwrite(save_img_path, img_noise)


if __name__=="__main__":
    test_noise()
    print("Done.")