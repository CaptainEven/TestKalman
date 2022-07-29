# encoding=utf-8

# encoding=utf-8

import os
import cv2
import numpy as np
from scipy.linalg import orth


def random_speckle(img, low=5, high=25):
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
        L = high / 255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += img * np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]).astype(np.float32)

    img *= 255
    img = np.clip(img, 0.0, 255)
    img = img.astype(np.uint8)

    return img, noise_level


def test_speckle():
    """
    :return:
    """
    img_name = "test_0.jpg"
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
    img_speckle, noise_level = random_speckle(img, low=2, high=25)
    ## -----

    print("Speckle noise level: {:d}".format(noise_level))
    cv2.imshow("Speckle", img_speckle)
    cv2.waitKey()

    save_img_path = save_dir + "/speckle_level_{:d}.jpg" \
        .format(noise_level)
    cv2.imwrite(save_img_path, img_speckle)


if __name__=="__main__":
    test_speckle()
    print("Done.")