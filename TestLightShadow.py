# encoding=utf-8

import os
import cv2
import numpy as np


def random_light_or_shadow(img, base=200, low=10, high=255):
    """
    @param img:
    @param base:
    """
    h, w, c = img.shape  # BGR or RGB

    ## ----- Randomly Generate Gauss Center
    center_x = np.random.randint(-h * 1.5, h * 1.5)
    center_y = np.random.randint(-h * 1.5, h * 1.5)

    radius_x = np.random.randint(int(h * 0.5), h * 1.5)
    radius_y = np.random.randint(int(h * 0.5), h * 1.5)

    delta_x = np.power((radius_x / 2), 2)
    delta_y = np.power((radius_y / 2), 2)

    x_arr, y_arr, c_arr = np.meshgrid(np.arange(w), np.arange(h), np.arange(c))
    weight = np.array(
        base * np.exp(-np.power((center_x - x_arr), 2) / (2 * delta_x))
        * np.exp(-np.power((center_y - y_arr), 2) / (2 * delta_y))
    )

    light_mode = np.random.randint(0, 2)
    if light_mode == 1:  # shadow
        img = img - weight
        img[img < 0] = low  # clipping
    else:  # light
        img = img + weight
        img[img > 255] = high  # clipping

    return img.astype(np.uint8)


if __name__ == "__main__":
    img_name = "test_plate.jpg"
    img_dir = "E:/PyProjs/TestExperiments"
    img_path = img_dir + "/" + img_name
    if not os.path.isfile(img_path):
        print("[Err]: invalid image path: {:s}".format(img_path))

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h, w, c = img.shape
    if max(w, h) > 1000:
        img = cv2.resize(img, (int(w // 2), int(h // 2)), cv2.INTER_AREA)

    ## ----- Run the testing
    img = random_light_or_shadow(img)
    ## -----

    cv2.imshow("Mosaic", img)
    cv2.waitKey()
    print("Done.")