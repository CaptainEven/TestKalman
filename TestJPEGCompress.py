# encoding=utf-8

import os
import cv2
import numpy as np


def random_jpeg_compress(img, low=50, high=95):
    """
    :param img:
    :return:
    """
    quality_factor = np.random.randint(low, high)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    result, enc_img = cv2.imencode(".jpg",
                                   img,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(enc_img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, quality_factor


def test_jpeg_compress():
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
    img_compress, quality_factor = random_jpeg_compress(img, low=10, high=95)
    ## -----


    print("JPEG quality factor: {:d}".format(quality_factor))
    cv2.imshow("JPEGCompress", img_compress)
    cv2.waitKey()

    save_img_path = save_dir + "/quality{:d}.jpg".format(quality_factor)
    cv2.imwrite(save_img_path, img_compress)

    print("Done.")

if __name__=="__main__":
    test_jpeg_compress()
    print("Done.")