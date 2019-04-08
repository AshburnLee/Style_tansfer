import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


class imgPreProcess:
    def __init__(self, name):
        self.name = name
        
    def getWidthHeigh(self):
        """Get the width and height of an image"""
        img_string = tf.read_file(self.name)
        img_decoded = tf.image.decode_image(img_string)
        sess = tf.Session()
        img_decoded_val = sess.run(img_decoded)
        width = img_decoded_val.shape[0]
        height = img_decoded_val.shape[1]

        return width, height
    
    def reSize(self, new_w, new_h):
        """Resize an image to 224*224"""
        img_string = tf.read_file(self.name)
        img_decoded = tf.image.decode_image(img_string)
        
        width, height = self.getWidthHeigh()
        img_decoded = tf.reshape(img_decoded, [1, width, height, 3])

        resize_img = tf.image.resize_bicubic(img_decoded, [new_w, new_h])

        sess = tf.Session()

        img_decoded_val = sess.run(resize_img)
        img_decoded_val = img_decoded_val.reshape((new_w, new_h, 3))
        img_decoded_val = np.asarray(img_decoded_val, np.uint8)

        return img_decoded_val
