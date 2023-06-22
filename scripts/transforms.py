import cv2
import numpy as np
from PIL import Image

class FOVExtraction(object):
    
    def __call__(self, img, tol=7):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) # PIL to cv
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol
        img_blue = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img_green = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img_red = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img_blue, img_green, img_red], axis=-1)
            
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # cv to PIL
        
        return img


class CircleCrop(object):

    def __call__(self, img):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) # PIL to cv

        height, width, _ = img.shape

        x = int(width/2)
        y = int(height/2)
        r = np.amin((x, y))

        circle_img = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
        img = cv2.bitwise_and(img, img, mask=circle_img)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # cv to PIL
        
        return img