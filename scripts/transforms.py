import numpy as np
from PIL import Image

class GetROI(object):
    """Crops black pixels in a retinography (gets the FOV)"""

    def __call__(self, img):
        img = np.array(img)

        # Mask of coloured pixels.
        mask = img > 0

        # Coordinates of coloured pixels.
        coordinates = np.argwhere(mask)

        # Binding box of non-black pixels.
        x0, y0, s0 = coordinates.min(axis=0)
        x1, y1, s1 = coordinates.max(axis=0) + 1 # slices are exclusive at the top.

        # Get the contents of the bounding box.
        img = img[x0:x1, y0:y1]
        img = Image.fromarray(img)
        
        return img