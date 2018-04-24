"""
Image class is hyperspectral image. It has height, width and channels.
"""
from spectral import open_image
import numpy as np
from .check_memory import check_memory


class image:


    def __init__(self, img_path):
        """
        Function locates hyperspectral image in disk. It does not load
        entire image into memory.

        img_path : path to IMAGE_NAME.hdr' file of aviris image 

        """
        self._read_only_image = open_image(img_path)

        self._img_path = img_path

        self._img_height = self._read_only_image.nrows

        self._img_width = self._read_only_image.ncols

        self._img_bands = self._read_only_image.nbands

    
    def load_image(self):
        """
        Function load hyperspectral data into memory. Image is loaded if enough memory is 
        available otherwise not.

        """

        mem = check_memory()

        if  mem.check_image_before_load(self.size):

            try :
                del self._loaded_img
            except:
                pass

            self._loaded_img = open_image(self.img_path).load()

            return self._loaded_img

        else:

            return 'Out of memory error. Free some space in memory or reduce image dimentions.'


    @property       
    def img_height(self):
        return self._img_height

    @img_height.setter
    def img_height(self, value):
        self._img_height = value

    @property       
    def img_width(self):
        return self._img_width

    @img_width.setter
    def img_width(self, value):
        self._img_width = value

    @property       
    def img_bands(self):
        return self._img_bands

    @img_bands.setter
    def img_bands(self, value):
        self._img_bands = value


    @property
    def img_path(self):
        return self._img_path

    @property
    def size(self):
        """
        Returns number of rows, columns and bands
        """
        return (self._img_height, self._img_width, self._img_bands)
