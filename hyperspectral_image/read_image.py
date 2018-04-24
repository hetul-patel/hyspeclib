from .image import image
from .check_memory import check_memory
import matplotlib.pylab as pylab 

class read_image(image):

    def __init__ (self, img_path):
        super().__init__(img_path)
        

    def sub_image(self):
        """
        Function loads hyperspectral image as array into memory with specified rows , columns and bands.

        Examples:

        1. img.sub_image()[ 0 : 10, 0 : 20, 0 : 10] loads first 10 rows, 10 columns and 10 bands

        2. img.sub_image()[ :,:, [55,80] ] loads only 55th and 80th bands for entire image


        """
        return self._read_only_image

    

