from ..hyperspectral_image import read_image
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import random as r
import numpy as np


class noise_removal:

    def __init__(self, read_image, min_threshold = 0, max_threshold = 1):

        self._image = read_image

        self._height = read_image.img_height

        self._width = read_image.img_width

        self._bands = read_image.img_bands

        self._min_threshold = min_threshold

        self._max_threshold = max_threshold


    def _random(self):

        return list(map(lambda _: [r.randint(0,self._height-1),r.randint(0,self._width-1)], range(10000)))

    def _reflectance_stat(self):
        
        indices = self._random()

        arr_reflectance = [ self._image.sub_image()[index] for index in indices ]

        max_reflectance = np.max(arr_reflectance,axis=0)

        min_reflectance = np.min(arr_reflectance,axis=0)

        return min_reflectance, max_reflectance


    def reflectance_plot(self):

        min_reflectance, max_reflectance = self._reflectance_stat()

        plt.figure(figsize=(16,9),dpi=150)
        plt.plot(min_reflectance,color='r',label='Minimum value')
        plt.plot(max_reflectance,color='b',label='Maximum value')
        plt.hlines([self._min_threshold,self._max_threshold],0,self._bands,colors='grey',linestyles='dashed')
        plt.title('Minimum and Maximum reflectance - band wise')
        plt.xticks(np.arange(0,425,10), np.arange(0,425,10), rotation=90)
        plt.xlabel('Band number')
        plt.ylabel('Reflectance')
        plt.legend()
        plt.show()

    def show_noisy_bands_with_min_max(self):

        min_reflectance, max_reflectance = self._reflectance_stat()

        noisy_bands_min_max = [ (i,min_reflectance[i],max_reflectance[i]) for i in range(self._bands) if min_reflectance[i] < self._min_threshold or max_reflectance[i] > self._max_threshold ]

        return noisy_bands_min_max

    def show_noisy_bands(self):

        min_reflectance, max_reflectance = self._reflectance_stat()

        noisy_bands = [ i for i in range(self._bands) if min_reflectance[i] < self._min_threshold or max_reflectance[i] > self._max_threshold ]

        return noisy_bands


    def remove_bands(self, hdr_file, list_of_noisy_bands=None):
        """
        Saves an image to disk.

        Arguments:

            `hdr_file` (str):

                Header file (with ".hdr" extension) name with path.

            `list_of_noisy_bands` (list) optional:

                If passed, bands specified in `list_of_noisy_bands` will be removed
                , otherwise noisy bands will be detected and removed automatically.


        """

        if list_of_noisy_bands == None:
            discarded_bands = self.show_noisy_bands()
        else:
            discarded_bands = list_of_noisy_bands
        
        retained_bands = list(set(np.arange(self._bands)) - set(discarded_bands))
        
        envi.save_image(hdr_file, self._image.sub_image()[:,:,retained_bands], interleave='bil', force=True, ext=None)

