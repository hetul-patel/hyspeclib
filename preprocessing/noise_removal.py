from ..hyperspectral_image import read_image
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import random as r
import numpy as np


class noise_removal:

    def __init__(self, read_image,ndvi_threshold = 125, NIR = 90 , RED = 55, min_threshold = 0, max_threshold = 1, data_ignore_value=-9999.0):

        self._image = read_image

        self._height = read_image.img_height

        self._width = read_image.img_width

        self._bands = read_image.img_bands

        self._min_threshold = min_threshold

        self._max_threshold = max_threshold

        self._data_ignore_value = data_ignore_value

        self._NIR = NIR

        self._RED = RED

        self._ndvi_threshold = ndvi_threshold

        self._bands_list = np.arange(self._bands)



    def _only_vegetation(self, arr_reflectance):

        output_array = list()

        for each_pixel in arr_reflectance:

            if (each_pixel[0] != self._data_ignore_value) and (self._calculate_ndvi(each_pixel[self._NIR], each_pixel[self._RED]) >= self._ndvi_threshold) :

                output_array.append(np.array(each_pixel))

        return np.array(output_array)


    def _random(self):

        r.seed(0)

        return list(map(lambda _: [r.randint(0,self._height-1),r.randint(0, self._width-1)], range(1000)))

    def _reflectance_stat(self):

        indices = self._random()

        arr_reflectance = [ self._image.sub_image()[index] for index in indices ]

        arr_reflectance_veg = self._only_vegetation(arr_reflectance)

        max_reflectance = np.max(arr_reflectance_veg,axis=0)

        min_reflectance = np.min(arr_reflectance_veg,axis=0)

        return min_reflectance, max_reflectance

    def _calculate_ndvi(self, NIR, RED):

        if NIR - RED == 0 :
            ndvi = 0
        else:
            ndvi = ( NIR - RED ) / ( NIR + RED )
            ndvi = 100 + ndvi * 100

        return ndvi


    def mean_stat(self):

        indices = self._random()

        arr_reflectance = [ self._image.sub_image()[index] for index in indices ]

        arr_reflectance_veg = self._only_vegetation(arr_reflectance)

        noisy_bands = self.show_noisy_bands()

        good_bands = list(set(self._bands_list) - set(noisy_bands))

        mean_val = np.mean(arr_reflectance_veg[:,good_bands])

        max_value = np.max(arr_reflectance_veg[:,good_bands])

        min_value = np.min(arr_reflectance_veg[:,good_bands])

        stdv = np.std(arr_reflectance_veg[:][good_bands])

        upper_bound = mean_val + 2*stdv

        lower_bound = mean_val - 2*stdv

        return {'mean':mean_val, "mean + 2*sigma":upper_bound, "mean - 2*sigma":lower_bound,"stdv":stdv,"max":max_value,"min":min_value}



    def reflectance_plot(self):

        min_reflectance, max_reflectance = self._reflectance_stat()

        for index,val in enumerate(max_reflectance):
            if val >= self._max_threshold:
                max_reflectance[index] = self._max_threshold + 0.1


        for index,val in enumerate(min_reflectance):
            if val <= self._min_threshold:
                min_reflectance[index] = self._min_threshold - 0.1


        plt.figure(figsize=(16,9),dpi=150)
        plt.scatter(self._bands_list,min_reflectance,color='r',label='Minimum value',marker='.')
        plt.scatter(self._bands_list,max_reflectance,color='b',label='Maximum value',marker='.')
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
