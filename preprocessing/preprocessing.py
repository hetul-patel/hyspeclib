#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .fit_in_memory import _fit_in_memory
from .noise_removal import noise_removal
import spectral.io.envi as envi
from ..hyperspectral_image import read_image
import numpy as np
import os



class preprocessing:
    
    def __init__(self, img_path, save_directory, available_memory_gb):
        
        self._max = available_memory_gb
        self._img_path = img_path
        self._save_path = save_directory
        self._image_name = img_path.split('/')[-1].split('.')[0]
        
        partition = _fit_in_memory(self._img_path, available_memory_gb=self._max)
        
        self._list_of_partitions = partition.patitions()
        self._total_partitions  = len(self._list_of_partitions)
        
        del partition

        print('Image will be saved in {} partitions.'.format(self._total_partitions))                
        
    def _calculate_ndvi(self, NIR, RED):
        
        if NIR - RED == 0 : 
            ndvi = 0 
        else:
            ndvi = ( NIR - RED ) / ( NIR + RED )
            ndvi = 100 + ndvi * 100
    
        return ndvi

    def _get_retained_bands(self, noisy_bands, total_bands):

        return  list(set(np.arange(total_bands)) - set(noisy_bands))
    
    def perform(self, ndvi_threshold = 125, bad_reflectance_value = -9999. , NIR = 90 , RED = 55, min_threshold = 0, max_threshold = 1, noisy_bands=None  ):
        
        img  = read_image(self._img_path)

        if noisy_bands == None:
            
            noise_rem = noise_removal(img,min_threshold=min_threshold, max_threshold=max_threshold)
            
            noisy_bands = noise_rem.show_noisy_bands()

        retained_bands = self._get_retained_bands(noisy_bands,img.img_bands)
                                
        masking_pixel = [0.0 for i in range(len(retained_bands))]

        print('--------------- Performing Preprocessing ---------------\n')
        
        for index,each_partion in enumerate(self._list_of_partitions):
            
            print('\nPartition : {} / {} running...\n'.format(index+1, self._total_partitions))
            
            sub_image = img.sub_image()[each_partion[0]:each_partion[1],:,retained_bands]
                        
            for index_row, each_row in enumerate(sub_image):
                
                for index_pixel, each_pixel in enumerate(each_row):
                    
                        
                    if (each_pixel[0] == bad_reflectance_value) or (self._calculate_ndvi(each_pixel[NIR],each_pixel[RED]) < ndvi_threshold) :
                        sub_image[index_row, index_pixel] = masking_pixel
            
                    
            if not os.path.exists(self._save_path):
                os.makedirs(self._save_path)
                        
            envi.save_image(self._save_path + self._image_name + '_part_' + str(index+1)+'.hdr', sub_image,force=True,interleave='bil',ext=None)
            
            del sub_image
            
        
        print('\nPreprocessing completed. Output directory : '+self._save_path)
        print('\n\n---------------------------------------------------------')

 
