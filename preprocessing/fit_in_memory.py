#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from spectral import open_image
import numpy as np


class _fit_in_memory:
    
    def __init__(self,img_path, available_memory_gb = 2 ):
        
        self._img = open_image(img_path)
        self._available_memory_bytes = available_memory_gb * 10**9
        self._nrows = self._img.nrows
        self._ncols = self._img.ncols
        self._nbands = self._img.nbands
        self._sample_size  = self._img.sample_size
        
        
    def patitions(self):
        
        size_of_row_bytes = self._ncols * self._nbands * self._sample_size
        
        max_number_of_rows = self._available_memory_bytes // size_of_row_bytes
        
        last_block_row_count = self._nrows%max_number_of_rows
        
        total_blocks = int(np.ceil(self._nrows/max_number_of_rows))

        if total_blocks == 1:
            list_of_partition = list([[0,self._nrows]])
            return list_of_partition
        
        list_of_partition = list()
        
        for block in range(total_blocks):
            
            start_row = int(block*max_number_of_rows)
    
            block_rows_count = max_number_of_rows
        
            if block == total_blocks-1:
                block_rows_count = last_block_row_count

            end_row = int(start_row + block_rows_count)
            
            list_of_partition.append([start_row,end_row])
            
        return list_of_partition