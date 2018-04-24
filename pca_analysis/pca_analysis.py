#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from spectral import principal_components
import matplotlib.pyplot as plt
from spectral import open_image
import glob
import numpy as np
import time
import pandas as pd


class pca_analysis:
    
    def __init__(self, image_directory_path=None ):
        
        if image_directory_path!=None:
            self._image_directory_path = image_directory_path
            self._processed_images = glob.glob(image_directory_path+'**/**.hdr',recursive=True)
            self._total_images = len(self._processed_images)
            print('Total {} images found in directory.'.format(self._total_images))
    
    def _fetch_pca(self, image_path):
        
        t1 = time.time()
        #print("Fetching file : "+ file_name)
        img = open_image(image_path)
        
        nbands = img.nbands

        pc = principal_components(img)

        t2 = time.time() - t1
        print('\nTook {:.4f} seconds / {:.4f} min to run.\n'.format(t2,t2/60))    
        return nbands, pc  
    
    
    def _write_to_file(self, image_name, nbands, principal_component_analysis):
        
        #t1 = time.time()

        eigen_vectors = principal_component_analysis.eigenvectors
        eigen_values = principal_component_analysis.eigenvalues


        with open(self._image_directory_path+image_name+'_pca.csv', mode='w') as file:
            
            for pc_number in range(nbands):
                
                file.write( str(pc_number+1))
                file.write(','+str(eigen_values[pc_number]))
                for component in eigen_vectors[pc_number]:
                    file.write(',' + str(np.round(np.real(component),decimals=4)))
                file.write('\n')

       # t2 = time.time() - t1
        #print('Writing to disk complete... took {} seconds'.format(t2))
        
    def perform(self):
        
        try :

            for index,image in enumerate(self._processed_images):
    
                print('Current image {}/{}'.format( index+1, self._total_images ))
                
                image_name = image.split('/')[-1].split('.')[0]
    
                nbands, temp_pca_analysis = self._fetch_pca(image)
    
                self._write_to_file(image_name, nbands, temp_pca_analysis)
    
                del temp_pca_analysis
            
            print('Eigen Values and Eigen Vectors are saved successfully at : '+self._image_directory_path)
        except:
            print('Please pass preprocessed images directory in pca_analysis(?) as parameter\nor try again.' )
          
    
    # Exploration
    
    def _fetch_array(self, image_stat):
        vectors = image_stat.drop(labels=[0,1], axis=1)
        return np.asarray(vectors), np.asarray(image_stat[1])        
            
    def _cumulative_sum_of_percentage(self, array_of_eigen_values, filepath):
    
        cumulative_percentage = 0
        more_than_zero = True
        count = 0
        sum_of_eigen_value = np.sum(array_of_eigen_values)
        filename = filepath.split('/')[-1].split('.')[0]
        
        out = list()
        
        print('\nEigen Values for partition : {} \n'.format(filename))

        while more_than_zero:
            eigen_value = array_of_eigen_values[count]
            percent = eigen_value*100/sum_of_eigen_value
            cumulative_percentage += percent
            count += 1            
            out.append([count,np.round(eigen_value, decimals=3), np.round(percent, decimals=3), np.round(cumulative_percentage,decimals=3)])
            
            if np.round(eigen_value,decimals=3) == 0.0:
                more_than_zero = False
                
        out_df = pd.DataFrame(out,columns=['PC' , 'E.V' , 'Percentage' , 'Cumulative Percent'])
        
        print(out_df)
        
    def show_eigen_values(self, pca_result_directory):
        
        files = glob.glob(pca_result_directory+'**/**.csv',recursive=True)
    
        for file in files:
            
            stat = pd.read_csv(file, header=None)
            
            _, eigen_values = self._fetch_array(stat)
            
            self._cumulative_sum_of_percentage(eigen_values,file)
            
    def _eigen_component_contribution(self,vectors, eigen_values, top, max_pc = 2):
    
        sorted_bands = list()
    
        mydtype = [('band',int),('total',float)]
    
        vectors = vectors[:max_pc]
        
        contribution = zip(np.arange(len(vectors[0])),np.sum(vectors**2,axis=0))
        
        for item in contribution:
            sorted_bands.append(item)
        
        sorted_bands = np.asarray(sorted_bands, dtype=mydtype)
        sorted_bands = np.sort(sorted_bands, order='total')
    
        x = sorted_bands[-1*top:]
        #print(x)
        
        bands = []
        for band in x:
            #print(band[0])
            bands.append(band[0])
            
        return np.sort(bands)
    
    
    def _band_frequency_map(self,bands):
        sorted_bands = np.sort(bands, axis=None)
        
        bands_occurences = dict()
    
        for band in sorted_bands:
            if bands_occurences.get(band) == None:
                bands_occurences.update({band:1})
            else:
                bands_occurences.update({band:bands_occurences.get(band)+1})
        return bands_occurences
    
    def plot_band_frequncy(self,pca_result_directory, top, max_pc = 2, min_frequency=None):
        
        files = glob.glob(pca_result_directory+'**/**.csv',recursive=True)
        
        list_of_bands = list()
    
        for file in files:
            
            stat = pd.read_csv(file, header=None)
            
            vectors, eigen_values = self._fetch_array(stat)
            
            list_of_bands.append(self._eigen_component_contribution(vectors,eigen_values,top,max_pc))
            
        band_occurances = self._band_frequency_map(list_of_bands)
        
        if min_frequency!=None:
            filtered_dictionary = {}
            for band in band_occurances:
                if band_occurances[band] >= min_frequency:
                    filtered_dictionary.update({band:band_occurances[band]})
            band_occurances = filtered_dictionary
                
        
        fig = plt.figure()
        fig.set_figwidth(30)
        fig.set_figheight(5)
        plt.bar(band_occurances.keys(),band_occurances.values(),align='center',width=0.6)
        plt.xticks( np.arange(max(band_occurances.keys())+1),rotation=90)
        plt.grid(axis='y')
        plt.xlabel('Band Number')
        plt.ylabel('Number of occurences')
        plt.title('Total number of occurances in top {} bands for all {} images'.format(top,len(files)))
        plt.show()

    
    # Actutal Reduction
        
    
    def band_reduction(self,pca_result_directory,top,min_frequency, max_pc = 2):
        
        files = glob.glob(pca_result_directory+'**/**.csv',recursive=True)
        
        list_of_bands = list()
    
        for file in files:
            
            stat = pd.read_csv(file, header=None)
            
            vectors, eigen_values = self._fetch_array(stat)
            
            list_of_bands.append(self._eigen_component_contribution(vectors,eigen_values,top,max_pc))
            
        band_occurances = self._band_frequency_map(list_of_bands)
        
        frequent_bands = []
    
        for band in band_occurances:
            if band_occurances.get(band) >= min_frequency:
                #print(band,':',bands_occurences.get(band))
                frequent_bands.append(band)
        
        return frequent_bands
    
    

        
        

