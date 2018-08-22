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
    
    def __init__(self, image_directory_path=None, save_dir = None ):
        """

        :param image_directory_path: Path to where preprocessed images directory
        :param save_dir: Path where PCA statistics should be saved
        """
        
        if image_directory_path!=None:
            self._image_directory_path = image_directory_path
            self._processed_images = glob.glob(image_directory_path+'**/**.hdr',recursive=True)
            self._total_images = len(self._processed_images)
            self._save_dir = save_dir
            print('Total {} images found in directory.'.format(self._total_images))
    
    def _fetch_pca(self, image_path):
        """Function calculates the eigen vectors and eigen values for given image

        :param image_path: Image to perfrom PCA
        :return: number of bands and PCA statistics
        """
        
        t1 = time.time()
        print("Fetching file : "+ image_path)
        img = open_image(image_path)
        
        nbands = img.nbands

        pc = principal_components(img)

        t2 = time.time() - t1
        print('\nTook {:.4f} seconds / {:.4f} min to run.\n'.format(t2,t2/60))    
        return nbands, pc  
    
    
    def _write_to_file(self, image_name, nbands, principal_component_analysis):
        """Write Eigen vectors and eigen values from each image in file

        :param image_name: Image name
        :param nbands: Number of principal components to be saved
        :param principal_component_analysis: PCA statistics object returned from _fetch_pca method
        :return: None
        """
        t1 = time.time()

        eigen_vectors = principal_component_analysis.eigenvectors
        eigen_values = principal_component_analysis.eigenvalues


        with open(self._save_dir+image_name+'_pca.csv', mode='w') as file:
            
            for pc_number in range(nbands):
                
                file.write( str(pc_number+1))
                file.write(','+str(eigen_values[pc_number]))
                for component in eigen_vectors[pc_number]:
                    file.write(',' + str(np.round(np.real(component),decimals=4)))
                file.write('\n')

        t2 = time.time() - t1
        print('Writing to disk complete... took {} seconds'.format(t2))
        
    def perform(self):
        """
        Performs PCA on list of processed images one by one
        stores the Eigen Vectors and Eigen values in csv file format
        """
        
        try :
            
            print('\n\n---------------- Computing PCA Statistics -------------------\n')

            for index,image in enumerate(self._processed_images):
    
                print('\nCurrent image {}/{}\n'.format( index+1, self._total_images ))
                
                image_name = image.split('/')[-1].split('.')[0]
    
                nbands, temp_pca_analysis = self._fetch_pca(image)
    
                self._write_to_file(image_name, nbands, temp_pca_analysis)
    
                del temp_pca_analysis
            
            print('Eigen Values and Eigen Vectors are saved successfully at : '+self._save_dir)
    
            print('\n\n-------------------------------------------------------------\n')
        except:
            print('Please pass preprocessed images directory in pca_analysis(?) as parameter\nor try again.' )
            print('\n\n-------------------------------------------------------------\n')
                  
            
          
    
    # Exploration
    
    def _fetch_array(self, image_stat):
        """
        Returns the array of eigen vectors from dataframe

        :param image_stat: dataframe
        :return: array of eigen vectors
        """
        vectors = image_stat.drop(labels=[0,1], axis=1)
        return np.asarray(vectors), np.asarray(image_stat[1])        
            
    def _cumulative_sum_of_percentage(self, array_of_eigen_values, filepath):
        """
        Prints the eigen value, percentage variance explained and
        cumulative sum of eigen value upto current pc

        :param array_of_eigen_values: array of eigen values
        :param filepath: Path to image for which current PCA statistics was calculated
        :return: None
        """
    
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
        """
        Prints the eigen value, percentage variance explained and
        cumulative sum of eigen value upto current pc

        :param pca_result_directory: Path to saved PCA statistics .csv file

        """
        
        files = glob.glob(pca_result_directory+'**/**.csv',recursive=True)
    
        for file in files:
            
            stat = pd.read_csv(file, header=None)
            
            _, eigen_values = self._fetch_array(stat)
            
            self._cumulative_sum_of_percentage(eigen_values,file)
            
    def _eigen_component_contribution(self,vectors, eigen_values, top, max_pc = 2):

        """
        Functions returns the top N bands contributed in top M principal components

        :param vectors: Eigen vectors
        :param eigen_values: Eigen Values
        :param top: Maximum number of bands to be considered ( N )
        :param max_pc: Top M pcs to be selected for finding contribution of each band
        :return: list of bands
        """
    
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
        """
        Returns number of times each bands occured in top bands for each image.

        :param bands: List of bands for which frequency of occurence is calculated
        :return: List of number of occurrences for each band.
        """
        sorted_bands = np.sort(bands, axis=None)
        
        bands_occurences = dict()
    
        for band in sorted_bands:
            if bands_occurences.get(band) == None:
                bands_occurences.update({band:1})
            else:
                bands_occurences.update({band:bands_occurences.get(band)+1})
        return bands_occurences
    
    def plot_band_frequncy(self,pca_result_directory, top, max_pc = 2, min_frequency=None):
        """
        Plot the diagram of frequency of occurrence in top N bands of every image.

        :param pca_result_directory: Directory where PCA statistics csv file is saved
        :param top: Maximum number of bands to be considered ( N )
        :param max_pc: Number of top PCs to be selected for finding contribution of each band
        :param min_frequency: Minimum number of occurrences for plotting.
        :return: None
        """
        
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
        plt.bar(list(band_occurances.keys()),list(band_occurances.values()),align='center',width=0.6)
        plt.xticks( np.arange(max(list(band_occurances.keys()))+1),rotation=90)
        plt.grid(axis='y')
        plt.xlabel('Band Number')
        plt.ylabel('Number of occurences')
        plt.title('Total number of occurances in top {} bands for all {} images'.format(top,len(files)))
        plt.show()

    
    # Actutal Reduction
        
    
    def band_reduction(self,pca_result_directory,top,min_frequency, max_pc = 2):
        """
        Module returns the optimal number of band based on PCA
        statistics calculated on different parts of images.

        :param pca_result_directory: Directory where PCA statistics csv file is saved
        :param top: Maximum number of bands to be considered ( N )
        :param max_pc: Number of top PCs to be selected for finding contribution of each band
        :param min_frequency: Minimum number of occurrences for plotting.
        :return:
        """

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
    
    

        
        

