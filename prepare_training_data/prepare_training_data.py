from spectral import open_image
import pandas as pd
import spectral.io.aviris as aviris
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import matplotlib.colors
import matplotlib as mt
import numpy.linalg as lin

class prepare_training_data:

    def __init__(self,img_source_dir="",save_path="",noisy_bands=None):

        self._img_source = img_source_dir
        self._save_path = save_path
        self._noisy_bands = noisy_bands


    def extract(self):

        if self._img_source =="":
            print('Please give path to extracted crops directory.\n')
        if self._save_path == "":
            print('Please give path to save tranin')

        crop_list = sorted(glob.glob(self._img_source+'*.hdr'))
        crop_names = list()

        print('\n------------------Preparing training dataset---------------------\n\n')

        print('Total {} crops are found in directory\n\n'.format(len(crop_list)))

        fw_w = open(self._save_path,'w')

        for index,path in enumerate(crop_list):

            crop_name = path.split('/')[-1].split('.')[0]
            
            crop_names.append(crop_name)

            if self._noisy_bands is not None:

                img = open_image(path)
                all_bands = set(np.arange(img.nbands))
                img = img[:,:,list(all_bands-set(self._noisy_bands))]

            else:
                
                img = open_image(path)

            height = img.shape[0]
            width = img.shape[1]

            count = 0

            for i in range(height):
                for j in range(width):
                    temp = img[i,j]
                    if(np.mean(temp)>0):
                        count += 1
                        for band in range(temp.size):
                            fw_w.write(str(temp[band])+",")
                        fw_w.write(str(index+1)+'\n')
            print("Crop No. : {} \t| Name : {}  \t\t| Tota samples : {}".format(index+1, crop_name, count))

        print('\n\nProcess completed. File saved at : ',self._save_path)

        print('\n\nCrop List : ',crop_names)
        print('\n\n---------------------------------------------------------\n\n')

        fw_w.close()

    def outlier_analysis(self,data_path,class_labels):

        self._class_labels = class_labels

        train_data = pd.read_csv(data_path,header=None)
        self._train_data_array = np.asarray(train_data)

        low_dimentional = self._train_data_array[:,[57,85]]

        #Prepare a dictionary using train data for classwise array
        self._total_bands = train_data.shape[1]-1
        self._class_wise_data_array = dict()
        self._class_wise_data_array_full_dim = dict()

        for index,pixel in enumerate(low_dimentional):
            class_number = int(self._train_data_array[index,self._total_bands])

            if self._class_wise_data_array.get(class_number) != None:
                self._class_wise_data_array[class_number].append(pixel)
                self._class_wise_data_array_full_dim[class_number].append(self._train_data_array[index,:self._total_bands])
            else:
                self._class_wise_data_array.update({class_number:[]})
                self._class_wise_data_array[class_number].append(pixel)

                self._class_wise_data_array_full_dim.update({class_number:[]})
                self._class_wise_data_array_full_dim[class_number].append(self._train_data_array[index,:self._total_bands])

        self._size = len(self._class_wise_data_array)

        mean_vector = []
        cov_vector = []

        for i in range(self._size):
            sum_vector = np.zeros(shape=(1,2))
            total_class_sample = len(self._class_wise_data_array[i+1])
            for j in range(total_class_sample):
                sum_vector += self._class_wise_data_array[i+1][j]

            mean_vector.append(np.divide(sum_vector,len(self._class_wise_data_array[i+1])) )


        self._mean_vector = np.array(mean_vector).reshape(len(self._class_wise_data_array),2)

        print('\n\n------------------- Instructions -------------------\n\n')

        print('1. Use the visualise() method to plot two dimention representation of crops.\n')

        print('2. Use the remove_outliers() method to remove outliers. You can pass external limits with\n')

        print(""" rules = { 1:[0,0,0,0],2:[0,0,0,0]} as argument.\n\n You can pass Left, Right, Top, Bottom limits for manually remove outliers for any crop.\n """)

        print("\n3. Use save() method to save data after outlier removal.")


    def visualise(self):


        self._color = ['#FFFFFF','#3BCBD5','#F7CD0A','#990033','#FF3399','#339900','#666600',
                     '#000000','#0000FF','#CC7755','#FF8866','#FF9988','#EF5350','#F48FB1',
                     '#880E4F','#E1BEE7','#9FA8DA','#1E88E5','#26A69A', '#69F0AE','#FDD835',
                     '#6D4C41','#546E7A','#B71C1C']

        self._level = np.arange(len(self._color)+1)


        cmap , norm = matplotlib.colors.from_levels_and_colors(levels=self._level,colors=self._color)

        fig = plt.figure(figsize=(15,10),dpi=250)
        for index,crop in enumerate(self._mean_vector):
            label = str(index + 1)+' - '+ self._class_labels[index] +' : '+ str(len(self._class_wise_data_array[index+1]))
            plt.scatter(x=crop[0],y=crop[1],label=label,color=self._color[index+1],marker='o')

        for i in range(self._size):
            plt.scatter(x=np.array(self._class_wise_data_array[i+1])[:,0],y=np.array(self._class_wise_data_array[i+1])[:,1],color=self._color[i+1],marker='.',alpha=1)


        plt.legend(loc=1)
        plt.title('Visualization of crops in 2D')
        plt.xlabel('RED BAND - 57')
        plt.ylabel('NIR BAND - 85')
        plt.show()

    def _outlier_removal(self,class_num,arr,ext_left=0,ext_right=0,ext_top=0,ext_bottom=0):

        full_dim = self._class_wise_data_array_full_dim[class_num]

        Q1 = np.percentile(arr,25,axis=0)
        Q3 = np.percentile(arr,75,axis=0)

        IQR = Q3-Q1

        Upper_threshold = Q3 + IQR*1.5
        Lower_threshold = Q1 - IQR*1.5

        if ext_left >0:
            Lower_threshold[0] = ext_left
        if ext_bottom >0:
            Lower_threshold[1] = ext_bottom
        if ext_right >0:
                Upper_threshold[0] = ext_right
        if ext_top >0:
                Upper_threshold[1] = ext_top


        clean_arr = []
        full_dim_arr = []

        for index,pixel in enumerate(arr):
            if (pixel[0] <= Upper_threshold[0]) and (pixel[0] >= Lower_threshold[0]) and (pixel[1] <= Upper_threshold[1]) and (pixel[1] >= Lower_threshold[1]):
                clean_arr.append(pixel)
                full_dim_arr.append(full_dim[index])

        return clean_arr, full_dim_arr

    def remove_outlier(self,rules=[]):

        external_rules = dict()

        for index in self._class_wise_data_array.keys():
            external_rules.update({index:[0,0,0,0]})

        for item in rules:
            external_rules.update({item:rules[item]})


        self._clean_classwise_data = {}
        self._clean_fulldim_data = {}

        for index in self._class_wise_data_array.keys():

            extern = external_rules[index]

            pc_arr,full_arr = self._outlier_removal(index,self._class_wise_data_array[index],extern[0],extern[1],extern[2],extern[3])

            self._clean_classwise_data.update({index:np.array(pc_arr)})

            self._clean_fulldim_data.update({index:np.array(full_arr)})

        fig = plt.figure(figsize=(15,10),dpi=250)

        for index,crop in enumerate(self._mean_vector):
            label = str(index + 1)+'-'+ self._class_labels[index] +' : '+ str(len(self._clean_classwise_data[index+1]))
            plt.scatter(x=crop[0],y=crop[1],label=label,color=self._color[index+1],marker='o')

        for i in range(len(self._clean_classwise_data)):
            plt.scatter(x=np.array(self._clean_classwise_data[i+1])[:,0],y=np.array(self._clean_classwise_data[i+1])[:,1],color=self._color[i+1],marker='.',alpha=1)

        plt.legend(loc=1)
        plt.title('Visualization of crops in 2D')
        plt.xlabel('RED BAND - 57')
        plt.ylabel('NIR BAND - 85')
        plt.show()

    def save(self, path_to_save):

        fw_w = open(path_to_save,'w')

        for crop in self._clean_fulldim_data.keys():
            for pixel in self._clean_fulldim_data[crop]:
                for i in range(self._total_bands):
                    fw_w.write(str(pixel[i])+",")
                fw_w.write(str(crop)+"\n")

        fw_w.close()
