from spectral import imshow
from .read_image import read_image
import numpy as np
import pandas as pd
import matplotlib as mt
import matplotlib.pyplot as plt

class view_classified_image:


    def __init__(self, classified_image_path, class_labels=None ,color_list=None):


        self._classified_img = np.array(pd.read_csv(classified_image_path, header=None))

        self._max_class = np.max(self._classified_img)
        
        if color_list!=None:
            self._color_list = color_list
        else:
            tmp_list = ['#FFFFFF','#3BCBD5','#F7CD0A','#990033','#FF3399','#339900','#666600',
                         '#000000','#0000FF','#CC7755','#FF8866','#FF9988','#EF5350','#F48FB1',
                         '#880E4F','#E1BEE7','#9FA8DA','#1E88E5','#26A69A', '#69F0AE','#FDD835',
                         '#6D4C41','#546E7A','#B71C1C']
                         
            self._color_list = tmp_list[:self._max_class]
        
        
        self._color_rgb = self._color_rgb_list(self._color_list)
        

        if class_labels == None:
            class_labels = str(np.arange(self._max_class))

        plt.figure(figsize=(self._max_class,1))
            
        try:
            plt.bar(x=np.arange(self._max_class), height=1,width=1, color = self._color_list, tick_label = class_labels, edgecolor='#000000')
        except:
            plt.bar(left=np.arange(self._max_class), height=[1 for i in range(self._max_class)],width=1, color = self._color_list, tick_label = class_labels, edgecolor='#000000')

        plt.title('Color pallete for classes')
        plt.show()

    def _color_rgb_list(self,color_list):
        color_rgb = list()

        for color_hex in color_list:
            hex1,hex2,hex3 = color_hex[1:3], color_hex[3:5], color_hex[5:7]

            r,g,b = np.int(hex1, base=16), np.int(hex2, base=16), np.int(hex3, base=16)

            color_rgb.append([r,g,b])

        return color_rgb


    def show_and_save(self, save_color_img_path):
        
        cmap , norm = mt.colors.from_levels_and_colors(levels=np.arange(1,self._max_class+1,1),colors=self._color_list[1:],extend='neither')

        plt.figure(figsize=(self._classified_img.shape[1]//50,self._classified_img.shape[0]//50))
        plt.imshow(self._classified_img,cmap=cmap, norm=norm)
        plt.savefig(save_color_img_path,format='png')
        plt.show()

    def _helper(self, img_arr):

        unlabeled = self._max_class

        for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                if img_arr[i,j] == unlabeled:
                    img_arr[i,j] = 0

        return img_arr

    def overlay_on_raw_img(self, path_to_raw_img):

        raw_img = read_image(path_to_raw_img)

        v = imshow(raw_img.sub_image()[:,:,:], classes = self._helper(np.copy(self._classified_img)), colors = self._color_rgb)
