#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as spio
from pyts.image import GramianAngularField, MarkovTransitionField
from mpl_toolkits.axes_grid1 import ImageGrid
import random
import os


class CWUB:
    def __init__(self, path='.', window_length = 64):
        self.path = path
        self.w_length = window_length
        self._merge_datasets()
    
    def GAF_representation(self):
        self._GAF_Image_representation()

    def MTF_representation(self):
        self._MTF_Image_representation()

        

    def _merge_datasets(self):
        """
        **Note:** In the method, the whole seprate.mat files are mereged into one dictionary for 
        easy access and further analysis

        **Note::** Here we assume that all thre .mat files exist in the same directory that includes this file
        """
        data_root_path = self.path 
        data_paths = []
        for file in os.listdir(data_root_path):
            if '.mat' in file:
                data_paths.append(data_root_path+'/'+file)
        print(data_paths)
        whole_data = {}
        for file in data_paths:   
            mat = spio.loadmat(file, squeeze_me=True) # Load the dataset
            name = file.replace('.mat','').replace('./','')  
            data = {}
            for i in range(1,len(mat)-2): 
                data[i]=mat.get('C{}'.format(i))
            whole_data[name] = data
        self.whole_data = whole_data
        print(f'The datasets are sucessfully merged')

    def _GAF_Image_representation(self):
        gasf = GramianAngularField(image_size=1., method='summation')
        sliced_data = {}
        image_matrix ={}

        for d_name in self.whole_data.keys(): # this loop goes through each dataset
            dataset = self.whole_data[d_name]
            image_matrix[d_name] ={}
            for class_category in dataset.keys():# this loop goes through the eaech class of dataset 
                data = dataset[class_category]
                dummy = np.empty((len(data)//self.w_length, self.w_length))
                for i in range((len(data)//self.w_length)): # this loop goes through each class of a dataset
                    dummy[i,:] = data[i*self.w_length: (i+1)*self.w_length]
                image_matrix[d_name][class_category] = gasf.transform(dummy)
                del dummy
        del dataset, 
        self.GAF_image_matrix = image_matrix
        print(f'The GAF image representaion with window size {self.w_length} is finished')
        

        def _MTF_Image_representation(self):
            mtf = MarkovTransitionField(image_size=1., n_bins= self.w_length, ) #strategy ='uniform'
        sliced_data = {}
        _image_matrix ={}

        for d_name in self.whole_data.keys(): # this loop goes through each dataset
            dataset = self.whole_data[d_name]
            sliced_data[d_name] = {}
            _image_matrix[d_name] ={}
            for class_category in dataset.keys():# this loop goes through the eaech class of dataset 
                data = dataset[class_category]
                dummy = np.empty((len(data)//self.w_length, self.w_length))
                for i in range((len(data)//self.w_length)): # this loop goes through each class of a dataset
                    dummy[i,:] = data[i*self.w_length: (i+1)*self.w_length]
                _image_matrix[d_name][class_category] = gasf.transform(dummy)
                del dummy
        self.MTF_image_matrix = _image_matrix
        print(f'The MTF image representaion with window size {self.w_length} is finished')

if __name__ == '__main__':
    
    CWUB_object = CWUB()
    CWUB_object.GAF_representation()



# %%
