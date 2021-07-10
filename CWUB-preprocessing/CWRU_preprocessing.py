#%%
'''
! pip install pandas
! conda install -c conda-forge pyts
'''
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
        self._Image_representation()
        

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

    def _Image_representation(self):
        gasf = GramianAngularField(image_size=1., method='summation')
        sliced_data = {}
        image_matrix ={}

        for d_name in self.whole_data.keys(): # this loop goes through each dataset
            dataset = self.whole_data[d_name]
            sliced_data[d_name] = {}
            image_matrix[d_name] ={}
            for class_category in dataset.keys():# this loop goes through the eaech class of dataset 
                data = dataset[class_category]
                dummy = np.empty((len(data)//self.w_length, self.w_length))
                for i in range((len(data)//self.w_length)): # this loop goes through each class of a dataset
                    dummy[i,:] = data[i*self.w_length: (i+1)*self.w_length]
                image_matrix[d_name][class_category] = gasf.transform(dummy)
                del dummy
        self.image_matrix = image_matrix
        print(f'The image representaion with window size {self.w_length} is finished')

    def random_plots(self, num_plots):
        """
        num_plots: random number of images you wouldlike to see from the generated images
        **Note:** Make sure to pick a number that is dividable by 3
        """
        _selected_images = []
        _titles = []
        for dataset_name in self.image_matrix.keys():
            for i in range(num_plots//3):
                _dummy1 = np.random.randint(low=1, high=10, size=(1,))[0] ## choosing a random class
                _dummy2 = np.random.randint(low=0, high=500, size=(1,))[0] ## choosing a random image
                _selected_images += [self.image_matrix[dataset_name][_dummy1][_dummy2, :, :]]
                _titles += [ dataset_name + ' | '  + str(_dummy1) + ' | '  + str(_dummy2)]

        # Show the images for some random time series
        fig = plt.figure(figsize=(45, 25))
        grid = ImageGrid(fig, 111,
                        nrows_ncols=(num_plots//3, 3),
                        axes_pad=0.45,
                        share_all=True,
                        cbar_location="right",
                        cbar_mode="single",
                        cbar_size="7%",
                        cbar_pad=0.7,
                        )
        for image, title, ax in zip(_selected_images, _titles, grid):
            im = ax.imshow(image, cmap='rainbow', origin='lower')
            ax.set_title(title, fontdict={'fontsize': 12})
        ax.cax.colorbar(im)
        ax.cax.toggle_label(True)
        plt.suptitle('Gramian Angular Fields', y=0.98, fontsize=16)
        plt.show()
        

if __name__ == '__main__':

    CWUB_object = CWUB()
    CWUB_object.random_plots(num_plots=15)






# %%
