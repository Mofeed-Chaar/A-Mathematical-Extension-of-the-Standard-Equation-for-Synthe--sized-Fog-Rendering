import glob
import os 
import torch

class ModelParameters:
    def __init__(self):
        self.matrix_size = 19 # the Gaussian matrix filter size
        self.sigma= 5.0 
        self.mu_x=0
        self.mu_y=self.mu_x
        self.fog_scale = 3 # it is beta in the equation
        self.image_size = (480,320) # (W,H) the size of image to rescall it for prediction
        self.orginal_image_size = (2048,1024) # (W,H)

        self.baseline = 22
        self.focal = 5.6
        self.generat_depth = True # if True, the depth image will generated, False: there is depth image on datasets
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = 'highst_acc_model.pt'


    def get_model(self):
        model = torch.load(self.model_path,weights_only=False)
        model = model.to(self.device)
        model.eval()
        print('Model has uploaded')
        return model




class DataConfig:
    def __init__(self):
        self.path_images = '/mnt/E/Data/Waymo/data/samples/images'
        self.path_depth = '/mnt/E/Data/cityscapes/image-segmentation/Data/depth'
        self.shuffle = True
        parameters = ModelParameters()
        self.folder_to_save_extended_equation = 'fog_generated/fog_Gaussian_beta'+str(parameters.fog_scale)
        self.folder_to_save_standard_equation = 'fog_generated/fog_beta'+str(parameters.fog_scale)
        self.model_path = 'highst_acc_model.pt'


    def __getitem__(self, index):
        return self._pathes[index]