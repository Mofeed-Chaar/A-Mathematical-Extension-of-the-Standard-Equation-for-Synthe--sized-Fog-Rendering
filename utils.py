
import numpy as np
import os
import cv2
import torch

from config.parameters import ModelParameters, DataConfig
from matplotlib import pyplot as plt

baseline = 22
focal = 5.6

param = ModelParameters()

def get_predection(path=None,model=None):

    device = param.device
    image_size = param.image_size
    
    image = cv2.imread(path)
    orginal_image=image.shape
    image = cv2.resize(image, image_size)
    image = image/256

    img = torch.from_numpy(image).float().to(device)

    img = img.unsqueeze(axis=0)
    img = img.permute(0,3,1,2)
    with torch.no_grad(): 
        preds = model(img)*126
    preds = torch.round(preds).to(torch.int16)
    preds = torch.where(preds<0,0,preds)
    preds = torch.where(preds>126,126,preds)

    preds = preds.squeeze()
    pred_im = preds.detach().cpu().numpy()
    pred_im = cv2.resize(pred_im,(orginal_image[1],orginal_image[0])).astype(int)
    
    pred_im = pred_im.astype(int)
    return pred_im


def generate_2d_gaussian(size, sigma,mu_x=0,mu_y=0):
    if size % 2 == 0:
        raise ValueError(f"matrix_size must be odd, got {matrix_size}.")
    """Generate a 2D Gaussian kernel centered in the matrix."""
    ax = np.linspace(-(size // 2), size // 2, size)
    coefficient = 1/(sigma*(np.pi)**0.5) 
    xx, yy = np.meshgrid(ax, ax)
    kernel = coefficient*np.exp(-((xx-mu_x)**2 + (yy-mu_y)**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)  # Normalize total sum to 1


def apply_fog_standard(image, depth_map, beta=0.1, atmospheric_light=(255, 255, 255)):
    """
    Apply fog effect to an image based on depth information.
    :param image: Input image (RGB format)
    :param depth_map: Depth map (grayscale, same size as image)
    :param beta: Fog density coefficient (higher values = denser fog)
    :param atmospheric_light: Color of the atmospheric light (default is white fog)
    :return: Foggy image
    """
    # Normalize depth map to [0, 1]
    #depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    
    # Compute fog effect
    transmission = np.exp(-beta * depth_map)  # Exponential decay based on depth
    transmission = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)
    foggy_image = (image * transmission + (np.array(atmospheric_light) * (1 - transmission)))
    
    return foggy_image.astype(np.uint8)


def apply_fog_extended(img,depth_map,beta=0.01,group_size=3,atmospheric_light=(255, 255, 255)):
    """
    Apply fog effect to an image based on depth information using an extended equation.
    :param image: Input image (RGB format)
    :param depth_map: Depth map (grayscale, same size as image), where the value of pixels is the distance in meters
    :param beta: Fog density coefficient (higher values = denser fog)
    :param atmospheric_light: Color of the atmospheric light (default is white fog)
    :return: Foggy image
    """
    trnas_filter = group_size
    group_size = trnas_filter//2
    foggy_image = np.zeros(img.shape)
    transmission = np.exp(-beta * depth_map)
    transmission = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)
    for m in range(group_size,img.shape[0]-group_size):
        for n in range(group_size,img.shape[1]-group_size):
        

            filter_constant = np.full((trnas_filter,trnas_filter,3),(np.array(atmospheric_light) * (1 - transmission[m,n,:])))/(trnas_filter**2)
            foggy_image[m-group_size:m+group_size+1,n-group_size:n+group_size+1,:] = foggy_image[m-group_size:m+group_size+1,n-group_size:n+group_size+1,:] + (img[m-group_size:m+group_size+1,n-group_size:n+group_size+1,:] * (transmission[m,n,:]/(trnas_filter**2)) + filter_constant)


    return foggy_image.astype(np.uint8)



def apply_fog_gaussian_filter(img,depth_map,beta=0.01,gaussian_matrix=None,atmospheric_light=(255, 255, 255)):
    """
    Apply fog effect to an image based on depth information using an extended equation.
    :param image: Input image (RGB format)
    :param depth_map: Depth map (grayscale, same size as image), where the value of pixels is the distance in meters
    :param beta: Fog density coefficient (higher values = denser fog)
    :param atmospheric_light: Color of the atmospheric light (default is white fog)
    :return: Foggy image
    """
    trnas_filter = gaussian_matrix.shape[0]
    group_size = gaussian_matrix.shape[0]//2
    foggy_image = np.zeros(img.shape)
    transmission = np.exp(-beta * depth_map)
    transmission = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)

    gaussian_matrix = np.repeat(gaussian_matrix[:, :, np.newaxis], 3, axis=2)
    #print(gaussian_matrix.shape)
    for m in range(group_size,img.shape[0]-group_size):
        for n in range(group_size,img.shape[1]-group_size):
        

            filter_constant = np.full((trnas_filter,trnas_filter,3),(np.array(atmospheric_light) * (1 - transmission[m,n,:])))
            foggy_image[m-group_size:m+group_size+1,n-group_size:n+group_size+1,:] = foggy_image[m-group_size:m+group_size+1,n-group_size:n+group_size+1,:] + (img[m-group_size:m+group_size+1,n-group_size:n+group_size+1,:] * (transmission[m,n,:]) + filter_constant)*gaussian_matrix


    return foggy_image.astype(np.uint8)




def get_distance_pixel(pixel):
	if (pixel < 2):
		return 32000
	disparity = (float(pixel)-1)/256
	d = (baseline*focal)/disparity
	return d
def get_distance(img):
    if len(img.shape) == 3:
        img = img[:,:,0]
    img_distance = np.zeros((img.shape[0],img.shape[1]))
	
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            img_distance[m,n]=get_distance_pixel(img[m,n])
    	
    return img_distance/100

def get_img_name(path):
    img_name = path.split(os.path.sep)[-1]
    return img_name


def get_img_dir(path,depth_dir=-3):
    img_name = path.split(os.path.sep)[depth_dir:-1]
    img_name = os.path.sep.join(img_name)

    return img_name



def get_depth_name(img_name):
    list_img = img_name.split('_')
    depth = '_'.join(list_img[:-1])
    depth = depth + '_disparity.png'
    return depth


def colormap_depth(path=None,img=None):
    if path is not None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # Normalize to 8-bit
    img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_8bit = img_normalized.astype(np.uint8)

    # Apply colormap
    img_colored = cv2.applyColorMap(img_8bit, cv2.COLORMAP_JET)

    # Save result
    return img_colored


def plot_matrix(matrix,title='Positive Gaussian Matrix with Gaussian Diagonal Sums'):
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title(title)
    return plt



def synthesize_fog(img=None,Depth=None,Gaussian_filter=True):
    img_distance = get_distance(depth)
    pass