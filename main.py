import cv2
import glob
import tqdm
import os
import torch 
import numpy as np

from matplotlib import pyplot as plt


from config.parameters import ModelParameters, DataConfig



from utils import (
	generate_2d_gaussian,
	apply_fog_standard,
	apply_fog_extended,
	apply_fog_gaussian_filter,
	get_distance,
	get_img_name,
	get_img_dir,
	get_depth_name,
	colormap_depth,
	plot_matrix,
	get_predection
	)


data = DataConfig()
param = ModelParameters()





#Block name: Directories-------- defined the directory of all requirements for this project
path_images = data.path_images #'E:\Data\Waymo\data\extracted\images'
path_depth = data.path_depth

folder_to_save_extended_equation = data.folder_to_save_extended_equation
folder_to_save_standard_equation = data.folder_to_save_standard_equation
model_path = data.model_path
#End Block Directories.


#Block name: parameters
fog_scale = param.fog_scale
use_new_approach = True # If false, we use the standard equation, true for the new extended equation
beta = float(fog_scale/100)
matrix_size = param.matrix_size # the size of Gausian matrix. 'it should be odd number'
generat_depth = param.generat_depth

sigma = param.sigma
# End parameters


model = param.get_model()

img_pathes = glob.glob(os.path.join(path_images,'*','*','*.png'))
if not generat_depth:
	dep_pathes = glob.glob(os.path.join(path_depth,'*','*','*.png'))

print(f'The Number of images to synthsizing the fog is {len(img_pathes)}')

gaussian_matrix = generate_2d_gaussian(matrix_size, sigma=sigma)
plt_matrix = plot_matrix(gaussian_matrix,title='')
os.makedirs('img',exist_ok = True)
plt_matrix.savefig('img/gaussian_matrix.png')
loop = tqdm.tqdm(range(len(img_pathes)))
for i in loop:
	img_path = img_pathes[i]
	img_name = get_img_name(img_path)
	img_dir = get_img_dir(img_path)

	if (os.path.isfile(os.path.join(folder_to_save_extended_equation,img_dir,img_name))):
		continue

	
	if generat_depth:
		depth = get_predection(path=img_path,model=model)
	else:
		depth_path = dep_pathes[i]
		depth = cv2.imread(depth_path)
	img_distance = get_distance(depth)

	img = cv2.imread(img_path)
	

	path_save_standard = os.path.join(folder_to_save_standard_equation,img_dir)
	path_save_extended = os.path.join(folder_to_save_extended_equation,img_dir)
	os.makedirs(path_save_standard,exist_ok=True)
	os.makedirs(path_save_extended,exist_ok=True)


	fog_standard = apply_fog_standard(img, img_distance, beta=beta)
	fog_img_gaussian=apply_fog_extended(img,img_distance,beta=beta,group_size=matrix_size)
	#fog_img = apply_fog_gaussian_filter(img,img_distance,beta=beta,gaussian_matrix=gaussian_matrix)

	cv2.imwrite(os.path.join(path_save_standard,img_name), fog_standard)
	cv2.imwrite(os.path.join(path_save_extended,img_name),fog_img_gaussian)
	



print('-------Done-------')
'''
img_pathes = glob.glob(os.path.join(path_images,'*','*','*.png'))

num_test = 0 # number of image to test
path_img_test = pathes[num_test]#'other-codes/waymo.png'



img_test = cv2.imread(path_img_test)
img_name = get_img_name(path_img_test)
print(path_img_test)
print(img_name)
depth_name = get_depth_name(img_name)
print(depth_name)


depth_path = os.path.join(path_depth,depth_name)
img_depth = cv2.imread(depth_path)
print(img_depth.shape)
depth_distance = get_distance(img_depth[:,:,0])

gaussian_matrix = generate_2d_gaussian(matrix_size, sigma=sigma)

img_standard = apply_fog_standard(img_test, depth_distance, beta=beta, atmospheric_light=(255, 255, 255))
img_gausian = apply_fog_gaussian_filter(img_test,depth_distance,beta=beta,gaussian_matrix=gaussian_matrix,atmospheric_light=(255, 255, 255))





cv2.imwrite(os.path.join('img-test','depth_orginal.png'),img_depth)
cv2.imwrite(os.path.join('img-test',depth_name),depth_distance)

cv2.imwrite(os.path.join('img-test','img_standard.png'),img_standard)
cv2.imwrite(os.path.join('img-test','img_gausian.png'),img_gausian)

print('done')

'''