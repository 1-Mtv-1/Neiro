import numpy as np

from PIL import ImageDraw

from PIL import Image

import os

from numba import jit

import warnings

warnings.filterwarnings('ignore')
np.random.seed(1)

@jit(parallel=True)
def tanh(x):
    return np.tanh(x)   

@jit(parallel=True)
def softmax(x):
    temp = np.exp(x)    
    return temp/np.sum(temp,axis=1,keepdims=True)    
 
@jit(parallel=True)
def get_image_selction(layer, row_from, row_to, col_from, col_to):
    selection = layer[:,row_from:row_to,col_from:col_to]
    return selection.reshape(-1,1,row_to-row_from,col_to-col_from) 

def open_Net(path_to_weights,path_to_kernls, weights_size, kernls_size,name):
    print(f'-- start opening model {name} --')
    weights_matrix = np.ones(weights_size)
    weights = open(path_to_weights,'r')
    x_mat = 0
    y_mat = 0
    
    for x in weights.readlines():
        x_split = x.replace('[','')
        x_split = x_split.replace(']','')
        x_split = x_split.replace('\n','')
        x_split = x_split.split(' ')

        y_mat = 0
        for y in x_split:
            if y!='':
                weights_matrix[x_mat][y_mat] = float(y)
                y_mat+=1
        x_mat+=1  
    weights.close()  
    print('    weights ✓')

    weights = open(path_to_kernls,'r')
    k = weights.readlines()
    kernls = []

    for w in range(0,len(k),3):
        kernls.append(k[w]+k[w+1]+k[w+2])
    kernls_matrix = np.ones(kernls_size)
    x_mat = 0
    y_mat = 0
    for x in kernls:
        x_split = x.replace('[','')
        x_split = x_split.replace(']','')
        x_split = x_split.replace('\n','')
        x_split = x_split.split(' ')
        y_mat = 0
        for y in x_split:
            if y!='':
                kernls_matrix[x_mat][y_mat] = float(y)
                y_mat+=1
        x_mat+=1  
    weights.close()
    print('    kernls ✓')
    print('-- model successfully opened --')
    print()
    return (weights_matrix, kernls_matrix)


@jit(parallel=True)
def return_res(layer_0, kernls, weights_1_2, shape1,shape2,shape3):
    kernel_rows = 3 
    kernel_cols = 3 
    layer_0 = layer_0.reshape(shape1,shape2,shape3)
    sects = list()
    for row_start in range(layer_0.shape[1]-kernel_rows):
        for col_start in range(layer_0.shape[2]-kernel_cols):
            sect = get_image_selction(layer_0,row_start,row_start+kernel_rows,col_start,col_start+kernel_cols)
            sects.append(sect) 
    expanded_input = np.concatenate(sects,axis=1) 
    es = expanded_input.shape    
    flatted_input = expanded_input.reshape(es[0]*es[1],-1) 
    kernel_output = flatted_input.dot(kernls) 
    layer_1 = tanh(kernel_output.reshape(es[0],-1))  
    layer_2 = softmax(np.dot(layer_1, weights_1_2)) 
    return (bool(np.argmax(layer_2)),layer_2[0][1])


def pre_init():

    print('--- Start init neiro ---')
    print()

    model_28 = open_Net('models/big_data_28/weights_28px.txt','models/big_data_28/kernels_28px.txt',(10000,2),(9,16),'28px')
    model_128 = open_Net('models/small_data_128/weights_128px.txt','models/small_data_128/kernels_128px.txt',(250000,2),(9,16),'128px')

    return model_28,model_128


def single_res(path_to_image,model_28,model_128):
    image_28 = np.asarray(Image.open(path_to_image).convert('L').resize((28,28)),dtype='uint8')/255
    image_128 = np.asarray(Image.open(path_to_image).convert('L').resize((128,128)),dtype='uint8')/255
    res = (return_res(image_28,model_28[1],model_28[0],1,28,28)[0] + return_res(image_128,model_128[1],model_128[0],1,128,128)[1])/2
    return res


def switch_res(path_to_image,model_28,model_128):
    human_list = []
    image = Image.open(path_to_image)
    resize_width = 1000
    ot = (image.size[0],image.size[1])
    y_cord = resize_width/ot[0]
    k_resize = ot[0]/resize_width
    resize_image = image.convert('L').resize((resize_width,round(y_cord*ot[1])))
    image_size_y = round(y_cord*ot[1])
    image_size_x = resize_width
    crop_size_list = [1000,950,900,850,800,750,700,650,600,550,500,400,300]
    
    for crop_image_size in crop_size_list:
        for y in range(0, image_size_y-crop_image_size, crop_image_size//16):
            for x in range(0, image_size_x-crop_image_size, crop_image_size//16):
                continue_n = False
                for alr in human_list:
                    if (alr[0]<=x+crop_image_size/3) and (alr[0]+alr[2]>=x) and (alr[1]<=y+crop_image_size/3) and (alr[1]+alr[2]>=y):
                        
                        continue_n = True
                if continue_n: 
                    continue
                crop_28 = resize_image.crop((x, y, x+crop_image_size, y+crop_image_size)).convert('L').resize((28,28))
                asarray_28 = np.asarray(crop_28, dtype='uint8')
                res_28 = return_res(asarray_28/255,model_28[1],model_28[0],1,28,28)
                if res_28[1] > 0.90:
                    crop_128 = resize_image.crop((x, y, x+crop_image_size, y+crop_image_size)).convert('L').resize((128,128))
                    asarray_128 = np.asarray(crop_128, dtype='uint8')
                    res_128 = return_res(asarray_128/255,model_128[1],model_128[0],1,128,128)
                    if res_128[1]>0.90:
                        human_list.append([x, y, crop_image_size, (res_28[1]+res_128[1])*50 ])
    person_counter = 0 
    for sq in human_list:
        person_counter+=1
        draw = ImageDraw.Draw(image)
        draw.rectangle((sq[0]*k_resize, sq[1]*k_resize, (sq[0]+sq[2])*k_resize, (sq[1]+sq[2])*k_resize),outline=(0, 255, 0),width=2)
    image.save('static/image.jpg')

    
