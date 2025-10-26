# vim: expandtab:ts=4:sw=4
import os
import numpy as np
import cv2
import scipy.io as sio
import json
import torch

# The maximum ID in the dataset.
#MAX_LABEL = 2618   #Dataset inteiro para veículos
#MAX_LABEL = 268     #Dataset com iou acima de 0.3 e LSS treinado para vehicle.car
#MAX_LABEL = 654      #Dataset com iou acima de 0.2 e LSS treinado para vehicle
#MAX_LABEL = 180     #Dataset com iou acima de 0.3 e LSS treinado para vehicle
#MAX_LABEL = 1147     #Dataset inteiro para a categoria human
#MAX_LABEL = 17     #Dataset com iou acima de 0.3 para a categoria human
#MAX_LABEL = 85     #Dataset com iou acima de 0.2 para a categoria human
MAX_LABEL = 306     #Dataset com iou acima de 0.1 para a categoria human

#128,64,3
IMAGE_SHAPE = 24, 24, 64

#CONCLUÍDO
def read_train_split_to_str(dataset_dir):
    """Read training data to list of filenames.

    Parameters
    ----------
    dataset_dir : str
        Path to the nuscenes_reid dataset directory.

    Returns
    -------
    (List[str], List[int], List[int])
        Returns a tuple with the following values:

        * List of image filenames (full path to image files).
        * List of unique IDs for the individuals in the images.
        * List of camera indices.

    """
    filenames, ids, camera_indices = [], [], []
    
    #Obtendo arquivo json do dataset
    data_folder = os.path.join(dataset_dir, 'data')
    filepath = os.path.join(dataset_dir, "object_data.json")
    with open(filepath) as json_file:
        object_dict:dict = json.load(json_file)           
    
    #Obtendo todos os elmentos com tag item para extração dos dados
    removed = 0
    for key in object_dict.keys():
        if str(object_dict[key]['category']).startswith('human') == False:
            removed += 1
            continue
        for i, data_path in enumerate(object_dict[key]["data_files"]):
            filenames.append(data_path)
            ids.append(int(object_dict[key]['id']))
            camera_indices.append(object_dict[key]["cam_id"][i])
            
    print(f'Total removidos: {removed}')
    print(f'Total: {int(MAX_LABEL) - removed}')
    
    return filenames, ids, camera_indices

#CONCLUÍDO
def read_train_split_to_tensor(dataset_dir, tensor_shape=(24,24)):
    """Read training images to memory. This consumes a lot of memory.

    Parameters
    ----------
    dataset_dir : str
        Path to the VeRi dataset directory.

    Returns
    -------
    (ndarray, ndarray, ndarray)
        Returns a tuple with the following values:

        * Tensor of images in BGR color space of shape 128x64x3.
        * One dimensional array of unique IDs for the individuals in the images.
        * One dimensional array of camera indices.

    """    
    # Definindo a camada convolucional que reduz os canais de 64 para 3
    conv_layer = torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)
    upsample_layer = torch.nn.Upsample(size=(128,64), mode='bilinear', align_corners=False)
    
    filenames, ids, camera_indices = read_train_split_to_str(dataset_dir)

    tensors = np.zeros((len(filenames), IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]), np.float32)
    for i, filename in enumerate(filenames):
        if i % 1000 == 0:
            print("Reading %s, %d / %d" % (dataset_dir, i, len(filenames)))
        t_data = torch.from_numpy(np.load(filename))        
        #t_conv = conv_layer(t_data)
        #Add batch layer to tensor (unsqueeze) and then remove this layer after output the model (squeeze)
        #t_up = upsample_layer(t_conv.unsqueeze(0)).squeeze(0)
        tensors[i] = t_data.permute(1,2,0).detach().numpy()
    
    ids = np.asarray(ids, np.int64)
    camera_indices = np.asarray(camera_indices)
    return tensors, ids, camera_indices


if __name__ == '__main__':
    dataset_dir = './downloads/datasets/nuscenes_reid'
    read_train_split_to_str(dataset_dir)