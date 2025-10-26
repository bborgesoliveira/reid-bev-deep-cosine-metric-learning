import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
import cv2
from cv2.typing import MatLike
import numpy as np
from PIL import Image

import torch
from torch import nn
from tools.lss_transform.src.tools import normalize_img, img_transform, denormalize_img, gen_dx_bx, get_batch_iou
from tools.lss_transform.src.models import compile_model, LiftSplatShoot
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

import json
import time
from datetime import datetime

plt.ioff()

# Path to your nuScenes dataset
DATASET_PATH = './downloads/datasets/nuscenes'
destination_path = './downloads/datasets/nuscenes_reid_human_iou10/'

max_x = 0
max_y = 0

# Initialize NuScenes object
#nusc = NuScenes(version='v1.0-trainval', dataroot=DATASET_PATH, verbose=True)

class LSSTransformModified(LiftSplatShoot):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LSSTransformModified, self).__init__(grid_conf, data_aug_conf, outC)
        
        self.outC = outC
        
    def compile(self):
        self.model = compile_model(self.grid_conf, self.data_aug_conf, outC=self.outC)
        #Path to pre-trained model
        if self.outC == 1:
            modelf = '/lasi/cosine_metric_learning/tools/lss_transform/models/1channel/model525000.pt'
        elif self.outC == 3:
            modelf = '/lasi/cosine_metric_learning/tools/lss_transform/models/3channel/model730000.pt'
        elif self.outC == 64:
            #modelf = '/lasi/cosine_metric_learning/tools/lss_transform/models/64channel/model670000.pt'
            #modelf = '/lasi/cosine_metric_learning/tools/lss_transform/models/dual_loss_64channel/model830000.pt'
            #modelf = '/lasi/cosine_metric_learning/tools/lss_transform/models/dual_loss_64channel_vehicles/model290000.pt'
            modelf = '/lasi/cosine_metric_learning/tools/lss_transform/models/dual_loss_64channel_all/model290000.pt'
        self.model.load_state_dict(torch.load(modelf))
        self.model.eval()
        
    def transform(self, imgs, rots, trans, intrins, post_rots, post_trans, binimg=None):
        """ Apply transform to image. Return image mapped to BEV and iou if binimg is not None.         
        """
        x, pred_multi_channel, pred_one_channel = self.model(imgs, rots, trans, intrins, post_rots, post_trans)
        
        iou = 0
        if binimg is not None:
            # iou
            intersect, union, _ = get_batch_iou(pred_one_channel, binimg)
            iou = intersect / union
    
        return pred_multi_channel, iou

class NuscenesReId():
    def __init__(self, data_aug_conf, grid_conf):
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=DATASET_PATH, verbose=True)       
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()
        
    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']

        resize = max(fH/H, fW/W)
        resize_dims = (int(W*resize), int(H*resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        return resize, resize_dims, crop, flip, rotate
    
    def is_fully_visible(self, box: Box, image_size: tuple) -> bool:
        corners = view_points(box.corners(), np.eye(4), normalize=True)[:2, :]
        min_corner = corners.min(axis=1)
        max_corner = corners.max(axis=1)
        if np.any(min_corner < 0) or np.any(max_corner > np.array(image_size)):
            return False
        return True
    
    def box_to_bev(self, sample, ann): 
        """ Transform bbox coordinates of ann from image view to bev, using sample data to get
            the coordinates of the ego (center of view).
        """   
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound'])
        dx, bx, nx = dx.numpy(), bx.numpy(), nx.numpy()
        egopose = self.nusc.get('ego_pose',
            self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token'])
        box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        box.translate(trans)
        box.rotate(rot)

        pts = box.bottom_corners()[:2].T
        pts = np.round(
            (pts - bx[:2] + dx[:2]/2.) / dx[:2]
            ).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]                            

        return pts  
    
    def get_binimg(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if ((not inst['category_name'].split('.')[0] == 'vehicle') and
            (not inst['category_name'].split('.')[0] == 'human')): #Insere categoria para incluir pedestres no treinamento
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img).unsqueeze(0)
    
    def get_image_data(self, cam_data):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []    
        
        imgname = os.path.join(DATASET_PATH, cam_data['filename'])
        img = Image.open(imgname)
        
        sens = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
        tran = torch.Tensor(sens['translation'])
        intrin = torch.Tensor(sens['camera_intrinsic'])
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)
        
        # augmentation (resize, crop, horizontal flip, rotate)
        resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
        img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                resize=resize,
                                                resize_dims=resize_dims,
                                                crop=crop,
                                                flip=flip,
                                                rotate=rotate,
                                                )
        post_tran = torch.zeros(3)
        post_rot = torch.eye(3)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2
        
        imgs.append(normalize_img(img))
        intrins.append(intrin)
        rots.append(rot)
        trans.append(tran)
        post_rots.append(post_rot)
        post_trans.append(post_tran)
        
        #Adiciona dimensão que representa o número de câmeras, visto que foi utilizada apenas
        #uma câmera
        return(torch.stack(imgs).unsqueeze(0), torch.stack(rots).unsqueeze(0), 
                    torch.stack(trans).unsqueeze(0), torch.stack(intrins).unsqueeze(0), 
                    torch.stack(post_rots).unsqueeze(0), torch.stack(post_trans).unsqueeze(0))
        
    def save_to_file(self, feat, filename, bbox=None, rgb=False, plot_center=True, save_full_img=False):
        """ Cut features tensor (or image) in bbox and save to .npy (numpy) or .jpg file.
        """
        def draw_rect(ax, selected_corners, is_bev=False):
            prev = selected_corners[-1]
            for corner in selected_corners:
                if is_bev:
                    #Caso o plano seja BEV, as coordenadas da bbox ficam invertidas
                    ax.plot([prev[1], corner[1]], [prev[0], corner[0]], color='b', linewidth=2)
                else:
                    ax.plot([prev[0], corner[0]], [prev[1], corner[1]], color='b', linewidth=2)
                prev = corner
                
        def plot(data, final_dim=(128,352), centralize=[0,0,0,0]):
            val = 0.01
            fH, fW = final_dim
            fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
            gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
            gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)
            
            plt.clf()
            ax = plt.subplot(gs[0, :])
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            summed = torch.sum(data, dim=0)            
                       
            array = summed.detach().numpy()
            X, Y = np.nonzero(array)
            nonzero = array[np.nonzero(array)]
            c = nonzero.flatten()     
            
            scatter = ax.scatter(X, Y, c=c, cmap='viridis', marker='o')
            
            #Recebe coordenadas delimitadoras do frame (x_min, x_max, y_min, y_max)
            if centralize != [0,0,0,0]:
                ax.axis(centralize)
                            
            return ax
                
        val = 0.01
        final_dim=(128, 352)
        fH, fW = final_dim
        fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
        gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
        gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)
                
        if rgb:   
            if bbox is not None:
                A = np.min(bbox[:,0])
                B = np.max(bbox[:,0])
                C = np.min(bbox[:,1])
                D = np.max(bbox[:,1]) 
                x_min, y_min = int(A), int(C)
                x_max, y_max = int(B), int(D)
            
            plt.clf()
            ax = plt.subplot(gs[0, :])
            #ax = plt.subplot()
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            
            #cropped_img = feat.crop((A, C, B, D))         
             
            if bbox is not None:
                draw_rect(ax, bbox) 
            #plt.axis([x_min, x_max, y_min, y_max])
            #plt.gca().invert_yaxis()
            plt.imshow(feat) 
            plt.savefig(filename)
            plt.close()
            return      
        else:
            #Getting bbox coordinates
            if bbox is not None:
                #print(bbox)
                #Na transformação para o plano BEV, as coordenadas se invertem com os eixos X e Y.
                #A e B estão no eixo Y e C e D no eixo X
                A = np.min(bbox[:,0])+1
                B = np.max(bbox[:,0])+1
                C = np.min(bbox[:,1])+1
                D = np.max(bbox[:,1])+1
                x_min, y_min = int(C), int(A)
                x_max, y_max = int(D), int(B)
                #print(f'\nx_min = {x_min}\tx_max = {x_max}\ny_min = {y_min}\ty_max = {y_max}')
                
                global max_x, max_y
                
                #print(f'\nCropped size: {cropped_feat.shape}')
                #_, s_x, s_y = cropped_feat.shape
                s_x = x_max - x_min
                s_y = y_max - y_min
                if s_x > max_x:
                    max_x = s_x
                if s_y > max_y:
                    max_y = s_y
                    
                print(f'max_x = {max_x}\tmax_y = {max_y}\n')
                
                #Calcula mínimo e máximo de uma janela de 32x88 com a bbox no centro
                #height, width = (32,88)
                height, width = (24,24)               
                x_window_gap = int((width-s_x) / 2)
                x_window_min = x_min - x_window_gap
                x_window_max = x_max + x_window_gap
                y_window_gap = int((height-s_y) / 2)
                y_window_min = y_min - y_window_gap
                y_window_max = y_max + y_window_gap
                
                #Corrigindo tamanho da janela se for menor que width e height.
                #Acontece nos casos em que a diferença entre a janela e o objeto seja ímpar
                if x_window_max - x_window_min < width:
                    x_window_min -= 1
                if y_window_max - y_window_min < height:
                    y_window_min -= 1
                
                #Cutting features tensor or image in bbox coordinates
                #cropped_feat = torch.clone(feat)
                cropped_feat = torch.zeros(feat.shape)
                cropped_feat[:,x_min:x_max, y_min:y_max] = feat[:,x_min:x_max, y_min:y_max]
                cropped_tensor = cropped_feat[:, x_window_min:x_window_max, y_window_min:y_window_max]
                #print(f'Cropped_shape: {cropped_tensor.shape}')
                if cropped_tensor.shape != (64, width, height):
                    return False
                
                ax = plot(cropped_feat, centralize=[x_window_min, x_window_max, y_window_min, y_window_max])
                
                # if bbox is not None:                 
                #     draw_rect(ax, bbox, True)
                    
                # if plot_center:
                #     plt.scatter([99], [99], color='r', marker='x')
                
                #Saving tensor to .npy
                tensor_filename = filename.replace('/images/','/data/')
                with open(tensor_filename, 'wb') as f:
                    np.save(f, cropped_tensor.detach().numpy())

                #print('saving', filename)
                plt.savefig(filename.replace('.npy', '_cropped.jpg'))
                plt.close()
                
            if save_full_img:
                plt.clf()
                #ax = plt.subplot(gs[0, :])
                ax = plt.subplot(gs[0, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                # conv_layer = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
                # x = conv_layer(feat)  
                min_dataset = -3.9697017669677734
                max_dataset = 6.039807319641113 
                #print(f'\nfeat_shape: {feat.shape}\n')
                x = feat.detach().numpy()
                x = (x-min_dataset) / (max_dataset - min_dataset)   
                X, Y, Z = np.nonzero(x)
                nonzero = x[np.nonzero(x)]
                c = nonzero.flatten()   
                
                scatter = ax.scatter(Y, Z, c=c, cmap='viridis', marker='o')
                
                ax = plot(feat) 
                if bbox is not None:                 
                    draw_rect(ax, bbox, True)            
                
                if plot_center:
                    ax.scatter([99], [99], color='r', marker='x')
                
                # plt.setp(ax.spines.values(), color='b', linewidth=2)
                # #plt.imshow(x.squeeze(0).detach(), vmin=0, vmax=1, cmap='Blues')
                # plt.imshow(x.squeeze(0).detach(), vmin=-3.9697017669677734, vmax=6.039807319641113, cmap='viridis')
                
                print('saving', filename)
                plt.savefig(filename.replace('.npy', '_full.jpg'))
                plt.close()
            
            return True                  
        
        
    
def extract_images(output_dir='output', image_shape=(256,704), is_sample=False, save_full_img=False):
    """ Processa imagens do Nuscenes para converter em BEV, identificar por objeto e exportar imagens
        e tensor de saída (formato width x height x 64)
    """
    
    H=900
    W=1600
    resize_lim=(0.193, 0.225)
    final_dim=(128, 352)
    bot_pct_lim=(0.0, 0.22)
    rot_lim=(-5.4, 5.4)
    rand_flip=True

    xbound=[-50.0, 50.0, 0.5]
    ybound=[-50.0, 50.0, 0.5]
    zbound=[-10.0, 10.0, 20.0]
    dbound=[4.0, 45.0, 1.0]
    
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': cams,
        'Ncams': 5,
    }
    
    nusc = NuscenesReId(data_aug_conf=data_aug_conf, grid_conf=grid_conf)
    model = LSSTransformModified(grid_conf=grid_conf, data_aug_conf=data_aug_conf, outC=64)
    model.compile()
    
    os.makedirs(output_dir, exist_ok=True)
    data_folder = output_dir.replace('images', 'data')
    os.makedirs(data_folder, exist_ok=True)
    object_dict = {}
    
    start = datetime.now()
    
    total = len(nusc.nusc.sample)
    for i, sample in enumerate(nusc.nusc.sample):
        print(f'Tempo gasto: {datetime.now() - start}')
        print(f'\nIniciando sample {i} de {total}')
        #Interrompe quando houver 100 objetos detectados
        if is_sample:
            if len(object_dict.keys()) >= 10:
                break
        for cam_id, sensor_name in enumerate(['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']):
            cam_data = nusc.nusc.get('sample_data', sample['data'][sensor_name])
            cam_intrinsics = nusc.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])['camera_intrinsic']
            original_img_path = os.path.join(DATASET_PATH, cam_data['filename'])
            image = cv2.imread(original_img_path)
            image_size = (image.shape[1], image.shape[0])  # (width, height)
                        
            ann_tokens = sample['anns']
            for ann_token in ann_tokens:
                ann = nusc.nusc.get('sample_annotation', ann_token)
                instance_token = ann['instance_token']
                
                visibility_token = int(ann['visibility_token'])
                
                if visibility_token < 4:
                    continue  # Ignora objetos que não tenham "boa" visibilidade
                
                category_name = ann['category_name']
                
                #if 'vehicle' in category_name or 'human' in category_name:
                if 'human' in category_name:
                    data_path, boxes, camera_intrinsic = nusc.nusc.get_sample_data(sample['data'][sensor_name], BoxVisibility.ALL, [ann['token']])
                    
                    if len(boxes) == 0:
                        continue
                    
                    box = boxes[0]
                    
                    if nusc.is_fully_visible(box, image_size):
                        #Replicando código do render para desenhar um quadrado que cubra todo o objeto
                        corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
                        #Obtendo os cantos <X=A, >X=B, <Y=C e >Y=D
                        A = np.min(corners[0,:])
                        B = np.max(corners[0,:])
                        C = np.min(corners[1,:])
                        D = np.max(corners[1,:])                          
                                   
                        box_bev = nusc.box_to_bev(sample, ann)
                        
                        ################################
                        #Caso as coordenadas da bbox no bev sejam negativas, maiores que o tamanho
                        #máximo da imagem (200,200) ou tiver dimensão nula, ignora sample
                        print(f'instance: {instance_token}')
                        objs = 0
                        if instance_token in object_dict:
                            if 'images' in object_dict[instance_token]:
                                objs = len(object_dict[instance_token]["images"])
                                
                        #print(f'id: {objs}')
                        #print(f'{box_bev}')
                        out_of_bev = False
                        x_max = np.max(box_bev[:,1])
                        x_min = np.min(box_bev[:,1])
                        y_max = np.max(box_bev[:,0])
                        y_min = np.min(box_bev[:,0])
                        area_bev = (x_max - x_min)*(y_max-y_min)
                        #Verifica dimensão nula
                        if area_bev == 0:
                            continue
                        
                        for coord in box_bev:
                            #Verifica extremos da imagem
                            if (coord[0] < 0 or coord[0] > 200 or 
                                coord[1] < 0 or coord[1] > 200):                                
                                out_of_bev = True
                                break
                        
                        if out_of_bev:
                            #print('Ignored...')
                            continue   
                        
                        ################################                         
                        
                        #Corta imagem na bounding box
                        #cropped_img = image.crop((A, C, B, D))
                        x_min, y_min = int(A), int(C)
                        x_max, y_max = int(B), int(D)
                        cropped_img = highlight_image_frame(image, x_min, x_max, y_min, y_max)
                        cropped_img = image[y_min:y_max, x_min:x_max]
                        area = (x_max - x_min)*(y_max - y_min)
                                                
                        imgs, rots, trans, intrins, post_rots, post_trans = nusc.get_image_data(cam_data=cam_data)
                        
                        #Obtem imagem binária da amostra
                        binimg = nusc.get_binimg(sample)
                        
                        bev_image, iou = model.transform(imgs, rots, trans, intrins, post_rots, post_trans, binimg=binimg)                     
                        print(f'iou: {iou}')   
                        if iou < 0.1:
                             continue                   
                        
                        if len(cropped_img) == 0 or area < 1000:
                            continue
                        
                        #Salva arquivo com base no instance_token, que identifica o objeto                     
                        if instance_token not in object_dict:
                            object_dict[instance_token] = {
                                'images': [],
                                'data_files': [],
                                'intrinsics': [],
                                'category': category_name,
                                'bbox': [],
                                'cam_name': [],
                                'cam_id': []
                            }                     
                                                
                        image_filename = f"{instance_token}_{len(object_dict[instance_token]['images'])}.jpg"
                        image_path = os.path.join(output_dir, image_filename)
                                                                            
                        data_path = ''
                        saved = False
                        for i, x in enumerate(bev_image): 
                            #print(x.shape)  
                            if box_bev is not None:     
                                data_path = os.path.join(output_dir, f"{instance_token}_{len(object_dict[instance_token]['images'])}_{i}_bev.npy")                 
                                saved = nusc.save_to_file(x, data_path, box_bev, save_full_img=save_full_img)
                                if saved is False:
                                    #Se retornar None ao salvar, indica erro no tamanho do arquivo
                                    #Nesse caso ignora os demais arquivos
                                    break
                                data_path = data_path.replace('/images/', '/data/')
    
                        if saved is False:
                            continue
                        
                        edges = np.array([[A, C], [A, D], [B, D], [B, C]])
                        pil_img = Image.open(original_img_path) 
                        nusc.save_to_file(pil_img, image_path, edges, True)
                        
                        #Salva imagem binária
                        filename = os.path.join(output_dir, f"{instance_token}_{len(object_dict[instance_token]['images'])}_{i}_binimg.jpg")
                                                
                        plt.clf()
                        ax = plt.subplot()
                        ax.get_xaxis().set_ticks([])
                        ax.get_yaxis().set_ticks([])
                        
                        #cropped_img = feat.crop((A, C, B, D))   
                        X, Y, Z = np.indices(binimg.shape)
                        c = binimg.flatten()      
                        scatter = ax.scatter(Y, Z, c=c, cmap='viridis', marker='o')
                        ax.scatter([99], [99], color='r', marker='x')
                        ax.scatter(sum(box_bev[:,1])/4, sum(box_bev[:,0])/4, color='g', marker='x')
                        plt.savefig(filename)
                        plt.close()
                            
                        #Salvando imagem com opencv
                        #cv2.imwrite(image_path, image)
                        #cv2.imwrite(os.path.join(output_dir, f"{instance_token}_{len(object_dict[instance_token]['images'])}_redim.jpg"), cropped_img)
                        #cv2.imwrite(os.path.join(output_dir, f"{instance_token}_{len(object_dict[instance_token]['images'])}_bev.jpg"), bev_image)
                        
                        object_dict[instance_token]['images'].append(image_path)
                        object_dict[instance_token]['data_files'].append(data_path)
                        object_dict[instance_token]['intrinsics'].append(cam_intrinsics)
                        object_dict[instance_token]['bbox'].append(box_bev.tolist())
                        object_dict[instance_token]['cam_name'].append(sensor_name)
                        object_dict[instance_token]['cam_id'].append(f'c00{cam_id+1}')
                        
                        # Salva parcialmente o documento json
                        with open(os.path.join(destination_path, 'object_data.json'), 'w') as f:
                            json.dump(object_dict, f, indent=4)
                            
                        print(f'{len(object_dict.keys())} objetos salvos...')
                        
    #Eliminando objetos com menos de 5 imagens
    total = len(object_dict.keys())
    filtered_dict = {}
    for i, token in enumerate(object_dict):
        print(f'Analisando objeto {i} de {total}')
        if len(object_dict[token]['images']) >= 4:
            filtered_dict[token] = object_dict[token] 
        else:
            #Remove imagens salvas do objeto não adicionado
            rm_list = [x for x in os.listdir(output_dir) if x.startswith(token)]
            for item in rm_list:    
                path = os.path.join(output_dir, item)            
                os.remove(path)  
                
            rm_list = []
            rm_list = [x for x in os.listdir(data_folder) if x.startswith(token)]
            for item in rm_list:
                #Remove também arquivo na pasta data
                data_path = os.path.join(data_folder, item)                
                os.remove(data_path)
                
    print(f'Tempo gasto: {datetime.now() - start}')
    
    return filtered_dict




def resize_image_without_distortion(img, largura_destino, altura_destino):
    # Obtém as dimensões da imagem original
    altura_original, largura_original = img.shape[:2]

    # Calcula a razão de aspecto da imagem original e da imagem de destino
    razao_original = largura_original / altura_original
    razao_destino = largura_destino / altura_destino

    if razao_original > razao_destino:
        # A imagem é mais larga do que a proporção de destino
        nova_largura = largura_destino
        nova_altura = int(largura_destino / razao_original)
    else:
        # A imagem é mais alta do que a proporção de destino
        nova_altura = altura_destino
        nova_largura = int(altura_destino * razao_original)

    # Redimensiona a imagem mantendo a razão de aspecto original
    img_redimensionada = cv2.resize(img, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)

    # Calcula os valores de preenchimento (padding) para centralizar a imagem
    padding_horizontal = (largura_destino - nova_largura) // 2
    padding_vertical = (altura_destino - nova_altura) // 2

    # Aplica o preenchimento para alcançar as dimensões de destino
    img_com_padding = cv2.copyMakeBorder(img_redimensionada, 
                                         padding_vertical, altura_destino - nova_altura - padding_vertical, 
                                         padding_horizontal, largura_destino - nova_largura - padding_horizontal, 
                                         cv2.BORDER_CONSTANT, 
                                         value=[0, 0, 0])  # Preenche com preto, mas pode usar outra cor

    return img_com_padding

def highlight_image_frame(img:MatLike, x_min:int, x_max:int, y_min:int, y_max:int):
    highlighted_image = np.zeros((img.shape[0], img.shape[1],3))
    highlighted_image[y_min:y_max, x_min:x_max] = img[y_min:y_max, x_min:x_max]
    return highlighted_image

def filter_data_folder():
    data_path = os.path.join(destination_path, 'data')
    with open(os.path.join(destination_path, 'object_data.json'), 'r') as f:
        object_dict = json.load(f)
        
    list_files = os.listdir(data_path)
    total = len(list_files)
    remove_list = []
    for i, file in enumerate(list_files):
        print(f'Analisando objeto {i} de {total}')
        file_path = os.path.join(data_path, file)
                
        token = file.split('_')[0]
        if token not in object_dict.keys():
            remove_list.append(file_path)
            os.remove(file_path)
            
    print(f'\nTotal após filtragem: {total - len(remove_list)}\n')
    print(remove_list[:2])
    
    print(f'\nTotal de objetos na pasta: {len(os.listdir(data_path))}')
    
def update_data_files_to_json():
    data_path = os.path.join(destination_path, 'data')
    with open(os.path.join(destination_path, 'object_data.json'), 'r') as f:
        object_dict:dict = json.load(f)
        
    for key in object_dict.keys():
        data_files = []
        for image_path in object_dict[key]['images']:
            data_files.append(image_path.replace('.jpg', '_0_bev.npy'))
        
        object_dict[key]['data_files'] = data_files
        
    # Save the object data dictionary if needed
    with open(os.path.join(destination_path, 'object_data.json'), 'w') as f:
        json.dump(object_dict, f, indent=4)

def enumerate_tokens():
    data_path = os.path.join(destination_path, 'data')
    with open(os.path.join(destination_path, 'object_data.json'), 'r') as f:
        object_dict:dict = json.load(f)
        
    for i, key in enumerate(object_dict.keys()):        
        object_dict[key]['id'] = i+1
        
    # Save the object data dictionary if needed
    with open(os.path.join(destination_path, 'object_data.json'), 'w') as f:
        json.dump(object_dict, f, indent=4)

def fix_data_path():
    data_path = os.path.join(destination_path, 'data')
    with open(os.path.join(destination_path, 'object_data.json'), 'r') as f:
        object_dict:dict = json.load(f)
        
    for key in object_dict.keys():
        for i, data_path in enumerate(object_dict[key]['data_files']):
            object_dict[key]['data_files'][i] = data_path.replace('/images/', '/data/')
        
    # Save the object data dictionary if needed
    with open(os.path.join(destination_path, 'object_data.json'), 'w') as f:
        json.dump(object_dict, f, indent=4)

def check_sensors():
    data_path = os.path.join(destination_path, 'data')
    with open(os.path.join(destination_path, 'object_data.json'), 'r') as f:
        object_dict:dict = json.load(f)
        
    # cam_dict = dict()
        
    # nusc = NuScenes(version='v1.0-trainval', dataroot=DATASET_PATH, verbose=True) 
    # cams = [x for x in nusc.sensor if x['modality'] == 'camera']
    # for cam in cams:
    #     cam['intrinsics'] = []
    #     cam_dict[cam['token']] = cam
    
    # for sample in nusc.sample:
    #     for key in cam_dict.keys():
    #         cam = cam_dict[key]
    #         cam_data = nusc.get('sample_data', sample['data'][cam['channel']])
    #         cam_intrinsics = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])['camera_intrinsic']
            
    #         if cam_intrinsics not in cam['intrinsics']:
    #             cam['intrinsics'].append(cam_intrinsics)
                
    # with open('./downloads/datasets/nuscenes_reid/cam_data.json', 'w') as f:
    #     json.dump(cam_dict, f, indent=4)
            
    with open(os.path.join(destination_path, 'cam_data.json'), 'r') as f:
        cam_data:dict = json.load(f)
        
    for key in object_dict.keys():
        for x in object_dict[key]['intrinsics']:
            flag = False
            for cam_key in cam_data.keys():
                if x in cam_data[cam_key]['intrinsics']:
                    flag = True
                    
                    if 'cam_name' not in object_dict[key]:
                        object_dict[key]['cam_name'] = []
                    if 'cam_id' not in object_dict[key]:
                        object_dict[key]['cam_id'] = []
                    
                    object_dict[key]['cam_name'].append(cam_data[cam_key]['channel'])
                    object_dict[key]['cam_id'].append(cam_data[cam_key]['id'])
                    break
            
            if flag == False:
                print(f'Objeto {key} não corresponde.')
                
    with open(os.path.join(destination_path, 'object_data.json'), 'w') as f:
        json.dump(object_dict, f, indent=4)
            

def open_npy_img(filename):
    conv_layer = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
    
    val = 0.01
    final_dim=(128, 352)
    fH, fW = final_dim
    fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)
        
    plt.clf()
    ax = plt.subplot(gs[0, :])
    #ax = plt.subplot()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
    t_data = torch.from_numpy(np.load(f'{filename}.npy'))      
    t_conv = t_data  
    if t_data.shape[0] != 3 and t_data.shape[0] != 1:
        t_conv = conv_layer(t_data)
    
    t_conv = t_conv.permute(1,2,0).detach().numpy()
            
    plt.imshow(t_conv) 
    plt.savefig(f'{filename}.jpg')
    plt.close()

if __name__ == '__main__':  
    # open_npy_img('tensor')
    # open_npy_img('tensor_out_three')
    # open_npy_img('binimg')
    # exit()
                         
    # Run the extraction process
    object_data = extract_images(output_dir=os.path.join(destination_path, 'images'), is_sample=False, save_full_img=False)

    # Save the object data dictionary if needed
    with open(os.path.join(destination_path, 'object_data.json'), 'w') as f:
        json.dump(object_data, f, indent=4)
        
    print('\n\nAtualizando numeração dos tokens:')
    enumerate_tokens()