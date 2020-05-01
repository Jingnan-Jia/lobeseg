# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:20:10 2017
@author: fferreira
"""

import numpy as np
from futils.util import sample_scan
from  scipy import ndimage
from futils.vpatch import deconstruct_patch,reconstruct_patch
from tensorflow.keras.models import model_from_json
import tensorflow.keras.backend as K
import time


def one_hot_decoding(img,labels,thresh=[]):
    
    new_img = np.zeros((img.shape[0],img.shape[1]))
    r_img   = img.reshape(img.shape[0],img.shape[1],-1)

    aux = np.argmax(r_img,axis=-1)
    
    
    for i,l in enumerate(labels[1::]):
        if(thresh==[]):        
            new_img[aux==(i+1)]=l
        else:
            new_img[r_img[:,:,i+1]>thresh] = l

    return new_img



class v_segmentor(object):
    def __init__(self,batch_size=1,model='MODEL.hdf5',ptch_sz=128,ptch_z_sz=64,trgt_sz=256,trgt_z_sz=128, patching=True,
                  trgt_space_list=None, task='lobe'):
        self.batch_size = batch_size
        self.model      = model
        self.ptch_sz    = ptch_sz   
        self.z_sz       = ptch_z_sz
        self.trgt_sz    = trgt_sz
        self.trgt_z_sz  = trgt_z_sz
        self.trgt_space_list = trgt_space_list # 2.5, 1.4, 1.4
        if task=='lobe':
            self.labels = [0, 4, 5, 6, 7, 8]
        elif task=='vessel':
            self.labels = [0, 1]
        else:
            print('please assign the task name for prediction')

        
        if(self.trgt_sz!=self.ptch_sz) and patching and (self.ptch_sz!=None and self.ptch_sz!=0) :
            self.patching = True
        else:
            self.patching = False

        if type(self.model) is str: # if model is loaded from a file
            model_path = model.split(".hdf5")
            model_path = model.split(".hdf5")[0]+'.json'
            with open(model_path, "r") as json_file:
                json_model = json_file.read()
                self.v = model_from_json(json_model)

            self.v.load_weights((self.model))
        else: # model is a tf.keras model directly in RAM
            self.v = self.model

    def _normalize(self,scan):
        """returns normalized (0 mean 1 variance) scan"""
        
        scan = (scan - np.mean(scan))/(np.std(scan))
        return scan
    
    def clear_memory(self):
        # K.clear_session()
        return None

    def predict(self,x, ori_space_list=None, stride = 0.25):
        self.ori_space_list = ori_space_list #ori_space_list: 0.5, 0.741, 0.741
        #save shape for further upload
        original_shape = x.shape     #ct_scan.shape: (717,, 512, 512),
        
        #normalize input
        x = self._normalize(x)

        print('self.trgt_space_list', self.trgt_space_list)
        x1 = time.time()
        if self.trgt_space_list is not None and self.trgt_space_list[0] is not None:  #rescale scan to trgt spaces: 2.5, 1.4, 1.4
                print ('rescaled to new spacing  ')
                zoom_seq = np.array (self.ori_space_list, dtype='float') / np.array (self.trgt_space_list, dtype='float')
                print('zoom_seq', zoom_seq)
                x = ndimage.interpolation.zoom (x, zoom_seq, order=1, prefilter=1) # 143, 271, 271
                print('size after rescale to trgt spacing:', x.shape)

                # x = x[..., np.newaxis]
        elif self.trgt_sz:  #rescale scan to trgt sizes: 256,256,128
                print('rescaled to target size')
                x =  sample_scan(x[:,:,:,np.newaxis],self.trgt_sz,self.trgt_z_sz)
                x = x[..., 0]
                print('size after rescale to new size:', x.shape) # 64, 144, 144, 1
        else:
            print('please assign how to rescale')

        x2 = time.time()
        print('time for rescale:', x2 - x1)
        x3 = time.time()
        
        #let's patch this scan (despatchito)
        if(self.patching):
            print ('start patching')
            x_patch = deconstruct_patch(x,patch_shape=(self.z_sz,self.ptch_sz,self.ptch_sz), stride = stride)
            print('x_patch.shape', x_patch.shape) #(125, 64, 128, 128,1)
            x4 = time.time()
            print('time for deconstruct patch:', x4-x3)
        else:
            x_patch = x
            # x_patch = rescale_x


        if len (x_patch.shape)==4:
            if x_patch.shape[-1]==1:
                x_patch = x_patch[np.newaxis, ...]
            else:
                x_patch = x_patch[..., np.newaxis]
        elif len (x_patch.shape)==3:
            x_patch = x_patch[np.newaxis, ..., np.newaxis]
        else:
            raise Exception('shape of x_patch {} is not coeect'.format(x_patch.shape))

        # x_patch = x_patch
        #update shape to NN - > slice axis is the last in the network
        x_patch = np.rollaxis(x_patch,1,4) # 48, 144, 144, 80, 1
        
      
        #run predict
        x5 = time.time()
        pred_array = self.v.predict(x_patch,self.batch_size,verbose=0)
        x6 = time.time()
        print('time for prediction:', x6-x5)


        # chooses our output :P (0:main pred, 1:aux output, 2-3: deep superv)
        if len(pred_array)>1:
            pred = pred_array[0]  # (125, 128, 128, 64, 6)
        else:
            pred = pred_array
        
        #turn back to image shape
        pred = np.reshape(pred,(pred.shape[0],self.ptch_sz,self.ptch_sz,self.z_sz,-1))
        pred = np.rollaxis(pred,3,1) # (125, 64, 128, 128, 6)
        
        
        
        if(self.patching):
            x7 = time.time()
            pred = reconstruct_patch(pred, original_shape=x.shape,stride = stride)
            x8 = time.time()
            print('time for reconstruct patch:', x8-x7)

        

        
        #one hot decoding
        masks = []
        for p in pred:
            masks.append(one_hot_decoding(p,self.labels))
        masks=np.array(masks,dtype='uint8')

        if self.trgt_space_list is not None and self.trgt_space_list[0] is not None:
            print ('rescaled to original spacing  ')
            zoom_seq = np.array (self.trgt_space_list, dtype='float') / np.array (self.ori_space_list, dtype='float')
            print('zoom_seq', zoom_seq)
            final_pred = ndimage.interpolation.zoom (masks, zoom_seq, order=0, prefilter=False)
            # x = x[..., np.newaxis]
        elif self.trgt_sz:
            print('rescaled to original size', original_shape)
            # upsample back to original shape
            zoom_seq = np.array (original_shape, dtype='float') / np.array (masks.shape, dtype='float')
            print('zoom_seq', zoom_seq)
            final_pred = ndimage.interpolation.zoom (masks, zoom_seq, order=0, prefilter=False)
        else:
            final_pred = masks
            print('please assign the correct rescale method')

        print('after rescale, the shape is: ', final_pred.shape)
        if final_pred.shape[0] != original_shape[0]:

            nb_slice_lost = abs(original_shape[0] - final_pred.shape[0])

            if original_shape[0] > final_pred.shape[0]:
                print('there are {} slices lost along z axis, they will be repeated by the last slice'.format(nb_slice_lost))
                for i in range(nb_slice_lost):
                    added_slice = np.expand_dims(final_pred[-1], axis=0)
                    final_pred = np.concatenate((final_pred, added_slice))
                print('after repeating, the shape became: ', final_pred.shape)
            else:
                print('there are {} slices more along z axis, they will be cut'.format(nb_slice_lost))
                final_pred = final_pred[:original_shape[0]] # original shape: (649, 512, 512)
                print('after cutting, the shape became: ', final_pred.shape)


        if final_pred.shape[1] != original_shape[1]:

            nb_slice_lost = abs(original_shape[1] - final_pred.shape[1])

            if original_shape[1] > final_pred.shape[1]:
                print('there are {} slices lost along x,y axis, they will be repeated by the last slice'.format(nb_slice_lost))
                for i in range(nb_slice_lost):
                    added_slice = final_pred[:, -1, :]
                    added_slice = np.expand_dims(added_slice, axis=1)
                    print('x axis final_pred.shape', final_pred.shape)
                    print('x axis add_slice.shape', added_slice.shape)
                    final_pred = np.concatenate((final_pred, added_slice), axis=1)
                    print('after first repeating, the shape is: ', final_pred.shape)



                    added_slice = np.expand_dims(final_pred[:, :, -1], axis=2)
                    print('y axis final_pred.shape', final_pred.shape)
                    print('y axis add_slice.shape', added_slice.shape)
                    final_pred = np.concatenate((final_pred, added_slice), axis=2)
                print('after repeating, the shape became: ', final_pred.shape)
            else:
                print('there are {} slices more along x,y axis, they will be cut'.format(nb_slice_lost))
                final_pred = final_pred[:, :original_shape[1], :original_shape[1]]  # original shape: (649, 512, 512)
                print('after cutting, the shape became: ', final_pred.shape)


        print('final_pred.shape: ', final_pred.shape)
        return final_pred
        
#       