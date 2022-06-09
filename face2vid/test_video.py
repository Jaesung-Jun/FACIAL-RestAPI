

from .options.test_options import TestOptions

from .data.data_loader import CreateDataLoader
from .models.models import create_model
from .util.util import tensor2im

import torch
import glob
import os

from skimage.io import imsave
from constants import Constants
from logger import logger
import cv2
from cv2 import VideoWriter_fourcc

#from cv2 import VideoWriter_fourcc,imread,resize, VideoWriter
#from collections import OrderedDict
#from .data.custom_dataset_data_loader import CreateDataset
#from imageio import get_writer
#import numpy as np
#from tqdm import tqdm
#from os.path import join, exists, abspath, dirname
#import ffmpeg

def test_video(uid, selected_model_str):

    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    opt.test_id_name = "test1"

    opt.blink_path = f'{Constants.DEFAULT_USER_DIRECTORY}/{uid}/test-result/{uid}.npz'
    opt.name = ""
    opt.model = "pose2vid"
    opt.dataroot = f"{Constants.DEFAULT_USER_DIRECTORY}/{uid}/rendering"
    opt.which_epoch = "latest" 
    opt.netG = "local" 
    opt.ngf = 32 
    opt.label_nc = 0 
    opt.n_local_enhancers = 1 
    opt.no_instance = True
    opt.resize_or_crop = "resize"
    opt.isTrain = False
    opt.load_pretrain = f"{Constants.DEFAULT_NEED_FILES}/{selected_model_str}/checkpoints/"

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    #dataset_size = len(data_loader)
    #print('#test images = %d' % dataset_size)
    model = create_model(opt).cuda()
    logger(f"Model loaded in GPU #{torch.cuda.current_device()}. | Request from {uid} | Processing {selected_model_str}", level="INFO")
    
    #if opt.verbose:
    #    print(model)
    #img_root = f'{Constants.DEFAULT_NEED_FILES}/{selected_model_str}/test_images/'
    img_root = f'{Constants.DEFAULT_USER_DIRECTORY}/{uid}/test_images'
    
    if not os.path.exists(img_root):
        os.makedirs(img_root)

    logger(f"Model inference started. It will take time little bit long.. | Request from {uid} | Processing {selected_model_str}", level="EXE_START")
    #logger(f" | Request from {uid} | Processing {selected_model_str}", level="WARNING")
    
    for i, data in enumerate(dataset):
        label = data['label']
        
        #cur_frame = model.inference(label)
        cur_frame = model.inference(label)
        prev_frame = cur_frame.data[0]

        if i+7<= len(dataset):
            frameindex = (i+7)
        else:
            frameindex = i + 7 - len(dataset)
            
        imsave(img_root+'/{:06d}.jpg'.format(frameindex), tensor2im(prev_frame))

    logger(f"Model inference successfully finished. | Request from {uid} | Processing {selected_model_str}", level="EXE_FINISH")
    fps = 30
    fourcc=VideoWriter_fourcc('M','J','P','G')
    videoWriter = cv2.VideoWriter(f'{Constants.DEFAULT_USER_DIRECTORY}/{uid}/complete_{uid}.avi',fourcc,fps,(720,720))
    
    im_names=os.listdir(img_root)
    im_names = sorted(glob.glob(os.path.join(img_root, '*[0-9]*.jpg')))

    logger(f"Video writting started. | Request from {uid} | Processing {selected_model_str}", level="EXE_START")

    for im_name in range(len(im_names)):
        frame=cv2.imread(im_names[im_name])
        frame = cv2.resize(frame, (720,720)) 
        #print (im_name)
        videoWriter.write(frame)
    #print(videoWriter)

    torch.cuda.empty_cache()
    
    logger(f"Video writting successfully finished | Request from {uid} | Processing {selected_model_str}", level="EXE_FINISH")

    videoWriter.release()

    