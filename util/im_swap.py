import os 
import cv2
import glob
import torch
import shutil
import numpy as np
from tqdm import tqdm
from util.reverse2original import reverse2wholeimage
import moviepy.editor as mp
from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import  time
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


def swap_image(video_path, id_vetor, swap_model, detect_model, save_path, temp_results_dir='./temp_results', crop_size=224, no_simswaplogo = False,use_mask =False):
    
    
    
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    ret = True
    
    spNorm =SpecificNorm()
    
    if use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net =None


    frame = cv2.imread(video_path)
    
    detect_results = detect_model.get(frame,crop_size)

    if detect_results is not None:
        # print(frame_index)
        if not os.path.exists(temp_results_dir):
                os.mkdir(temp_results_dir)
        frame_align_crop_list = detect_results[0]
        frame_mat_list = detect_results[1]
        swap_result_list = []
        frame_align_crop_tenor_list = []
        for frame_align_crop in frame_align_crop_list:

            # BGR TO RGB
            # frame_align_crop_RGB = frame_align_crop[...,::-1]

            frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

            swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
            swap_result_list.append(swap_result)
            frame_align_crop_tenor_list.append(frame_align_crop_tenor)

        reverse2wholeimage(frame_align_crop_tenor_list,swap_result_list, frame_mat_list, crop_size, frame, logoclass,\
            os.path.join(temp_results_dir, 'output.jpg'),no_simswaplogo,pasring_model =net,use_mask=use_mask, norm = spNorm)
      
    print("Done!!!")