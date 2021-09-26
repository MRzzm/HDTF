import numpy as np
from models import VideoGenerator
from opts import parse_opts
from torchvision import transforms
import os
from math import log10
import torch
import numpy as np
import json
import random
import cv2
import subprocess
from utils import visualize_dense_flow,make_coordinate_grid
if __name__ == "__main__":
    opt = parse_opts()
    ### load trained model
    video_generator = VideoGenerator(opt.input_channel, opt.encoder_num_down_blocks, opt.encoder_block_expansion,
                                opt.encoder_max_features, opt.houglass_num_blocks,
                                opt.houglass_block_expansion, opt.houglass_max_features,
                                opt.num_bottleneck_blocks).cuda()
    old_dict = torch.load(opt.model_path)['state_dict']['net_g']
    new_dict = {}
    for k, v in old_dict.items():
        name = k[7:].replace('dense_motion','foreground_matting').replace('attention_mask','matting_mask')
        new_dict[name] = v
    video_generator.load_state_dict(new_dict)
    video_generator.eval()
    #### load reference image
    reference_image = cv2.imread(opt.image_path)
    reference_tensor = torch.from_numpy(reference_image / 255).permute(2, 0, 1).float().unsqueeze(0).cuda()
    #### load approximate dense flow
    Fapp = np.load(opt.dense_flow_path)
    tem = Fapp[0,:,:,0]
    #### output setting
    ## generated video
    synthetic_video_path = os.path.join(opt.res_path,os.path.basename(opt.image_path)[:-4] + '_video.mp4')
    if os.path.exists(synthetic_video_path):
        os.remove(synthetic_video_path)
    videowriter_synthetic_video = cv2.VideoWriter(synthetic_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (512, 512))
    ## app dense flow
    Fapp_video_path = synthetic_video_path.replace('video.mp4','Fapp.mp4')
    if os.path.exists(Fapp_video_path):
        os.remove(Fapp_video_path)
    videowriter_Fapp = cv2.VideoWriter(Fapp_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (512, 512))
    ## matting mask
    matting_mask_path = synthetic_video_path.replace('video.mp4','matting_mask.mp4')
    if os.path.exists(matting_mask_path):
        os.remove(matting_mask_path)
    videowriter_matting_mask = cv2.VideoWriter(matting_mask_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (128, 128))
    ## revised dense flow
    revised_dense_path = synthetic_video_path.replace('video.mp4', 'revised_dense.mp4')
    if os.path.exists(revised_dense_path):
        os.remove(revised_dense_path)
    videowriter_revised_dense = cv2.VideoWriter(revised_dense_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (128, 128))
    ## foreground mask
    foreground_mask_path = synthetic_video_path.replace('video.mp4', 'foreground_mask.mp4')
    if os.path.exists(foreground_mask_path):
        os.remove(foreground_mask_path)
    videowriter_foreground_mask = cv2.VideoWriter(foreground_mask_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (128, 128))
    ## warped image
    warped_image_path = synthetic_video_path.replace('video.mp4', 'warped_image.mp4')
    if os.path.exists(warped_image_path):
        os.remove(warped_image_path)
    videowriter_warped_image = cv2.VideoWriter(warped_image_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (128, 128))
    #########  generate video frame by frame
    frame_length = Fapp.shape[0]
    for i in range(frame_length):
        print('generating frame {}/{} '.format(i, frame_length))
        Fapp_i = Fapp[i,:,:]
        Fapp_i_visual = visualize_dense_flow(Fapp_i - make_coordinate_grid((Fapp_i.shape[0],Fapp_i.shape[1])))
        videowriter_Fapp.write(Fapp_i_visual)
        with torch.no_grad():
            Fapp_i = torch.from_numpy(Fapp_i).float().cuda().unsqueeze(0)
            res_out = video_generator(reference_tensor, Fapp_i)
            ## synthetic_video
            synthetic_video_i = res_out['synthetic_image'] * 255
            synthetic_video_i = synthetic_video_i.cpu().squeeze().permute(1, 2, 0).float().detach().numpy().astype(np.uint8)
            videowriter_synthetic_video.write(synthetic_video_i)
            ## warped image
            warped_image_i = res_out['warped_image']* 255
            warped_image_i = warped_image_i.cpu().squeeze().permute(1, 2, 0).float().detach().numpy().astype(np.uint8)
            videowriter_warped_image.write(warped_image_i)
            ## foreground_mask
            foreground_mask_i = res_out['foreground_mask']* 255
            foreground_mask_i =np.expand_dims(foreground_mask_i.cpu().squeeze().detach().numpy().astype(np.uint8),2)
            foreground_mask_i = foreground_mask_i.repeat(3,2)
            videowriter_foreground_mask.write(foreground_mask_i)
            ## dense_flow_foreground_vis
            dense_flow_foreground_vis_i = foreground_mask_i/255.0 * cv2.resize(Fapp_i_visual,(128,128))
            dense_flow_foreground_vis_i = dense_flow_foreground_vis_i.astype(np.uint8)
            videowriter_revised_dense.write(dense_flow_foreground_vis_i)
            ### matting_mask
            matting_mask_i = res_out['matting_mask'] * 255
            matting_mask_i = matting_mask_i.cpu().squeeze().detach().numpy().astype(np.uint8)
            matting_mask_i = np.expand_dims(matting_mask_i,2).repeat(3,2)
            videowriter_matting_mask.write(matting_mask_i)

    videowriter_Fapp.release()
    videowriter_synthetic_video.release()
    videowriter_warped_image.release()
    videowriter_foreground_mask.release()
    videowriter_revised_dense.release()
    videowriter_matting_mask.release()

    if os.path.exists(opt.audio_path):
        video_add_audio_path = synthetic_video_path.replace('.mp4', '_add_audio.mp4')
        if os.path.exists(video_add_audio_path):
            os.remove(video_add_audio_path)
        cmd = 'ffmpeg -i {} -i {} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {}'.format(
            synthetic_video_path,
            opt.audio_path,
            video_add_audio_path)
        subprocess.call(cmd, shell=True)


