# Code of animation-to-video module 
### inference

 1. Download the trained model (`checkpoint_animation2video.pth`), approximate dense flow (`mengnalisa_Fapp.npy, taile_Fapp.npy`) in [google drive](https://drive.google.com/drive/folders/1OQ87Pp42-NYcNhPc41kW0UHw0Z_ll84h?usp=sharing).
 2. Put the `checkpoint_animation2video.pth` into **./checkpoints**
 3. Put the `mengnalisa_Fapp.npy, taile_Fapp.npy` into **./test_data**
 4. run 
> inference --image_path=./test_data/mengnalisa.jpg --dense_flow_path=./test_data/mengnalisa_Fapp.npy

or
> inference --image_path=./test_data/taile.jpg --dense_flow_path=./test_data/taile_Fapp.npy

to generate all intermediate results of animation-to-video module.