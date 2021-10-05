

# HDTF
Flow-guided One-shot Talking Face Generation with a High-resolution Audio-visual Dataset 
<a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Flow-Guided_One-Shot_Talking_Face_Generation_With_a_High-Resolution_Audio-Visual_Dataset_CVPR_2021_paper.pdf" target="_blank">paper</a>    <a href="https://github.com/MRzzm/HDTF/blob/main/Supplementary%20Materials.pdf" target="_blank">supplementary</a>   [demo video](https://www.youtube.com/watch?v=uJdBgWYBTww)

## Details of HDTF dataset
**./HDTF_dataset** consists of *youtube video url*, *video resolution* (in our method, may not be the best resolution), *time stamps of talking face*, *facial region* (in the our method) and *the zoom scale* of the cropped window.

**xx_video_url.txt:** 


```
format:     video name | video youtube url
```
**xx_resolution.txt:**
```
format:    video name | resolution(in our method)
```

**xx_annotion_time.txt:**
```
format:    video name | time stamps of clip1 | time stamps of clip2 | time stamps of clip3....
```
**xx_crop_wh.txt:**
```
format:    video name+clip index | min_width | width |  min_height | height (in our method)
```
**xx_crop_ratio.txt:**
```
format:    video name+clip index | window zoom scale
```


## Processing of HDTF dataset
When using HDTF dataset, 

 - We provide video and url in  **xx_video_url.txt**. (the highest definition of videos are 1080P or 720P).  Transform video into **.mp4** format and transform interlaced video to progressive video as well.

 - We split long original video into talking head clips with time stamps in **xx_annotion_time.txt**.  Name the splitted clip as **video name_clip index.mp4**. For example, split the video  *Radio11.mp4 00:30-01:00 01:30-02:30*  into *Radio11_0.mp4* and *Radio11_1.mp4* .

 - Our work does not always download videos with the best resolution, so we provide two cropping methods. Thanks @universome and @Feii Yin for pointing out this problem! 

	1. Download the video with reference resulotion in **xx_resolution.txt** and crop the facial region with fixed window size in **xx_crop_wh.txt**. (This method is as same as ours, but the downloaded video may not be the best resolution).
	2. First, download the video with best resulotion. Then, detect the facial landmark in the splitted talking head clips and count the square window of the face, specifically, count the facial region in each frame and merge all regions into one square range. Next,  enlarge the window size with **xx_crop_ratio.txt**. Finally, crop the facial region. 

- We resize all cropped videos into **512 x 512** resolution.


The HDTF dataset is available to download under a <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank"> Creative Commons Attribution 4.0 International License</a>. **Thanks @universome for provding the the script of data processing, pls visit [here](https://github.com/universome/HDTF) for more details.** If you face any problems when processing HDTF, pls contact me.

## Inference code
#### code of audio-to-animation
coming soon......

#### code of constructing approximate dense flow
The code is in **./code_constructing_Fapp**, pls visit [here](https://github.com/MRzzm/HDTF/tree/main/code_constructing_Fapp) for more details. 

#### code of animation-to-video module
The code is in **./code_animation2video**, pls visit [here](https://github.com/MRzzm/HDTF/tree/main/code_animation2video) for more details. 

#### code of reproducing other works
coming soon......

## Reference
if you use HDTF, pls reference

```
@inproceedings{zhang2021flow,
  title={Flow-Guided One-Shot Talking Face Generation With a High-Resolution Audio-Visual Dataset},
  author={Zhang, Zhimeng and Li, Lincheng and Ding, Yu and Fan, Changjie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3661--3670},
  year={2021}
}
```
