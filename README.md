# HDTF
Flow-guided One-shot Talking Face Generation with a High-resolution Audio-visual Dataset 
<a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Flow-Guided_One-Shot_Talking_Face_Generation_With_a_High-Resolution_Audio-Visual_Dataset_CVPR_2021_paper.pdf" target="_blank">paper</a> 

## Details of HDTF dataset
**./HDTF_dataset** consists of *youtube video url*, *time stamps of talking face* and *facial region* in the video.
**xx_video_url.txt:** 


```
format:     video name | video youtube url
```

**xx_annotion_time.txt:**
```
format:    video name | time stamps of clip1 | time stamps of clip2 | time stamps of clip3....
```
**xx_crop_wh.txt:**
```
format:    video name+clip index | min_width | width |  min_height | height
```
## Processing of HDTF dataset
When using HDTF dataset, 

 1. We provide video and url in  **xx_video_url.txt**. (the highest definition of videos are 1080P or 720P).  Transform video into **.mp4** format and transform interlaced video to progressive video as well.

2. We split long original video into talking head clips with time stamps in **xx_annotion_time.txt**.  Name the splitted clip as **video name_clip index.mp4**. For example, split the video  *Radio11.mp4 00:30-01:00 01:30-02:30*  into *Radio11_0.mp4* and *Radio11_1.mp4* .

3. We crop the facial region with fixed window size in **xx_crop_wh.txt** and resize the video into **512 x 512** resolution.


The HDTF dataset is available to download under a <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank"> Creative Commons Attribution 4.0 International License</a>.

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
