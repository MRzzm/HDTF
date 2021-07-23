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
If you use HDTF dataset, pls

 1. Download videos from  **xx_video_url.txt** with <a href="https://github.com/soimort/you-get" target="_blank">you-get</a>  tool or <a href="https://github.com/ytdl-org/youtube-dl" target="_blank">youtube-dl</a> tool. (pls download the highest definition version: 1080P or 720P).  Transform video into **.mp4** format. You'd better transform interlaced video to porgressive video as well.

2. Split original long video into appropriate talking head clips with time stamps in **xx_annotion_time.txt**.  Name the splitted clip as **video name_clip index.mp4**. For example, split the video  *Radio11.mp4 00:30-01:00 01:30-02:30*  into *Radio11_0.mp4* and *Radio11_1.mp4* .

3. crop the facial region with fixed window size in **xx_crop_wh.txt** and resize the video into **512 x 512** resolution.




## Inference Code
coming soon......
