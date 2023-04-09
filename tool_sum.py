from moviepy.editor import *

video_name = 'flashmob'
name = video_name + '/' + video_name
video1 = VideoFileClip("demo_sum/" + name +  "_mps.mp4")  # [960, 540]
video2 = VideoFileClip("demo_sum/" + name + "_bev.mp4")
video3 = VideoFileClip("demo_sum/" + name + "_pymaf.mp4")
video4 = VideoFileClip("demo_sum/" + name + "_ik.mp4")
print('video1 size: ', video1.size)
print('video2 size: ', video1.size)
print('video3 size: ', video1.size)
print('video4 size: ', video1.size)

video = clips_array([[video1, video2], 
                     [video3, video4]])  # VideoFileClip [ [video_1, video_2] ]一行4列 (3840, 540)， [ [video_1], [video_2] ] [1920，1080]
print('video size: ', video.size)
# video = video.resize(width=video.size[0] / 2, height=video.size[1] / 2)  # resize  因为是一排两列，所以只需要将宽缩小成一半
video = video.resize(width=1920, height=1080)
print('video_sum_resize size: ', video.size)
video.write_videofile("demo_sum/sum.mp4")


# 上面的是将单个视频合成一个视频
# 下面的是给视频加标题

video = VideoFileClip("demo_sum/sum.mp4")
print('video size: ', video.size)

title_duration = video.duration
title_font_size = 40
title_font = '/usr/share/fonts/truetype/ubuntu/Ubuntu-L.ttf'
width, height = video.size
title1 = TextClip("MPS",fontsize=title_font_size, color='red', font=title_font).set_position((width/4*1,0)).set_duration(title_duration)# mix_audio = CompositeAudioClip([video_1.audio, video_2.audio, video_3.audio, video_4.audio])
title2 = TextClip("BEV",fontsize=title_font_size, color='red', font=title_font).set_position((width/4*3,0)).set_duration(title_duration)# mix_audio = CompositeAudioClip([video_1.audio, video_2.audio, video_3.audio, video_4.audio])
title3 = TextClip("PyMAF-X",fontsize=title_font_size, color='red', font=title_font).set_position((width/4, height/2)).set_duration(title_duration)# mix_audio = CompositeAudioClip([video_1.audio, video_2.audio, video_3.audio, video_4.audio])
title4 = TextClip("Our",fontsize=title_font_size, color='red', font=title_font).set_position((width/4*3,height/2)).set_duration(title_duration)# mix_audio = CompositeAudioClip([video_1.audio, video_2.audio, video_3.audio, video_4.audio])

video = CompositeVideoClip([video, title1, title2, title3, title4])
video.write_videofile("demo_sum/" + video_name + "_sum_title.mp4")


# video1 = VideoFileClip("demo_sum/action/1_out.mp4")  # [960, 540]
# video2 = VideoFileClip("demo_sum/action/2_out.mp4")
# video3 = VideoFileClip("demo_sum/action/3_out.mp4")
# video4 = VideoFileClip("demo_sum/action/4_out.mp4")
# print('video1 size: ', video1.size)
# print('video2 size: ', video1.size)
# print('video3 size: ', video1.size)
# print('video4 size: ', video1.size)

# video = concatenate_videoclips([video1, video2, 
#                      video3, video4])  # VideoFileClip [ [video_1, video_2] ]一行4列 (3840, 540)， [ [video_1], [video_2] ] [1920，1080]
# print('video size: ', video.size)
# # video = video.resize(width=video.size[0] / 2, height=video.size[1] / 2)  # resize  因为是一排两列，所以只需要将宽缩小成一半
# video = video.resize(width=1920, height=1080)
# print('video_sum_resize size: ', video.size)
# video.write_videofile("demo_sum/sum.mp4")


'''
pip install moviepy 
会报错 convert-im6.q16: attempt to perform an operation not allowed by the security policy `@/tmp/tmp6miuy_6o.txt' @ error/property.c/InterpretImageProperties/3518.
sudo vim  /etc/ImageMagick-6/policy.xml
注释掉下面这句话，因为下面这句话是没有权限访问 @/tmp/tmp6miuy_6o.txt
<!-- <policy domain="path" rights="none" pattern="@*"/> -->


'''



