
import argparse
import json
import os.path as osp
import os
import subprocess
import cv2

def video_to_images(vid_file, img_folder=None, return_info=False):

    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')

    try:
        subprocess.call(command)
    except:
        subprocess.call(f'{" ".join(command)}', shell=True)

    print(f'Images saved to \"{img_folder}\"')

    img_shape = cv2.imread(osp.join(img_folder, '000001.png')).shape

    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder


def images_to_video(img_folder, output_vid_file, re=''):
    input = f'{img_folder}/%06d.png'
    if re:
      input = f'{img_folder}/{re}'
    command = [
        'ffmpeg', '-y', '-threads', '16', '-i', input, '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-v', 'error',
        '-pix_fmt', 'yuv420p', '-an', output_vid_file,
    ]

    print(f'Running \"{" ".join(command)}\"')
    try:
        subprocess.call(command)
    except:
        subprocess.call(f'{" ".join(command)}', shell=True)
  

if __name__ == '__main__':
  # 函数来自于pymaf
  parser = argparse.ArgumentParser()
  parser.add_argument('-m',  help='model', default='v', type=str)
  parser.add_argument('-i',  help='input', type=str)
  parser.add_argument('-o',  help='output', default='', type=str)
  parser.add_argument('--re',  help='re', default='',type=str)

  cfg = parser.parse_args()
  if cfg.o == '':
    if cfg.m == 'v':
      cfg.o = os.path.join(cfg.i, '../images')
    else:
      cfg.o = os.path.join(cfg.i, '../output.mp4')
  print('input: ' + cfg.i)
  print('output: ' + cfg.o)

  if cfg.m == 'v':
    video_to_images(vid_file=cfg.i, img_folder=cfg.o)
  elif cfg.m == 'i':
    images_to_video(img_folder=cfg.i, output_vid_file=cfg.o, re=cfg.re)
  else:
    print('invalid mode: {}'.format(cfg.m))

