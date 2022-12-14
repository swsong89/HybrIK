import cv2
from hybrik.utils.transforms import torch_to_im, im_to_torch
import numpy as np 
def img_to_torch_std(img):  # HWC
  img = im_to_torch(img)  # CHW
  # mean
  img[0].add_(-0.406)
  img[1].add_(-0.457)
  img[2].add_(-0.480)

  # std
  img[0].div_(0.225)
  img[1].div_(0.224)
  img[2].div_(0.229)
  return img

def torch_std_to_img(img):
    # std
  img[0].mul_(0.225)
  img[1].mul_(0.224)
  img[2].mul_(0.229)

  # mean
  img[0].sub_(-0.406)
  img[1].sub_(-0.457)
  img[2].sub_(-0.480)



  img = torch_to_im(img)
  # img = img*255
  img.astype(np.uint8)
  return img

input_image = cv2.cvtColor(cv2.imread('demo/crowd1.jpg'), cv2.COLOR_BGR2RGB)
cv2.imshow('input_image', input_image)
test_img = input_image/255
test_img = test_img*255
test_img = cv2.cvtColor(test_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imshow('test_img', test_img)


img = input_image.copy()
out = img_to_torch_std(img)
torch_std = torch_std_to_img(out)
output_img = torch_std*255
image_vis = cv2.cvtColor(torch_std, cv2.COLOR_RGB2BGR)
cv2.imshow('torch_std', image_vis)
output_img_vis = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
cv2.imshow('output_img_vis', output_img_vis)
key = cv2.waitKey(0)  # 等待按键命令, 1000ms 后自动关闭
cv2.destroyAllWindows()


