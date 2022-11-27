#hr48
CONFIG='./configs/256x192_adam_lr1e_3_hrw48_cam_2x_w_pw3d_3dhp.yaml '
# CKPT='pretrained_model/pretrained_hr48.pth'
CKPT='pretrained_model/hybrik_hrnet48_w3dpw.pth'

# w_cam论文版本 res34
# CONFIG='./configs/256x192_adam_lr1e_3_res34_smpl_3d_cam_2x_mix_w_3dpw.yaml '
# CKPT='pretrained_model/pretrained_w_cam.pth'




PORT=${3:-23456}

HOST=$(hostname -i)
python ./scripts/validate_smpl_cam.py \
    --batch 8 \
    --gpus 0,1,2,3 \
    --world-size 1 \
    --flip-test \
    --launcher pytorch --rank 0 \
    --dist-url tcp://${HOST}:${PORT} \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT}
