# CONFIG='configs/256x192_adam_lr1e_3_res34_smpl_3d_cam_2x_mix_w_pw3d.yaml'

CONFIG='configs/256x192_adam_lr1e_3_hrw48_cam_2x_w_pw3d_3dhp.yaml'

# CKPT='pretrained_model/pretrained_hr48.pth'
# 

EXPID='test_3dpw'
PORT=${3:-23458}
HOST=$(hostname -i)
CUDA_VISIBLE_DEVICES='3' python ./scripts/train_smpl_cam.py \
    --nThreads 8 \
    --launcher pytorch --rank 0 \
    --dist-url tcp://${HOST}:${PORT} \
    --exp-id ${EXPID} \
    --cfg ${CONFIG} --seed 123123
