# Center-based 3D Object Detection and Tracking TRT transfer

## step 1
docker pull zhq199139/centerpoint:v1

## step 2
CUDA_VISIBLE_DEVICES=0 python tools/single_test.py configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_dcn_flip.py --work_dir /workspace/workdir --checkpoint /workspace/model/pytorch/original/voxelnet_converted.pth --testset --speed_test
