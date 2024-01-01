_base_ = './cascade_rcnn_r50_fpn_3x_voc.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
