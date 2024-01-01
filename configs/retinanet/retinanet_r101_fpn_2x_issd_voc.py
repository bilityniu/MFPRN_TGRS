_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/isdd_detection_coco.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
