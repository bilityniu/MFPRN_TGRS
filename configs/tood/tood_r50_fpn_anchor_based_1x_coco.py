_base_ = './tood_r50_fpn_2x_coco.py'
model = dict(bbox_head=dict(anchor_type='anchor_based'))
