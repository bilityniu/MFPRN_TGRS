_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_3x_isdd_voc.py'

model = dict(
    neck=dict(
        _delete_=True,
        type='HRFPN',
        in_channels=[32, 64, 128, 256],
        out_channels=256))
