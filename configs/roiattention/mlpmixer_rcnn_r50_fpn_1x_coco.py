_base_ = [
    '../_base_/datasets/utdac_detection_coco.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        # reg_decoded_bbox=True,
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        # loss_cls=dict(
        #     type='FocalLoss',
        #     use_sigmoid=True,
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=1.0),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        # init_cfg=dict(
        #          type='Normal',
        #          layer='Conv2d',
        #          std=0.01,
        #          override=dict(
        #              type='Normal',
        #              name='rpn_cls',
        #              std=0.01,
        #              bias_prob=0.01))
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='MLPBBoxHead',
            with_avg_pool=True,
            num_classes=4,
            inchanel=256,
            DS=256,
            mlp_dim=512,
            depth=2,
            num_convs=4,
            # reg_decoded_bbox=True,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            # loss_cls=dict(
            #     type='FocalLoss',
            #     use_sigmoid=True,
            #     gamma=2.0,
            #     alpha=0.25,
            #     loss_weight=1.0),
            # loss_bbox=dict(type='GIoULoss', loss_weight=1.3),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            # loss_bbox=dict(type='CIoULoss', loss_weight=12.0)
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        )),
        train_cfg=dict(
            rpn=dict(
                # assigner=dict(
                #     type='MaxIoUAssigner',
                #     pos_iou_thr=0.5,
                #     neg_iou_thr=0.4,
                #     min_pos_iou=0,
                #     match_low_quality=True,
                #     ignore_iof_thr=-1),
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                # sampler=dict(
                #     _delete_=True,
                #     type='CombinedSampler',
                #     num=512,
                #     pos_fraction=0.25,
                #     add_gt_as_proposals=True,
                #     pos_sampler=dict(type='InstanceBalancedPosSampler'),
                #     neg_sampler=dict(
                #         type='IoUBalancedNegSampler',
                #         floor_thr=-1,
                #         floor_fraction=0,
                #         num_bins=3)),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.00,
                nms=dict(type='soft_nms', iou_threshold=0.5),
                # nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100)
        ))
# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
