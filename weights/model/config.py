model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='SwinTransformer3D',
        patch_size=(2, 4, 4),
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=(8, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True),
    test_cfg=dict(average_clips='prob', max_testing_views=2))

dataset_type = 'VideoDataset'

test_pipeline = [
    dict(type='ShiYouDecordInit'),
    dict(
        type='SampleFrames',
        clip_len=7,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='ShiYouDecordDecode'),
    dict(type='Resize', scale=(-1, 512)),
    dict(type='CenterCrop', crop_size=448),
    dict(type='Flip', flip_ratio=0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
