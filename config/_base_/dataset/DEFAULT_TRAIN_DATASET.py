_base_ = [
    # 'DEFAULT_TRAIN_GQA_VARIANT.py',
    # 'DEFAULT_TRAIN_CLEVR_VARIANT.py',
    # 'DEFAULT_TRAIN_POINT_VARIANT.py',
    # 'DEFAULT_TRAIN_GPTGEN_VARIANT.py',
    # 'DEFAULT_TRAIN_VCR_VARIANT.py',
    # 'DEFAULT_TRAIN_VQAv2_VARIANT.py',
    # 'DEFAULT_TRAIN_VQAEX_VARIANT.py',
]

DEFAULT_TRAIN_DATASET = dict(
    flickr=dict(
        type='FlickrDataset',
        filename=r'./reactiondata/reaction_train.jsonl',
        image_folder=r'./reaction_image',
        template_file=r'./config/_base_/dataset/template/reaction.json',
    ),
    # **_base_.DEFAULT_TRAIN_GQA_VARIANT,
    # **_base_.DEFAULT_TRAIN_CLEVR_VARIANT,
    # **_base_.DEFAULT_TRAIN_POINT_VARIANT,
    # **_base_.DEFAULT_TRAIN_GPTGEN_VARIANT,
    # **_base_.DEFAULT_TRAIN_VCR_VARIANT,
    # **_base_.DEFAULT_TRAIN_VQAv2_VARIANT,
    # **_base_.DEFAULT_TRAIN_VQAEX_VARIANT,
)
