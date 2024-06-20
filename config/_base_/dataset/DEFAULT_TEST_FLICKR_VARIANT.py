FLICKR_TEST_COMMON_CFG = dict(
    type='FlickrDataset',
    image_folder=r'./reaction_image',
    max_dynamic_size=None,
)

DEFAULT_TEST_FLICKR_VARIANT = dict(
    FLICKR_EVAL_with_box=dict(
        **FLICKR_TEST_COMMON_CFG,
        filename=r'./reactiondata/reaction_eval.jsonl',
        template_file=r'./config/_base_/dataset/template/reaction.json',
    ),
    # FLICKR_EVAL_without_box=dict(
    #     **FLICKR_TEST_COMMON_CFG,
    #     filename=r'{{fileDirname}}/../../../data/CWB_flickr30k_eval.jsonl',
    #     template_file=r'{{fileDirname}}/template/image_cap.json',
    #),
    FLICKR_TEST_with_box=dict(
        **FLICKR_TEST_COMMON_CFG,
        filename=r'./reactiondata/reaction_test.jsonl',
        template_file=r'./config/_base_/dataset/template/reaction.json',
    )
    # FLICKR_TEST_without_box=dict(
    #     **FLICKR_TEST_COMMON_CFG,
    #     filename=r'{{fileDirname}}/../../../data/CWB_flickr30k_test.jsonl',
    #     template_file=r'{{fileDirname}}/template/image_cap.json',
    # ),
)
