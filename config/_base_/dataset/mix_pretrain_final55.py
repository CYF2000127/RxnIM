_base_ = ['DEFAULT_TRAIN_DATASET.py']

data_args = dict(
    train=dict(
        type='ConcatDatasetWithShuffle',  # 使用 ConcatDatasetWithShuffle，即使只有一个数据集
        cfgs=[
            {{_base_.DEFAULT_TRAIN_DATASET.flickr}}  # 使用 flickr 数据集的配置
        ]
    ),
    validation=None,  # 如果没有验证集，保持为 None
    test=None,  # 如果没有测试集，保持为 None

    # compute_metric 设置
    compute_metric=None,

    # padding collator 参数
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # generate 配置
    gen_kwargs=dict(
        max_new_tokens=1024,
        num_beams=1,
    ),
)
