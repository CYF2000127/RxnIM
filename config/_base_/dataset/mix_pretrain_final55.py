_base_ = ['DEFAULT_TRAIN_DATASET.py']

data_args = dict(
    train=dict(
        type='ConcatDatasetWithShuffle', 
        cfgs=[
            {{_base_.DEFAULT_TRAIN_DATASET.flickr}}  
        ]
    ),
    validation=None,  
    test=None,  

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
