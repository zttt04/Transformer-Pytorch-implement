这是第一次复现论文，作为入门深度学习的第一个小项目，中间过程也非常曲折，从一开始2天写完框架，欣喜若狂，到后面不断的修改数据处理与训练代码，再叠加开学等因素，断断续续将近一个月才完成项目。
下面简单介绍一下项目组成：
ATT3
├── DataProcessing                  # 数据预处理相关
│   ├── Test                        # 测试数据集
│   │   ├── de_data_test.parquet    # 德语测试数据
│   │   ├── en_data_test.parquet    # 英语测试数据
│   │   ├── test.parquet            # 整理后的数据（未使用）
│   │   └── test1.parquet           # 官网原始数据
│   ├── Tokenizer                   # 分词器相关
│   │   ├── wmt19-en-de             # 手动下载的预训练分词器
│   │   └── Tokenizer.py            # 分词逻辑
│   ├── Train                       # 训练数据集
│   │   ├── combined_train.parquet  # 合并后的训练数据
│   │   ├── de_data.parquet         # 德语训练数据
│   │   ├── en_data.parquet         # 英语训练数据
│   │   ├── train1.parquet          # 官网原始数据（分块）
│   │   ├── train2.parquet
│   │   └── train3.parquet
│   └── Val                         # 验证数据集（未使用）
│       ├── de_data_val.parquet     # 德语验证数据
│       ├── en_data_val.parquet     # 英语验证数据
│       ├── val.parquet             # 整理后的数据（未使用）
│       └── validation1.parquet     # 官网原始数据
├── Processing.py                   # 数据处理脚本
├── fig                             # 训练结果可视化
│   └── transformer_loss_final1.png # 最终损失曲线
├── logs                            # 训练日志
│   └── epoch_training_logs_final1.csv # 训练过程记录
├── Weight                          # 模型权重
│   └── transformer_epoch_10_final1.pth # 第10轮权重
├── config.py                       # 超参数配置
├── mask.py                         # 掩码生成逻辑
├── Readme.txt                      # 当前文档（持续更新）
├── test.py                         # 测试脚本
├── train.py                        # 训练脚本
└── Transformer_model.py            # Transformer 模型实现
原文的bleu是25.8 我自己单卡bs64 epoch10 训练时间30h 最后22。但根据每个epoch的bleu趋势来看多训练几轮应该24可以达到（有空再试试，希望不要打脸ww）
因为自己尝试复现的时候发现很多大佬的代码封装的都特别的好，确实可以开箱即用，但对于小白来说却很难从中学习，所以这个项目我设计的结构比较简单清楚，上面也对每一个部分都做了解释；
同时我还专门让ai帮我给代码加了注释，希望可以降低阅读学习门槛，帮助大家学有所获！
