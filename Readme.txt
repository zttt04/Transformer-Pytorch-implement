这是第一次复现论文，作为入门深度学习的第一个小项目，中间过程也非常曲折，从一开始2天写完框架，欣喜若狂，到后面不断的修改数据处理与训练代码，再叠加开学等因素，断断续续将近一个月才完成项目。
下面简单介绍一下项目组成：
ATT3                                     ##哈哈3意味着这是我的第三版
├── DataProcessing                       ##这个文件夹储存了预训练的tokenizer，数据集，以及处理数据的脚本
│   ├── Test                             ##测试数据集
│   │   ├── de_data_test.parquet         ##德语数据 
│   │   ├── en_data_test.parquet         ##英文数据
│   │   ├── test.parquet                 ##整理后的数据（没啥用）
│   │   └── test1.parquet                ##从官网下载的源数据
│   ├── Tokenizer                        ##储存tokenizer以及做分词
│   │   ├── wmt19-en-de                  ##从hf下载的比较好的预训练tokenizer，因为我网络不行，所以是手动下载的
│   │   └── Tokenizer.py                 ##分词操作
│   ├── Train                            ##训练集
│   │   ├── combined_train.parquet       ##由于官网的数据太大了是分成了三部分，所以我这里把三个整合起来
│   │   ├── de_data.parquet              ##德语数据
│   │   ├── en_data.parquet              ##英语数据
│   │   ├── train1.parquet               ##以下三个都是官网源数据
│   │   ├── train2.parquet          
│   │   └── train3.parquet          
│   └── Val                              ##验证集（但是我没有用hh）
│       ├── de_data_val.parquet          ##德语数据
│       ├── en_data_val.parquet          ##英语数据
│       ├── val.parquet                  ##处理后的数据（好像没用）
│       └── validation1.parquet          ##官网源数据
├── Processing.py                        ##数据处理脚本
├── fig                                  ##储存训练结果loss，其他没用的都删掉了
│   └── transformer_loss_final1.png  
├── logs                                 ##记录训练过程
│   └── epoch_training_logs_final1.csv  
├── Weight                               ##存储模型权重
│   └── transformer_epoch_10_final1.pth  
├── config.py                            ##超参数
├── mask.py                              ##制作掩码
├── Readme.txt                           ##正在写
├── test.py                              ##测试脚本
├── train.py                             ##训练脚本
└── Transformer_model.py                 ##模型框架
原文的bleu是25.8 我自己单卡bs64 epoch10 训练时间30h 最后22。但根据每个epoch的bleu趋势来看多训练几轮应该24可以达到（有空再试试，希望不要打脸ww）
因为自己尝试复现的时候发现很多大佬的代码封装的都特别的好，确实可以开箱即用，但对于小白来说却很难从中学习，所以这个项目我设计的结构比较简单清楚，上面也对每一个部分都做了解释；
同时我还专门让ai帮我给代码加了注释，希望可以降低阅读学习门槛，帮助大家学有所获！