import pyarrow.parquet as pq
import torch
from transformers import FSMTTokenizer
from tqdm import tqdm


mname = './wmt19-en-de'
tokenizer = FSMTTokenizer.from_pretrained(mname)

en_data = pq.read_table("../Train/en_data_train.parquet")
de_data = pq.read_table("../Train/de_data_train.parquet")

if 'en' in en_data.column_names and 'de' in de_data.column_names:
    en_sentences = en_data['en'].to_pandas().tolist()
    de_sentences = de_data['de'].to_pandas().tolist()

    en_all_idx = []
    de_all_idx = []

    for en_sentence in tqdm(en_sentences, desc='Processing English sentences'):
        new_en_sentence = tokenizer.bos_token + " " + en_sentence
        en_idx = tokenizer.encode_plus(
            new_en_sentence,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )["input_ids"]
        en_all_idx.append(en_idx)

    for de_sentence in tqdm(de_sentences, desc='Processing German sentences'):
        new_de_sentence = tokenizer.bos_token + " " + de_sentence
        de_idx = tokenizer.encode_plus(
            new_de_sentence,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )["input_ids"]
        de_all_idx.append(de_idx)

    # 将结果转换为张量（如果需要拼接或进一步处理）
    en_all_idx = torch.cat(en_all_idx, dim=0)
    de_all_idx = torch.cat(de_all_idx, dim=0)

    # 这里可以保存处理后的结果，例如使用torch.save
    torch.save(en_all_idx, 'en_processed_indexes_val.pt')
    torch.save(de_all_idx, 'de_processed_indexes_val.pt')

else:
    print("未找到所需的句子列")
