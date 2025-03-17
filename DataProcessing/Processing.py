import pandas as pd
import os

# 定义Train文件夹路径
train_folder = os.path.join('DataProcessing', 'Train')
# 获取Train文件夹下所有.parquet文件路径
train_files = [os.path.join(train_folder, file) for file in os.listdir(train_folder) if file.endswith('.parquet')]
# 读取并合并数据
combined_train_df = pd.concat([pd.read_parquet(file) for file in train_files], ignore_index=True)
# 保存合并后的数据
combined_train_path = os.path.join(train_folder, 'train.parquet')
combined_train_df.to_parquet(combined_train_path, index=False)

combined_train_path = os.path.join('DataProcessing', 'Train', 'combined_train.parquet')
combined_train_df = pd.read_parquet(combined_train_path)

# 提取数据
extracted_data = []
for _, row in combined_train_df.iterrows():
    translation_dict = row['translation']
    en_value = translation_dict.get('en', None)
    de_value = translation_dict.get('de', None)
    extracted_data.append({'en': en_value, 'de': de_value})

# 创建 DataFrame
extracted_df = pd.DataFrame(extracted_data)

# 查看提取后的数据预览
print("提取后的en和de列数据预览：")
print(extracted_df.head())

# 保存为 .parquet 文件
train_folder = os.path.join('DataProcessing', 'Train')

# 保存 en 数据
en_path = os.path.join(train_folder, 'en_data_train.parquet')
extracted_df[['en']].dropna().to_parquet(en_path, index=False)

# 保存 de 数据
de_path = os.path.join(train_folder, 'de_data_train.parquet')
extracted_df[['de']].dropna().to_parquet(de_path, index=False)

print(f"en 数据已保存到: {en_path}")
print(f"de 数据已保存到: {de_path}")
def get_file_size(file_path):
    return os.path.getsize(file_path)
train_folder = os.path.join('DataProcessing', 'train')
en_path = os.path.join(train_folder, 'en_data_train.parquet')
de_path = os.path.join(train_folder, 'de_data_train.parquet')
if os.path.exists(en_path):
    en_size = get_file_size(en_path)
    en_df = pd.read_parquet(en_path)
    en_shape = en_df.shape
    en_count = en_shape[0]
    print(f"en 文件大小: {en_size} 字节")
    print(f"en DataFrame 形状: {en_shape}")
    print(f"en 数据行数: {en_count}")
else:
    print(f"{en_path} 不存在")

# 获取并打印 de 文件的大小、形状和行数
if os.path.exists(de_path):
    de_size = get_file_size(de_path)
    de_df = pd.read_parquet(de_path)
    de_shape = de_df.shape
    de_count = de_shape[0]
    print(f"de 文件大小: {de_size} 字节")
    print(f"de DataFrame 形状: {de_shape}")
    print(f"de 数据行数: {de_count}")
else:
    print(f"{de_path} 不存在")


def get_file_size(file_path):
    return os.path.getsize(file_path)

#看看第一行数据
print("en数据第一行：")
print(en_df.head(1))
print("de数据第一行：")
print(de_df.head(1))


