import pandas as pd
from datasets import dataset_dict, load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from typing import Text
import random, string, re, os, kss

cur_addr = os.getcwd()
new_df = pd.read_parquet(cur_addr+r"\\summary_gen.parquet")
filtered_df = new_df

### 구두점이 있는 문장 합치기 ###
for i in range(len(filtered_df)):
    filtered_df['text'][i] = filtered_df['text'][i].strip()
    try:
        if filtered_df['text'][i].endswith('. '):
            filtered_df['text'][i] = filtered_df['text'][i][:-2] + '.'
        elif filtered_df['text'][i].endswith('.'):
            pass
        elif filtered_df['text'][i].endswith(' .'):
            filtered_df['text'][i] = filtered_df['text'][i][:-2] + '.'
        else:
            print(i, '번째 문장')
            print(filtered_df['text'][i])

        if filtered_df['text'][i].endswith(' '):
            filtered_df['text'][i] = filtered_df['text'][i][:-1] + '.'
        # else:
        #     filtered_df['text'][i] = filtered_df['text'][i] + '.'

    except Exception as e:
        print(e)
        print(i)
        print(filtered_df['text'][i])
        print('################################################')

### 특수문자 삭제 ###
remove_index = []
for i in range(len(filtered_df)):
    text = filtered_df['text'][i]
    if '�' in text:
        remove_index.append(i)
    elif '삭제.' in text[-5:]:
        remove_index.append(i)

filtered_df.drop(remove_index, inplace=True)
filtered_df.reset_index(drop=True, inplace=True)

### 결측치 제거 ###
filtered_df = filtered_df.dropna(subset=['text'])
filtered_df = filtered_df.reset_index(drop=True)

"""
아래 방법은 확실한 방법이 아님 
아래방법으로는 중간에 '.'가 있으면 필터링을 못함 
"""
### corpus filtering ###
filtered_df['len_elements'] = filtered_df['text'].apply(lambda x: len(x.split('. ')) if isinstance(x, str) else 1)
filtered_df = filtered_df.reset_index(drop=True)
filtered_df_1 = filtered_df[(filtered_df['len_elements'] >= 2) & (filtered_df['len_elements'] < 15 )] 
filtered_df_2 = filtered_df[filtered_df['len_elements'] < 2 ]
filtered_df_1 = filtered_df_1.reset_index(drop=True)
# filtered_df['len_elements'].value_counts()

### filtered_df_1를 보고 직접 수정해야함함 ###
filtered_df_1['text'] = ['검은색, 붉은색, 녹색의 다채로운 쌀과, 찹쌀 등을 생산하고 있다.', '이 화장품 회사는 예전부터 1위 자리를 지키고 있다.']
filtered_df_1['len_elements'] = 1

filtered_df_2 = filtered_df_2.reset_index(drop=True)
filtered_df_2 = pd.concat([filtered_df_2, filtered_df_1], axis=0)
filtered_df_2 = filtered_df_2.reset_index(drop=True)