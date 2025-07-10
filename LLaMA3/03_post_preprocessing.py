import pandas as pd
from datasets import dataset_dict, load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from typing import Text
import random, string, re, os, kss
from os import TMP_MAX

### Sentence Order Inference (SOI) ###
def make_text_allign_data(text):
    ### ' .'도 고려해야할 것 같은데, preprocessing에서 이미 제외시킴 ###
    ### 만약 ' '만 있으면? ###
    if not (text.endswith('.') or text.endswith('. ')):
        text += '.'
    if text.endswith('.'):
        text += ' '
    my_list = text.split('. ')
    my_list_with_dots = [element + '.' for element in my_list]
    
    ### 맨 마지막 문장이 '.'만 있으면 제외 ###
    if my_list_with_dots[-1] == '.':
        my_list_with_dots=my_list_with_dots[:-1]
        
    ### 문장장들을 섞음 ###
    random.shuffle(my_list_with_dots)
    if len(my_list_with_dots) == 1:
        raise Exception("allign 리스트 기이가 1입니다", text)
    
    tmp_input = '당신은 인공지능 비서입니다. 주어진 문장 리스트를 모두 활용하여 가장 정확한 단락을 생성하세요.'
    tmp_inst = f'이 문장들은 임의의 순서로 섞여 있습니다. 모든 문장을 활용하여 원본 단락의 순서와 내용을 올바른 순서로 재구성하세요.\n#문장 리스트: {my_list_with_dots}'
    
    ### 이미 위에서 고려한 것 같은데, 한번 더 확인 ###
    if text.endswith(' '):
        tmp_out = text[:-1]
    else:
        tmp_out = text
        
    return tmp_input, tmp_inst, tmp_out

### Last Sentence Prediction (LSP) ###
def make_completion_data(text):
    if not (text.endswith('.') or text.endswith('. ')):
        text += '.'
    if text.endswith('. '):
        text=text
    else:
        text = text +' '
    my_list_with_dots = text.split('. ')
    
    ### 마지막 문장이 아무것도 없은 빈깡통이면 제외외 ###
    if my_list_with_dots[-1] == '':
        my_list_with_dots = my_list_with_dots[:-1]
        
        if len(my_list_with_dots) == 1:
            raise Exception("Completions 텍스트를 나눈 리스트의 길이가 1입니다.", text)
        
        ### 마지막 문장 저장 (label) ###
        last_sentence = my_list_with_dots.pop()
        last_sentence += '.'
        
    remaining_paragraph = ''
    
    for i in range(len(my_list_with_dots)):
        ### 맨 마지막 문장이면 '.'을 문장 끝에 붙여줌 ###
        if i! = len(my_list_with_dots) - 1:
            remaining_paragraph += my_list_with_dots[i] + '. '
        else:
            remaining_paragraph += my_list_with_dots[i] + '.'
    
    tmp_input = '당신은 인공지능 비서입니다. 주어진 원문을 바탕으로 주어진 질문에 가장 적절한 답변을 생성하세요.'
    tmp_instruct = f'다음 텍스트에서 제공된 문맥을 정확히 이해하고, 마지막 문장을 자연스럽고 문맥에 맞게 완성하세요. 문장은 이전 내용과 논리적으로 연결되어야 합니다.\n#텍스트: {remaining_paragraph}'
    
    tmp_output = last_sentence
    
    return tmp_input, tmp_instruct, tmp_output

### Mask Prediction ###
def make_text_mask_data(text):
    if not (text.endswith('.') or text.endswith('. ')):
        text += '.'
        
    if text.endswith('. '):
        text = text[:-1]
        
    else:
        text = text
        
    ### 두 글자 이상의 한글 단어만 추출 ###
    words = re.findall(r'[가-힣]{2,}', text)
    random_word = random.choice(words)
    masked_text = text.replace(random_word, '<MASK>')
    
    tmp_input = '당신은 인공지능 비서입니다. 주어진 질문에 가장 적절한 답변을 제공하세요.'
    tmp_instruct = f'이 문제에서는 주어진 텍스트 내의 <MASK>로 표시된 부분에 들어갈 적절한 단어를 예측해야 합니다. <MASK>가 위치한 문장의 전체 문맥을 분석하여, 문장의 나머지 내용과 일관되게 <MASK>에 들어갈 가장 적합한 단어를 답하세요.\n#텍스트: {masked_text}'
    
    tmp_output = random_word
    
    return tmp_input, tmp_instruct, tmp_output

### Word Order Inference ###
def make_word_align(text):
    word_lst=[]
    
    for word in text.split(' '):
        ### 단어에서 특수문자 제거 ###
        out = re.sub(r"[^\w\s]", "", word)
        word_lst.append(out)
        
    word_lst = set(word_lst)
    word_lst = list(word_lst)
    random.shuffle(word_lst)
    
    tmp_input = '당신은 인공지능 비서입니다. 주어진 지시사항에 따라 가장 적절한 문장을 생성하세요.'
    tmp_instruct = f'이 문제에는 문장에서 공백을 기준으로 나누고, 구두점을 제거한 무작위로 섞인 단어들이 담긴 리스트가 제공됩니다. 이 리스트의 단어를 모두 활용하여 가장 문맥상 적절한 문장을 생성하세요.\n#단어리스트: {word_lst}'
    
    tmp_output = text
    
    return tmp_input, tmp_instruct, tmp_output

### preprocessed data load ###
cur_addr = os.getcwd()
new_df = pd.read_parquet(cur_addr+r"\\filtered_df.parquet")
filtered_df = new_df

input_lst=[]
output_lst=[]
inst_lst=[]
id_lst=[]

for i in tqdm(range(len(filtered_df))):
    try:
        text = filtered_df['text'][i]
        thred = random.random()
        
        if thred < 0.56:
            TMP_MAXtmp_id='word_align_aihub'
            
        else:
            #tmp_input, tmp_instruct, tmp_output = make_text_mask_data(text)
            tmp_input, tmp_instruct, tmp_output = make_completion_data(text)
            tmp_id='pre_mask_aihub'
            
        input_lst.append(tmp_input)
        inst_lst.append(tmp_instruct)
        output_lst.append(tmp_output)
        id_lst.append(tmp_id)
        
    except Exception as e:
        print(f"{i}번째 행")
        print(e)
        print(filtered_df['text'][i])
        print("---------------------------------------------")
        
# for i in tqdm(range(len(filtered_df))):
#     try:

#         text=filtered_df['text'][i]

#         thred=random.random()

#         if thred < 0.44 :
#             tmp_input, tmp_instruct, tmp_output = make_completion_data(text)
#             tmp_id='completion_aihub'
#         elif thred < 0.88:
#             tmp_input, tmp_instruct, tmp_output = make_text_allign_data(text)
#             tmp_id='text_allign_aihub'
#         else:
#             tmp_input, tmp_instruct, tmp_output = make_text_mask_data(text)
#             tmp_id='pre_mask_aihub'

#         input_lst.append(tmp_input)
#         inst_lst.append(tmp_instruct)
#         output_lst.append(tmp_output)
#         id_lst.append(tmp_id)

#     except Exception as e:
#         print(f"{i}번째 행")
#         print(e)
#         print(filtered_df['text'][i])
#         print("----------------------------------------")

# for i in tqdm(range(len(filtered_df))):
#     try:
#         text=filtered_df['text'][i]

#         thred=random.random()
#         if thred < 0.56 :
#             tmp_input, tmp_instruct, tmp_output = make_word_align(text)
#             tmp_id='word_allign_aihub'
#         else:
#             tmp_input, tmp_instruct, tmp_output = make_text_mask_data(text)
#             tmp_id='pre_mask_aihub'

#         input_lst.append(tmp_input)
#         inst_lst.append(tmp_instruct)
#         output_lst.append(tmp_output)
#         id_lst.append(tmp_id)

#     except Exception as e:
#         print(f"{i}번째 행")
#         print(e)
#         print(filtered_df['text'][i])
#         print("----------------------------------------")

# total_df = pd.concat([hub_df_1,hub_df_2],axis=0)        
hub_df_2 = pd.DataFrame({'input':input_lst,'instruction':inst_lst,'output':output_lst})
hub_df = hub_df_2.copy()
hub_df = hub_df.reset_index(drop=True)

hub_df.to_parquet(cur_addr+r"\\post_procesessed_data.parquet", engine="pyarrow", index=False)

# from huggingface_hub import login
# from datasets import Dataset
# login()
# dataset = Dataset.from_pandas(total_df)
# dataset.push_to_hub('oz1115/korea_summary_Thesis')