import os, json, random, string, re
import pandas as pd
from tqdm import tqdm
from datasets import dataset_dict, load_dataset, Dataset, DatasetDict
### json load ###
base_path = r"C:\\Users\\jjsd4\Desktop\\도서자료 요약"
train_path = "Training"
new_path = os.path.join(base_path, train_path)
data_folder = os.listdir(new_path)
df = pd.DataFrame(columns=["input", "instruction", "output"])
text_list = []
instruction_list = []
for d in data_folder:
    ### [원천]도서요약_train
    path = new_path + "\\" + d
    folder2 = os.listdir(path)
    for f2 in folder2:
        ### 기술과학, 기타, 사회과학, 예술
        path = new_path + "\\" + d + "\\" + f2
        folder3 = os.listdir(path)
        sub_df = pd.DataFrame(columns=["input", "instruction", "output"])
        for f3 in tqdm(folder3):
            ### json files ###
            path = new_path + "\\" + d + "\\" + f2 + "\\" + f3
            with open(path, "r", encoding="UTF8") as file:
                json_file = json.load(file)
            number = random.choice([0, 1])
            documents = json_file["passage"]
            documents_split = documents.split(".")
            documents_split = [d0 for d0 in documents_split if d0 != ""]
            ### 문장이 1개밖에 없거나 %문자가 포함되면 number=1 ###
            ### 문장 이어쓰기 task 제외 ###
            if len(documents_split) == 1:
                number = 1
            if "%" in documents:
                number = 1
            if number == 0:
                text = json_file["summary"]
                ### 문장 이어쓰기 TASK를 위해 마지막 문장 제거 ###
                documents = json_file["passage"]
                documents_split = documents.split(".")
                documents_split = [d0 for d0 in documents_split if d0 != ""]
                doc_text = ""
                for d0 in documents_split[:-1]:
                    doc_text += d0 + ". "
                ### 마지막 문장 (LABEL) 따로 저장 ###
                text_list.append(documents_split[-1].strip())                
                instruction_text = "주어진 문장 뒤에 자연스럽게 이어질 문장을 생성해주세요.\n\n문장: "
                instruction_text += doc_text.strip()
                instruction_list.append(instruction_text.strip())
            elif number == 1:
                documents = json_file["passage"]
                text_list.append(documents.strip())
                ### 제목과 요약문을 활용한 TASK ###
                text = json_file["summary"]
                title = json_file["metadata"]["doc_name"]
                label = json_file["metadata"]["kdc_label"]
                instruction_text = "주어진 제목과 요약문에 대한 정보를 토대로, 요약되기 전 문장을 유추해서 생성해주세요.\n\n"
                instruction_text += "제목: " + title + "\n"
                instruction_text += "요약문: " + text.strip()
                instruction_list.append(instruction_text.strip())
            else:
                documents = json_file["passage"]
                text = json_file["summary"]
                text_list.append(text.strip())
                ### 제목과 카테고리를 활용한 TASK ###
                title = json_file["metadata"]["doc_name"]
                label = json_file["metadata"]["kdc_label"]
                instruction_text = "주어진 제목과 카테고리에 대한 정보를 토대로, 적합한 문장을 생성해주세요.\n\n"
                instruction_text += "제목: " + title + "\n"
                instruction_text += "카테고리: " + "[" + label + "]"
                instruction_list.append(instruction_text.strip())
        sub_df["input"] = ""
        sub_df["instruction"] = instruction_list
        sub_df["output"] = text_list
        print(len(sub_df))
        df = pd.concat([df, sub_df], axis=0)
        text_list = []
        instruction_list = []
df.reset_index(drop=True, inplace=True)
df.to_parquet(base_path+r"\\summary_book.parquet", engine="pyarrow", index=False)