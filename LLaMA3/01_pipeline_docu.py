import os, json, random, string, re
import pandas as pd
from tqdm import tqdm
from datasets import dataset_dict, load_dataset, Dataset, DatasetDict
### json load ###
base_path = r"C:\\Users\\jjsd4\\Desktop\\문서요약 텍스트"
train_path = "Training"
new_path = os.path.join(base_path, train_path)
data_folder = os.listdir(new_path)
df = pd.DataFrame(columns=["input", "instruction", "output"])
text_list = []
instruction_list = []
### train_original_law.json, train_original_news.json, train_original_op.json ###
for d in data_folder:
    path = new_path + "\\" + d
    sub_df = pd.DataFrame(columns=["input", "instruction", "output"])
    print(path)
    with open(path, "r", encoding="UTF8") as file:
        json_file = json.load(file)
    documents = json_file["documents"]
    ### merge every sentence per id ###
    for i in tqdm(range(len(documents))):
        title = documents[i]["title"]
        # text = documents[i]["abstractive"][0]
        sentence_list = documents[i]["text"]
        ### choose one between two tasks ###
        number = random.choice([0,1])
        if number == 0:
            instruction_text ="주어진 문장을 적절하게 요약해주세요.\n\n문장: "
            for sentence in sentence_list:
                try:
                    if len(sentence) > 1:
                        for sub in sentence:
                            sen = sub["sentence"]
                            instruction_text += sen + " "
                    else:
                        sen = sentence[0]["sentence"]
                        instruction_text += sen + " "
                except:
                    pass
            instruction_list.append(instruction_text.strip())
            text_list.append(documents[i]["abstractive"][0])
        else:
            instruction_text = "주어진 문장에 적절한 제목을 생성하고, 내용을 요약해주세요.\n\n문장: "
            for sentenct in sentence_list:
                try:
                    if len(sentenct) > 1:
                        for sub in sentenct:
                            sen = sub["sentence"]
                            instruction_text += sen + " "
                    else:
                        sen = sentenct[0]["sentence"]
                        instruction_text += sen + " "
                except:
                    pass
            instruction_list.append(instruction_text.strip())
            answer = "제목: " + title + "\n" + documents[i]["abstractive"][0]
            text_list.append(answer)        
    sub_df["input"] = ""
    sub_df["instruction"] = instruction_list
    sub_df["output"] = text_list
    df = pd.concat([df, sub_df], axis=0)
    text_list = []
    instruction_list = []
df.reset_index(drop=True, inplace=True)
df["input"] = ""
df.to_parquet(base_path+"\\summary_docu.parquet", engine="pyarrow", index=False)