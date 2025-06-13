import os, json, random, string, re
import pandas as pd
from tqdm import tqdm
from datasets import dataset_dict, load_dataset, Dataset, DatasetDict

### json load ###
base_path = r"C:\\Users\\jjsd4\\Desktop\\018.논문자료 요약 데이터\\01.데이터"
train_path = r"1. Training\\1. 라벨링데이터_231101_add"
new_path = os.path.join(base_path, train_path)
data_folder = os.listdir(new_path)

df = pd.DataFrame(columns=["text", "doc_id", "domain"])
instruction_list = []
text_list = []
doc_id_list = []
doc_id = 0

for d in data_folder:
    path = new_path + "\\" + d
    folder = os.listdir(path)

    for f in folder:
        print(f)
        path = new_path + "\\" + d + "\\" + f
        sub_df = pd.DataFrame(columns=["text", "doc_id", "domain"])

        with open(path, "r", encoding="UTF8") as file:
            json_file = json.load(file)

        try:
            documents = json_file[0]["data"]
        except:
            documents = json_file["data"]

        for i in tqdm(range(len(documents))):
            text1 = documents[i]["summary_section"][0]["orginal_text"]
            text_list.append(text1)
            doc_id_list.append("AIHUB_논문자료_"+str(doc_id))
            doc_id += 1

        sub_df["domain"] = ""
        sub_df["doc_id"] = doc_id_list
        sub_df["text"] = text_list
        df = pd.concat([df, sub_df], axis=0)
        text_list = []
        instruction_list = []
        doc_id_list = []

df.to_parquet(new_path+"\\summary_thesis.parquet", engine="pyarrow", index=False)