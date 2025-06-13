import os, json, random, string, re
import pandas as pd
from tqdm import tqdm
from datasets import dataset_dict, load_dataset, Dataset, DatasetDict

### json load ###
base_path = r"C:\\Users\\jjsd4\\Desktop\\049.일반상식 문장 생성 평가 데이터\\01-1.정식개방데이터"
train_path = r"Training\\01.원천데이터"
new_path = os.path.join(base_path, train_path)
data_folder = os.listdir(new_path)

df = pd.DataFrame(columns =["text", "doc_id", "domain"])
instruction_list = []
text_list = []
doc_id_list = []
doc_id = 0

for f in tqdm(data_folder):
    print(f)
    path = default_path + "\\" + f
    sub_df = pd.DataFrame(columns=["text", "doc_id", "domain"])
    try:
        with open(path, "r", encoding="UTF8") as file:
            json_file = json.load(file)

        for documents in json_file:
            ### label ###
            text = documents["sentence"]
            genSentences = documents["genSentences"]
            genChoice = random.choice(genSentences)
            text_list.append(genChoice["label-scenes"].strip())

            ### words permutated ###
            # instruction_sentence = "임의의 순서대로 나열된 단어들을 보고 적절한 문장으로 재구성하세요.\n\n임의의 순서로 나열된 단어: ["
            # for concept in documents["concepts"]:
            #     instruction_sentence += concept["stem"] + " "
            # instruction_sentence = instruction_sentence.strip()[:-1]
            # instruction_sentence += "]"
            # instruction_list.append(instruction_sentence)
            doc_id_list.append("AIHUB_일반상식문장생성데이터_" +str(doc_id))
            doc_id += 1
    except Exception as e:
        print(e)
        pass

sub_df["domain"] = ""
sub_df["doc_id"] = doc_id_list
sub_df["text"] = text_list
df = pd.concat([df, sub_df], axis=0)
df.to_parquet(base_path+"\\summary_gen.parquet", engine="pyarrow", index=False)