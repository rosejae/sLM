
#
# data load for full tuning 
#

from datasets import load_dataset, DatasetDict

raw_dataset = load_dataset('nlpai-lab/kullm-v2', split='train')
sampled_dataset = raw_dataset.select(range(50000))


#
# load Korean tokenizer not llama3 
#

from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling

tokenizer = AutoTokenizer.from_pretrained('psymon/KoLlama2-7b')

context_length = 128
def tokenize(batch):
    outputs = tokenizer(
        batch['output'],
        max_length=context_length,
        truncation=True,
        return_overflowing_tokens=True,
        return_length=True
    )

    input_batch=[]
    for length, input_ids in zip(outputs['length'], outputs['input_ids']):
        if length==context_length:
        input_batch.append(input_ids)
    return {"input_ids": input_batch}

tokenized_datasets = sampled_dataset.map(tokenize, batched=True, remove_columns=raw_dataset.column_names)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

#
# full tuning model (pre-model)
#

import torch, gc
from transformers import LlamaForCausalLM
from transformers import LlamaConfig
from transformers import TrainingArguments
from transformers import Trainer

device = torch.device("cuda")

configuration = LlamaConfig()
configuration = LlamaConfig(**{
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 512,
    "initializer_range": 0.02,
    "intermediate_size": 1376,
    "max_position_embeddings": 128,
    "model_type": "llama",
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": False,
    "transformers_version": "4.28.0",
    "use_cache": True,
    "vocab_size": 32000
})

model = LlamaForCausalLM(configuration).to(device)

batch_size = 16 # 32
logging_steps = 10 # 100
learning_rate = 3e-3 # 5e-4
num_epochs = 1

args = TrainingArguments(
    output_dir='/content/drive/MyDrive/gdrive/llama_result/testllama',  # 학습 결과(모델, 체크포인트, 로그 등) 가 저장될 경로 지정
    per_device_train_batch_size=batch_size,       # 학습 시 사용되는 디바이스(예: GPU)당 배치 크기를 지정
    per_device_eval_batch_size=batch_size,        # 평가 시 사용되는 디바이스 당 배치 크기를 지정
    logging_steps=logging_steps,                  # 몇 스텝마다 로그를 기록할지 지정
    save_steps=logging_steps,                     # 몇 스텝마다 모델 체크포인트를 저장할지 지정
    gradient_accumulation_steps=8,                # 그라디언트 누적 스텝 수를 지정. 이를 통해 더 큰 가상 배치 크기를 사용
    num_train_epochs=num_epochs,                  # 전체 학습 데이터셋을 몇 번 반복할지 지정
    weight_decay=0.1,                             # 가중치 감쇠율을 지정. 이는 모델의 과적합을 방지하는데 도움
    warmup_steps=logging_steps,                   # 학습 초기의 워밍업 단계에서 사용할 스텝 수 지정
    lr_scheduler_type='cosine',                   # 학습률 스케줄러의 타입을 지정. 여기서는 'cosine' 스케줄러 사용
    learning_rate=learning_rate,                  # 초기 학습률 지정
    fp16=True,                                    # FP16(반 정밀도) 연산을 사용하여 훈련 속도를 높이고 메모리 사용량을 감소
    push_to_hub=False                             # 허깅페이스에 푸시할 지 여부
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
)

gc.collect()
torch.cuda.empty_cache()
trainer.train()

model.save_pretrained('pre_llama')
tokenizer.save_pretrained('pre_llama')

model.push_to_hub('oz1115/llama_pre_model')
tokenizer.push_to_hub('oz1115/llama_pre_tokenizer')



#
# data for PEFT
#

from datasets import load_dataset, DatasetDict

raw_dataset = load_dataset("maywell/ko_wikidata_QA", split="train")
train_test_split = raw_dataset.train_test_split(test_size=0.1) 
train_validation_split = train_test_split['train'].train_test_split(test_size=0.1)

dataset = DatasetDict({
    'train': train_validation_split['train'],
    'test': train_test_split['test'],
    'validation': train_validation_split['test'],
})

sampled_dataset = DatasetDict(
    {
        "train": dataset['train'].select(range(10000)).shuffle(),
        "valid": dataset['test'].select(range(1000)).shuffle()
    }
)

tokenized_datasets = sampled_dataset.map(tokenize, batched=True, remove_columns=raw_dataset.column_names)

#
# PEFT 
#

from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,    # 모델의 작업 유형을 지정합니다. 여기서는 TaskType.CAUSAL_LM으로 설정하여 인과 언어 모델링 작업을 수행
                       inference_mode=False,              # 추론 모드 설정 (False로 설정하여 학습 모드로 설정)
                       r=4,                               # 로우랭크 크기.  이는 매개변수의 효율성을 높이기 위해 사용되는 저차원 행렬의 랭크를 의미합니다. 여기서는 4로 설정
                       lora_alpha=16,                     # PEFT의 추론 간섭정도.  이는 로우랭크 행렬의 스케일링 팩터로, 모델의 학습 및 추론 성능에 영향을 미침. 여기서는 16으로 설정
                       lora_dropout=0.1,                  # 드롭아웃 비율을 설정. 드롭아웃은 과적합을 방지하기 위해 뉴런의 일부를 무작위로 비활성화하는 기법. 여기서는 0.1로 설정
                       )
model = get_peft_model(model, peft_config).to(device)
model.print_trainable_parameters()

args=TrainingArguments(
    output_dir="/content/drive/MyDrive/gdrive/llama_result/pre_llama",
    per_device_train_batch_size=4,        #데이터 배치 배치사이즈
    logging_steps=500,                    #훈련에서 로깅할 단계
    gradient_accumulation_steps=8,        #8단계마다 w조정
    num_train_epochs=1,                   #전체 훈련 데이터세트 반복 획수
    weight_decay=0.1,                     #w를 10%씩 손실을 고의로 일으며, overfitting을 방지
    lr_scheduler_type="cosine",             #LR 변화를 코사인 함수 형태로 변화
    learning_rate=5e-4,                   #학습률
    save_steps=1000,                      #기록 저장 스텝
    fp16=True,                            #16비트 부동소수점 연산(True:메모리 사용량 감소, 속도 증가)
    push_to_hub=False,                    # 허깅페이스 공유 여부
)

trainer=Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
)

trainer.train()
model.save_pretrained("/content/drive/MyDrive/gdrive/llama_result/peft_llama_adapter")

#
# pre model + PEFT
#

from transformers import LlamaForCausalLM
from peft import PeftModel, PeftConfig

base_model = LlamaForCausalLM.from_pretrained('pre_llama')
model_load = PeftModel.from_pretrained(base_model, '/content/drive/MyDrive/gdrive/llama_result/peft_llama_adapter').to(device)
model = model_load.merge_and_unload()

#
# inference
#

import os
os.stat('/content/drive/MyDrive/gdrive/llama_result/peft_llama_adapter/adapter_model.safetensors').st_size/(1024*1924)

question = "알고리즘 분석"
prompt=f"""{question}"""
inputs = tokenizer(prompt, return_tensors='pt')
inputs.to(device)
generate_ids = model.generate(inputs.input_ids, max_length=100)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]










