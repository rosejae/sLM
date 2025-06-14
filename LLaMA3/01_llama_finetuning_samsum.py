### Hugging Face login ###
import huggingface_hub
huggingface_hub.login()

###################
#### inference ####
###################

import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig       # 8비트 양자화
from llama_cookbook.configs import train_config as TRAIN_CONFIG

train_config = TRAIN_CONFIG()
train_config.model_name = "meta-llama/Llama-3.1-8B-Instruct"
train_config.num_epochs = 1                         # epoch 수
train_config.run_validation = False                 # 학습 과정에서 validation 수행 여부
train_config.gradient_accumulation_steps = 4       # 그라디언트 누적 스텝 수

train_config.batch_size_training = 8                 # 학습 시 사용하는 배치 크기
train_config.lr = 1e-4                              # 학습률
train_config.use_fast_kernels = True                # fast kernel 사용 여부
train_config.use_fp16 = True                        # FP16 연산 사용 여부
train_config.context_length = 1024 if torch.cuda.get_device_properties(0).total_memory < 16e9 else 2048  # T4 16GB or A10 24GB
train_config.batching_strategy = "packing"        # 배치 전략 (여기서는 parking 전략 사용)
train_config.output_dir = "/content/drive/MyDrive/gdrive/llama_result"

config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model =  LlamaForCausalLM.from_pretrained(        # 모델 호출
    train_config.model_name,
    quantization_config=config,
    device_map="auto",
    use_cache=False,
    attn_implementation="sdpa" if train_config.use_fast_kernels else None,
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)       
tokenizer.pad_token = tokenizer.eos_token

eval_prompt = """
Summarize this dialog:
A: Hi Tom, are you busy tomorrow’s afternoon?
B: I’m pretty sure I am. What’s up?
A: Can you go with me to the animal shelter?.
B: What do you want to do?
A: I want to get a puppy for my son.
B: That will make him so happy.
A: Yeah, we’ve discussed it many times. I think he’s ready now.
B: That’s good. Raising a dog is a tough issue. Like having a baby ;-)
A: I'll get him one of those little dogs.
B: One that won't grow up too big;-)
A: And eat too much;-))
B: Do you know which one he would like?
A: Oh, yes, I took him there last Monday. He showed me one that he really liked.
B: I bet you had to drag him away.
A: He wanted to take it home right away ;-).
B: I wonder what he'll name it.
A: He said he’d name it after his dead hamster – Lemmy  - he's  a great Motorhead fan :-)))
---
Summary:
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
# 토크나이저를 사용하여 텍스트 데이터를 토크나이징, pytorch텐서로 변환해서 gpu로 이동

model.eval()              
with torch.no_grad():     
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))


#####################
#### fine-tuning ####
#####################

### DataLoader ###
from llama_cookbook.configs.datasets import samsum_dataset
from llama_cookbook.data.concatenator import ConcatDataset
from llama_cookbook.utils.config_utils import get_dataloader_kwargs
from llama_cookbook.utils.dataset_utils import get_preprocessed_dataset

train_dataset = get_preprocessed_dataset(tokenizer, samsum_dataset, 'train')                  
if train_config.batching_strategy == "packing":                                               
        train_dataset = ConcatDataset(train_dataset, chunk_size=train_config.context_length)

train_dl_kwargs = get_dataloader_kwargs(train_config, train_dataset, tokenizer, "train")      
train_dataloader = torch.utils.data.DataLoader(                                              
    train_dataset,
    num_workers=train_config.num_workers_dataloader,
    pin_memory=True,
    **train_dl_kwargs,
)

### PEFT ###
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from dataclasses import asdict
from llama_cookbook.configs import lora_config as LORA_CONFIG

lora_config = LORA_CONFIG()     # LoRA 설정
lora_config.r = 8               # low rank size
lora_config.lora_alpha = 31     # LoRA의 스키일링 팩터
lora_dropout: float=0.01        # 드롭아웃 비율 0.01

peft_config = LoraConfig(**asdict(lora_config))   
model = prepare_model_for_kbit_training(model)   
model = get_peft_model(model, peft_config)       

### train ###
import torch.optim as optim
from llama_cookbook.utils.train_utils import train
from torch.optim.lr_scheduler import StepLR

model.train()
optimizer = optim.AdamW(                                    
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

results = train(
    model,
    train_dataloader,
    None,
    tokenizer,
    optimizer,
    scheduler,
    train_config.gradient_accumulation_steps,
    train_config,
    None,
    None,
    None,
    wandb_run=None,
)

model.save_pretrained(train_config.output_dir+"meta-llama-oz1115")

### TEST ###
model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

model.push_to_hub("oz1115/meta_llama_peft", use_auth_token=True)