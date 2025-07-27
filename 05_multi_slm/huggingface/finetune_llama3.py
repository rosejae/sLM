import torch
import transformers

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    prepare_model_for_kbit_training, 
    LoraConfig, 
    get_peft_model,
)

#
# load quantized model
#

# quantization
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_id = "beomi/Llama-3-Open-Ko-8B-Instruct-preview"
tokenizer = AutoTokenizer.from_pretrained(model_id, quantization_config=config)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cuda:0",
    trust_remote_code=True,
    quantization_config=config
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# print the number of weights
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params}\nall params: {all_param}\ntrainable%: {100 * trainable_params / all_param}"
    )

# Lora
config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

#
# load data
#

from datasets import load_dataset

dataset = load_dataset("nlpai-lab/databricks-dolly-15k-ko")

categories = ['closed_qa', 'information_extraction', 'summarization']
filtered_dataset = dataset['train'].filter(lambda example: example['category'] in categories)

SYSTEM_PROMPT = "You are an assistant for answering questions. You are given the extracted parts of a long document and a question. Provide a conversational answer. Don't make up an answer."

tokenizer.pad_token = tokenizer.eos_token

def get_rag_train_prompt(row):
    question = "Context에 따르면, " + row['instruction']
    context = row['context']
    answer = row['response']

    user_prompt = f'###Context:{context}\n###Question:{question}'

    messages = [
        {"role": "system", "content" : SYSTEM_PROMPT},
        {"role": "user", "content" : user_prompt},
        {"role": "assistant", "content" : answer}
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        padding=True,
        truncation=True
    )

    return {"input_ids": encoded}

new_dataset = filtered_dataset.map(get_rag_train_prompt)
# print(tokenizer.decode(new_dataset[1]['input_ids']))

#
# train
#

trainer = transformers.Trainer(
    model=model,
    train_dataset=new_dataset,
    args=transformers.TrainingArguments(
        auto_find_batch_size=True,
        gradient_checkpointing=True,
        warmup_steps=10,
        max_steps=2400,
        save_steps=200,
        save_total_limit=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir="outputs",
        optim="paged_adamw_8bit",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train()


