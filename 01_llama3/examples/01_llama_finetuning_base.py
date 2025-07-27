################
### pipeline ###
################

import transformers
import torch

model_id="meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"},
]
"""
terminators =[
    pipeline.tokenizer.eos_token,
    pipeline.tokenizer.convert_tokens_to_string(["<|eot_id|>"])
]
"""
# Use only the integer ID of the EOS token
eos_token_id = pipeline.tokenizer.eos_token_id

outputs=pipeline(
    messages,
    max_new_tokens=256,
    eos_token_id=eos_token_id,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)

print(outputs[0]["generated_text"])

##################
### model load ###
##################

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id="meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    #tokenizer.convert_tokens_to_string("<|eot_id|>")
]
outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)

response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))