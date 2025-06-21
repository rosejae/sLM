import pre_model
import peft
import data

import torch
import os

from transformers import LlamaForCausalLM
from peft import PeftModel

if __name__ == '__main__':
    
    ### pre model ###
    obj_pre_model = pre_model.llama_pre_model()
    _pre_model, _pre_tokenizer = obj_pre_model.llama_train()

    data.save_to_hf(_pre_model, 'llama3_pre_model', 'rosejae/llama3_pre_model')
    data.save_to_hf(_pre_tokenizer, 'llama3_pre_tokenizer', 'rosejae/llama3_pre_tokenizer')

    ### peft model ###
    obj_peft_model = peft.peft_model()
    _peft_model, _ = obj_peft_model.peft_train()
    
    data.save_to_hf(_peft_model, 'llama3_peft_model', 'rosejae/llama3_peft_model')
    
    ### merge ###
    device = torch.device('cuda')
    base_model = LlamaForCausalLM.from_pretrained('llama3_pre_model')
    model_load = PeftModel.from_pretrained(base_model, 'llama3_peft_model')
    model_load.to(device)
    model = model_load.merge_and_unload()

    # model.upload_to_hub('oz1115/llama_tuning_model')
    # os.stat('/content/drive/MyDrive/gdrive/llama_result/peft_llama_adapter/adapter_model.safetensors').st_size/(1024*1924)

    # 최종 모델에 쿼리 날려보기
    question = "알고리즘 분석"
    prompt=f"""{question}"""
    inputs = _pre_tokenizer(prompt, return_tensors='pt')
    inputs.to(device)
    generate_ids = model.generate(inputs.input_ids, max_length=100)
    _pre_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]