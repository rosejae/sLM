from datasets import (
    load_dataset, 
    DatasetDict,
)

from transformers import AutoTokenizer

def save_to_hf(model, model_name, hf_name):
    model.save_pretrained(model_name)
    model.push_to_hub(hf_name)

    print(f"Model saved to '{model_name}' and pushed to '{hf_name}'")
    return model_name  

def load_tokenizer(tokenizer_name='psymon/KoLlama2-7b'):
    return AutoTokenizer.from_pretrained(tokenizer_name)

class DataPreparer():
    def __init__(self, dataset_name='lpai-lab/kullm-v2', tokenizer_name='psymon/KoLlama2-7b', split='train', sample_size=50000, context_length=128, PEFT=False):
        """ 
        meta에서 제공하는 tokenizer는 영문을 훨씬 잘함
        우리의 데이터셋을 잘 tokenize 하는 모델을 사용하는 것을 추천함 
        """
        self.context_length = context_length
        self.tokenizer = load_tokenizer(tokenizer_name)
        
        self.raw_dataset = load_dataset(dataset_name, split)
        
        if PEFT == False:
            self.sampled_dataset = self.raw_dataset.select(range(sample_size))
            
        else:
            train_test_split = self.raw_dataset.train_test_split(test_size=0.1)
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
            
    def __call__(self):
        return self.sampled_dataset.map(self.tokenize, batched=True, remove_columns=self.raw_dataset.column_names)
    
    def tokenize(self, batch):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        outputs = tokenizer(
            batch['output'],
            max_length=self.context_length,
            truncation=True,
            return_overflowing_tokens=True,
            return_length=True
        )

        input_batch=[]
        for length, input_ids in zip(outputs['length'], outputs['input_ids']):
            if length == self.context_length:
                input_batch.append(input_ids)
                
        return {"input_ids": input_batch}
    
    def get_tokenizer(self):
        return self.tokenizer

if __name__ == "__main__":
    import torch
    
    from pre_model import model
    
    device = torch.device("cuda")
    
    tokenizer = load_tokenizer()
    
    prompt = "서울에 대해 알려줘"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    generate_ids = model.generate(inputs.input_ids, max_length=50, attention_mask=inputs.attention_mask) 
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    print(f"answer decoded: {output[0]}")