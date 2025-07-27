import torch
import gc

import data

from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

class llama_pre_model():
    def __init__(self):
        self.batch_size = 16 # 32 
        self.logging_steps = 10 # 100
        self.learning_rate = 3e-3 # 5e-4
        self.num_epochs = 1
        
        self.data_preparer = data.DataPreparer()
        self.tokenized_datasets = self.data_preparer()
        self.data_collator = DataCollatorForLanguageModeling(self.data_preparer.tokenizer, mlm=False)
        
        device = torch.device("cuda")
        configuration = self._llama_config()
        self.model = LlamaForCausalLM(configuration).to(device)

    def llama_train(self):      
        trainer = Trainer(
            model=self.model,
            tokenizer=self.data_preparer.tokenizer,
            args=self._train_config(),
            data_collator=self.data_collator,
            train_dataset=self.tokenized_datasets,
        )
         
        gc.collect()
        torch.cuda.empty_cache()
        trainer.train()
        
        return self.model, self.data_preparer.tokenizer

    def _llama_config(self):
        configuration = LlamaConfig()
        return LlamaConfig(**{
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
            "vocab_size": 32000,
        })
        
    def _train_config(self):
        return TrainingArguments(
            output_dir='/content/drive/MyDrive/gdrive/llama_result/testllama',  # 학습 결과(모델, 체크포인트, 로그 등) 가 저장될 경로 지정
            per_device_train_batch_size=self.batch_size,# 학습 시 사용되는 디바이스(예: GPU)당 배치 크기를 지정
            per_device_eval_batch_size=self.batch_size, # 평가 시 사용되는 디바이스 당 배치 크기를 지정
            logging_steps=self.logging_steps,           # 몇 스텝마다 로그를 기록할지 지정
            save_steps=self.logging_steps,              # 몇 스텝마다 모델 체크포인트를 저장할지 지정
            gradient_accumulation_steps=8,         # 그라디언트 누적 스텝 수를 지정. 이를 통해 더 큰 가상 배치 크기를 사용
            num_train_epochs=self.num_epochs,           # 전체 학습 데이터셋을 몇 번 반복할지 지정
            weight_decay=0.1,                      # 가중치 감쇠율을 지정. 이는 모델의 과적합을 방지하는데 도움
            warmup_steps=self.logging_steps,            # 학습 초기의 워밍업 단계에서 사용할 스텝 수 지정
            lr_scheduler_type='cosine',            # 학습률 스케줄러의 타입을 지정. 여기서는 'cosine' 스케줄러 사용
            learning_rate=self.learning_rate,           # 초기 학습률 지정
            fp16=True,                             # FP16(반 정밀도) 연산을 사용하여 훈련 속도를 높이고 메모리 사용량을 감소
            push_to_hub=False,                     # 허깅페이스에 푸시할 지 여부
        )