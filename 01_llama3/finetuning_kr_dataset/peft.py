import torch
import data

from transformers import (
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType,
)

""" 
이런 데이터를 사용해서 트레이닝을 한 번 더 시키면 
한국어에 대한 학습셋이 커짐 
훈련을 지속적으로 하면 모델 자체가 알고있는 폭이 넓어져
답변을 더 잘함
"""

class peft_model():
    def __init__(self):
        self.data_preparer = data.DataPreparer(dataset_name='maywell/ko_wikidata_QA', tokenizer_name='pre_llama', PEFT=True)
        self.tokenized_datasets = self.data_preparer()
        self.data_collator = DataCollatorForLanguageModeling(self.data_preparer.tokenizer, mlm=False)

        self.tokenizer = self.data_preparer.get_tokenizer()
        self.model = LlamaForCausalLM.from_pretrained("pre_llama")
        
        
        
    def peft_train(self):
        peft_config = self._peft_config()
        device = torch.device("cuda")
        model = get_peft_model(self.model, peft_config).to(device)

        model.print_trainable_parameters()
        train_config = self._train_config()
        
        trainer=Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.train_config,
            data_collator=self.data_collator,
            train_dataset=self.tokenized_datasets['train'],
        )
        
        trainer.train()
        
        return self.model, self.tokenizer
        
    def _train_config(self):
        return TrainingArguments(
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
        
    def _peft_config(self):
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,          # 작업 유형을 인과 언어 모델링으로 설정
            inference_mode=False,                   # 추론 모드 비활성화 (학습 모드 활성화)
            r=4,                                    # 로우랭크 크기 설정
            lora_alpha=16,                          # PEFT의 추론 간섭 정도 설정
            lora_dropout=0.1,                       # 드롭아웃 비율 설정
        )