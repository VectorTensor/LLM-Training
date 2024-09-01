from datasets import load_dataset
from transformers import GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType



def print_number_of_trainable_model_parameters(model):
    """
Calculates the number of trainable model parameters and returns a formatted string with the results.

Args:
    model (torch.nn.Module): The model for which to calculate the trainable parameters.

Returns:
    str: A formatted string containing the number of trainable model parameters, the total number of model parameters, and the percentage of trainable parameters.
"""

    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters : {trainable_model_params}\n all model parameters: {all_model_params} \n percentage : {trainable_model_params / all_model_params}"


class CustomSummaryTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _tokenize_function(self, example):
        tokenizer = self.tokenizer
        start_prompt = 'Summarize the following conversation. \n\n'
        end_prompt = '\n\nSummary: '
        prompt = [start_prompt + dialogue + end_prompt for dialogue in example['dialogue']]
        example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
        example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True,
                                      return_tensors='pt').input_ids
        return example

    def get_tokenized_dataset(self, dataset):
        tokenized_datasets = dataset.map(self._tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary'])
        return tokenized_datasets


class PEFTModelWrapper:

    def __init__(self, model):
        self._peft_trainer = None
        self._tokenized_datasets = None
        self._tokenizer = None
        self._model = model
        self._lora_config = LoraConfig(
            r=40,
            lora_alpha=40,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self._peft_training_args = TrainingArguments(
            output_dir='./results',
            save_steps=999999999,
            save_total_limit=None,
            auto_find_batch_size=True,
            learning_rate=1e-3,
            save_strategy="no",  # Prevents saving final model

            num_train_epochs=1,
            logging_steps=1,
            max_steps=300
        )
        self.rouge = evaluate.load("rouge")

    def train(self, tokenizer, tokenized_datasets):
        self._peft_trainer = Trainer(
            model=self._model,
            args=self._peft_training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
        )
        self._tokenized_datasets = tokenized_datasets
        self._tokenizer = tokenizer
        self._peft_trainer.train()
        self._peft_trainer.evaluate()

        peft_model_path = './peft-dialogue-summary-checkpoint-local'
        tokenizer.save_pretrained(peft_model_path)
        self._peft_trainer.model.save_pretrained(peft_model_path)

    def rouge_score(self, d, dataset):
        dialogues = dataset[d][0:10]['dialogue']
        human_baseline_summaries = dataset[d][0:10]['summary']

        peft_model_summaries = []

        for _, dialogue in enumerate(dialogues):
            prompt = f"""
          Summarize the following conversation.
          {dialogue}
          Summary:

          """

            input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

            peft_model_outputs = self._model.generate(input_ids=input_ids,
                                                      generation_config=GenerationConfig(max_new_tokens=200,
                                                                                         num_beams=1))
            peft_model_text_outputs = self._tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
            peft_model_summaries.append(peft_model_text_outputs)

        peft_model_results = self.rouge.compute(

            predictions=peft_model_summaries,
            references=human_baseline_summaries,
            use_aggregator=True,
            use_stemmer=True
        )

        return peft_model_results

    def train_rouge(self,dataset):
        return self.rouge_score('train',dataset)

    def test_rouge(self,dataset):
        return self.rouge_score('test',dataset)
