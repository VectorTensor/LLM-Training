from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
from datasets import load_dataset


class ZeroShotInference:
    def __init__(self,model_name):
        self._modelName = model_name
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=True)

    def generate(self,dialogue):

        prompt =f"""
            Dialogue
            {dialogue}
            
            What was going on?
            """

        inputs = self._tokenizer(prompt, return_tensors='pt')
        output = self._tokenizer.decode(
            self._model.generate(
                inputs["input_ids"],
                max_new_tokens=50
            )[0],
            skip_special_tokens=True
        )

        return output


