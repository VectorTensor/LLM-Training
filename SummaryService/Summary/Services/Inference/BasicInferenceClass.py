from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
from datasets import load_dataset

class BasicInferenceClass:
    def __init__(self,dataset,model_name):
        self._dataset = dataset
        self._modelName = model_name
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=True)

    def make_prompt(self,example_indices_full, text_to_summerize):
        prompt = ''
        for index in example_indices_full:
            dialogue = self._dataset['test'][index]['dialogue']
            summary = self._dataset['test'][index]['summary']

            prompt += f"""

                Dialogue:
                {dialogue}

                What was going on?
                {summary}
            """
        dialogue = text_to_summerize
        prompt += f"""

        Dialogue:
        {dialogue}
        What was going on?
        """
        return prompt

    def generate_output(self,indices,input_text):
        prmpt = self.make_prompt(indices,input_text)

        inputs = self._tokenizer(prmpt, return_tensors='pt')
        output = self._tokenizer.decode(
            self._model.generate(


            inputs["input_ids"],
            max_new_tokens = 50
        )[0],
        skip_special_tokens=True
        )
        return output




