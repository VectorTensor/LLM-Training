
from Inference.Inference import OneShot
from datasets import load_dataset

huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)
zf = OneShot(dataset, 'google/flan-t5-base')
print(zf.generate_one_shot_inferece('del'))