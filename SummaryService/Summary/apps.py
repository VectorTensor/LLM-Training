import inject
from django.apps import AppConfig
from .Services.Inference.ZeroShot import ZeroShotInference
from .Services.Inference.Inference import OneShot, FewShot
from datasets import load_dataset

class SummaryConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Summary'

    def ready(self):
        def my_config(binder):
            huggingface_dataset_name = "knkarthick/dialogsum"
            dataset = load_dataset(huggingface_dataset_name)
            binder.bind(ZeroShotInference,ZeroShotInference('google/flan-t5-base'))
            binder.bind(OneShot,OneShot(dataset,'google/flan-t5-base'))
            binder.bind(FewShot,FewShot(dataset,'google/flan-t5-base'))

        inject.configure(my_config)
