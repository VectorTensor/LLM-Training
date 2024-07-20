import inject
from django.apps import AppConfig
from .Services.Inference.ZeroShot import ZeroShotInference


class SummaryConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Summary'

    def ready(self):
        def my_config(binder):
            binder.bind(ZeroShotInference,ZeroShotInference('google/flan-t5-base'))

        inject.configure(my_config)
