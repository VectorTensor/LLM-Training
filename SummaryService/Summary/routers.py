from rest_framework import routers
from rest_framework.routers import DefaultRouter
from django.urls import path

from .views import ZeroshotView


class CustomRouter(DefaultRouter):
    def get_urls(self):
        urls = super().get_urls()
        urls += [
            path(r'v1/zeroshot', ZeroshotView.as_view(), name='custom-api'),
        ]
        return urls


router = CustomRouter()
