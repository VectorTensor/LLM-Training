from rest_framework import routers
from rest_framework.routers import DefaultRouter
from django.urls import path

from .views import ZeroshotView,OneshotView,FewshotView


class CustomRouter(DefaultRouter):
    def get_urls(self):
        urls = super().get_urls()
        urls += [
            path(r'v1/zeroshot', ZeroshotView.as_view(), name='custom-api'),
            path(r'v1/oneshot',OneshotView.as_view()),

            path(r'v1/fewshot',FewshotView.as_view())
        ]
        return urls


router = CustomRouter()
