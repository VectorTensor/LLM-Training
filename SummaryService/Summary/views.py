from django.shortcuts import render
from django.http import JsonResponse
from rest_framework import viewsets
from rest_framework.views import APIView
from .Services.Inference.ZeroShot import ZeroShotInference
import inject


# Create your views here.

class ZeroshotView(APIView):

    @inject.params(zero_shot = ZeroShotInference)
    def get(self,request,zero_shot:ZeroShotInference):
        x = zero_shot.generate('this is it')
        res = {
            'message': x
        }
        return JsonResponse(res)
