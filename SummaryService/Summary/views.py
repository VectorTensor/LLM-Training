from django.shortcuts import render
from django.http import JsonResponse
from rest_framework import viewsets
from rest_framework.views import APIView
from .Services.Inference.ZeroShot import ZeroShotInference
from .Services.Inference.Inference import OneShot ,FewShot
import inject
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication

# Create your views here.

class ZeroshotView(APIView):
    permission_classes = [IsAuthenticated,]
    authentication_classes = (JWTAuthentication,)
    @inject.params(zero_shot=ZeroShotInference)
    def get(self,request,zero_shot:ZeroShotInference):
        print(request.data.get('text'))
        x = zero_shot.generate(request.data.get('text'))
        res = {
            'summary': x
        }
        return JsonResponse(res)


class OneshotView(APIView):
    permission_classes = [IsAuthenticated,]
    authentication_classes = (JWTAuthentication,)
    @inject.params(one_shot=OneShot)
    def get(self,request,one_shot:OneShot):
        print(request.data.get('text'))
        x = one_shot.generate_one_shot_inferece(request.data.get('text'))
        res = {
            'summary': x
        }
        return JsonResponse(res)


class FewshotView(APIView):
    permission_classes = [IsAuthenticated,]
    authentication_classes = (JWTAuthentication,)
    @inject.params(few_shot=FewShot)
    def get(self,request,few_shot:FewShot):
        print(request.data.get('text'))
        x = few_shot.generate_few_shot_inferece(request.data.get('text'))
        res = {
            'summary': x
        }
        return JsonResponse(res)
