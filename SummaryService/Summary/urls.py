from django.urls import path, include
from .routers import router

from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns=[
    path('api/',include(router.urls)),
    path('api/token/', TokenObtainPairView.as_view(), ),
    path('api/token/refresh/', TokenRefreshView.as_view())
]