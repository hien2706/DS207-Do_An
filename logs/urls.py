from django.urls import path
from .views import predict_loan_status, log_api

urlpatterns = [
    path('predict/', predict_loan_status, name='predict_loan_status'),
    path('log/', log_api, name='log_api'),
]