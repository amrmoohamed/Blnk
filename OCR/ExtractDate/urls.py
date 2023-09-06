from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('result/<int:day>/<int:month>/<int:year>/<str:day_arabic>/<str:month_arabic>/<str:year_arabic>', views.result, name='result'),
    
]
