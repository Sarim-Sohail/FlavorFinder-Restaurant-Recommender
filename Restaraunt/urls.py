from django.contrib import admin
from django.urls import path,include
from Restaraunt import views

urlpatterns = [
path('home', views.index, name='home'),
path('recommendations', views.recommendations, name='recommendations'),
]