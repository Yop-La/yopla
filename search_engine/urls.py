from django.urls import path

from . import views


urlpatterns = [

path('', views.home),
path('ajax/search/', views.search, name='search')

]
