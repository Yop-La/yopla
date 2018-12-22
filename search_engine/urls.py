from django.urls import path

from . import views


urlpatterns = [

path('', views.home),
path('notebook1', views.notebook1),
path('notebook2', views.notebook2),
path('notebook3', views.notebook3),
path('ajax/search/', views.search, name='search')

]
