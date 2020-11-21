from django.urls import path
from . import views
 
app_name = 'audio_analysis'
urlpatterns = [
    path('',views.api.as_view()),
]
