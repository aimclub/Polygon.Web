from django.urls import path

from Scoring.views import index, FileUpload

urlpatterns = [
    path("", index, name="HomePage"),
    path('upload/', FileUpload, name="FileUpload"),
]
