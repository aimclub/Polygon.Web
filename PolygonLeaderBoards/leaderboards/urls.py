from django.urls import path

from leaderboards.views import index, leader_boards, upload, results, login, signup, logout, userpage, publish_results, \
    delete_publishment

urlpatterns = [
    path("", index, name="HomePage"),
    path('lboards/', leader_boards, name="LBoardsPage"),
    path('upload/', upload, name="UploadPage"),
    path('results/', results, name="ResultsPage"),
    path('publish/', publish_results, name="PublishResults"),
    path('delete/<slug:specification>/<int:user_id>', delete_publishment, name="DeletePublishment"),
    path('userpage/', userpage, name="UserPage"),
    path('login/', login, name="LoginPage"),
    path('signup/', signup, name="SignupPage"),
    path('logout/', logout, name="Logout"),
]
