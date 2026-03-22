from django.urls import path
from .views import diabetes_predict_api,heart_predict_api,liver_predict_api,generate_report,ecg_predict_api,generate_ecg_report,google_fit_data,login_user,register_user

urlpatterns = [
    path("api/diabetes/", diabetes_predict_api),
    path("api/heart/", heart_predict_api),
    path("api/liver/", liver_predict_api),
    path("api/report/", generate_report),
    path("predict-ecg/", ecg_predict_api),
    path("ecg-report/", generate_ecg_report),
    path("google-fit/", google_fit_data),
     path("api/login/", login_user),
    path("api/register/",register_user),
   
    



]
