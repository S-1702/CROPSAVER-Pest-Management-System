"""
URL configuration for pest_detection project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from members import views


urlpatterns = [
    path('admin/', admin.site.urls),
    # path('',views.index,name ="Welcome"),
    # path('upload/', views.upload_image, name='upload_image'),
    # path('detect',views.detect,name="Detecion Section"),
    # path('index',views.index,name="INDEX")
     path('',views.main,name="Index"),
    path('login/welcome',views.welcome,name="welcome"),
    path('login/signup/',views.register),
     path('welcome',views.welcome,name="welcome"),
    path('signup/',views.register,name = "Register"),
    path('send_otp',views.send_otp,name="Verification"),
    path('verfy_it',views.verify_it,name="Verification"),
    path("login/detect",views.detect,name="Detection"),
     path('login/',views.index,name="Login"),
     path("detect",views.detect,name="Detection"),
]
