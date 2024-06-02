from django.shortcuts import redirect, render
from .forms import UploadImageForm
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from django.core.files.storage import FileSystemStorage
import os
from django.conf import settings

from pyexpat.errors import messages
from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib import messages
from requests import request
from django.contrib.auth import authenticate
from django.contrib.auth import authenticate,login,logout,update_session_auth_hash
from django.contrib.auth.forms import UserCreationForm,SetPasswordForm
from django.contrib.auth import login
from django.views.decorators.csrf import csrf_protect
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders 
import random

import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Create your views here.

#success_user = User.objects.create_user(account['user'],account['password'],account['email'],account['mobile'])
#Credential Accounts

account={}
otp_number = str(random.randint(100000, 999999))
detection ={}



from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.contrib import messages

def main(request):
    return render(request,"main.html")


def index(request):
    # If the login was unsuccessful or it's not a POST request, render the login page
    return render(request, 'login.html')


@csrf_protect   
def welcome(request):
    if request.method=='POST':
        username=request.POST.get('username')
        password=request.POST.get('password')
            
        user=authenticate(username=username,password=password)
        print(username,password)
        if user is not None:
           login(request,user)
           messages.success(request,"Welcome,You are Successfully Logged in!!!")
           return render(request,"index.html")
        else:
            messages.error(request,"Username or Password is incorrect.Please try again..")
            return render(request,"error.html")
    
    return render(request,"index.html")

# Creating a Account
def register(request):
            
 return render(request,"signup.html")
        
        # Now Adding Some Conditions




def send_otp(request):
    if request.method == 'POST':

        account['user'] = request.POST.get("username")
        account['email']  = request.POST.get("email")
        account['mobile'] = request.POST.get("mobile")
        account['password'] = request.POST.get("password")
        account['repassword'] = request.POST.get("confirmPassword")
        account['method'] = request.POST.get('Verification')

        credential = {'name':account['user'],'email':account['email'],'mobile':account['mobile'],'password':account['password'],'repassword':account['repassword'],'method':account['method']}
        # Open the file in write mode
        with open('credential.txt', 'w') as file:
        # Write the content to the file
            file.write(str(credential))
        
        if account['method'] == 'email':
            # Your email credentials
            fromaddr = "anakeerth00@gmail.com"
            toaddr = request.POST.get("email")
            smtp_password = "ynjy hqya srqz vthz"

            # Create a MIMEMultipart object
            msg = MIMEMultipart()

            # Set the sender and recipient email addresses
            msg['From'] = fromaddr
            msg['To'] = toaddr
            
            # Set the subject
            msg['Subject'] = "CropSaver Otp Verification"

            # Set the email body
            body = f"Your OTP is: {otp_number}"
            msg.attach(MIMEText(body, 'plain'))

            try:
                # Connect to the SMTP server
                with smtplib.SMTP('smtp.gmail.com', 587) as server:
                    # Start TLS for security
                    server.starttls()

                    # Log in to the email account
                    server.login(fromaddr, smtp_password)

                    # Send the email
                    server.sendmail(fromaddr, toaddr, msg.as_string())

                # Email sent successfully, render a template
                return render(request, 'verification_otp.html')

            except Exception as e:
                # An error occurred while sending email, redirect with an error message
                messages.error(request, f"Error sending OTP email: {e}")
                return render(request,'signup.html')  # You need to replace 'verify_it' with the appropriate URL name
        else:
            # Invalid method, redirect with an error message
            messages.error(request, "Invalid verification method")
            return render(request,'signup.html')  # You need to replace 'verify_it' with the appropriate URL name

    # If the request method is not POST, redirect with an error message
    messages.error(request, "Invalid request method")
    return render(request,'signup.html') # You need to replace 'verify_it' with the appropriate URL name


def verify_it(request):
    
    if request.method=="POST":


       

        verifi_otp1 = request.POST.get("otp1")
        verifi_otp2 = request.POST.get("otp2")
        verifi_otp3 = request.POST.get("otp3")
        verifi_otp4 = request.POST.get("otp4")
        verifi_otp5 = request.POST.get("otp5")
        verifi_otp6 = request.POST.get("otp6")

        six_digits=f"{verifi_otp1}{verifi_otp2}{verifi_otp3}{verifi_otp4}{verifi_otp5}{verifi_otp6}"
        if six_digits==otp_number:

         my_user=User.objects.create_user(account['user'],account['email'],account['password'])
         my_user.save() 
         messages.success(request,"Your account has been Created Successfully!!!")
         redirect(index)


        # else:
        #     messages.success(request,"Registration Failed!!")
        #     return render(request, 'success.html',six_digits)
        
    return render(request,"index.html")  






def upload_image(request):
    
    if request.method == 'POST':
     uploaded_file = request.FILES['image']
     fs = FileSystemStorage()
     filename = fs.save(uploaded_file.name, uploaded_file)
     uploaded_file_url = fs.url(filename)

        # Path to the uploaded image
     image_file = os.path.join(settings.MEDIA_ROOT, filename)
        

from django.shortcuts import render
from django.http import HttpResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

from django.shortcuts import render
from django.http import HttpResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from django.shortcuts import render
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import ssl

from django.shortcuts import render
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import ssl

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import ssl
import torch
from torchvision import models, transforms
from PIL import Image
import requests

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import ssl
import torch
from torchvision import models, transforms
from PIL import Image
import requests

def detect(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        uploaded_file_url = fs.url(filename)

        # Path to the uploaded image
        image_file = os.path.join(settings.MEDIA_ROOT, filename)
        
        # Ignore SSL certificate verification
        ssl._create_default_https_context = ssl._create_unverified_context

        # Define transformations to be applied to the input image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load the pre-trained SqueezeNet model
        state_dict = torch.load('pest_detection/modseq_pest_detection_model.pth', map_location='cpu')
        model = models.squeezenet1_1(pretrained=False)
        model.classifier[1] = nn.Conv2d(512,20, kernel_size=(1, 1), stride=(1, 1))
        model.load_state_dict(state_dict)
        #labels = ['aphids','armyworm','beetle','bollworm','grasshopper','mites','mosquito','sawfly','stem_borer']
        labels=['Brown Marmorated Stink Bugs','Fall Armyworms','Spider Mites','Western Corn Rootworms','ants','aphids','armyworm','bees','beetle','bollworm','catterpillar','earwig','grasshopper','mites','mosquito','moth','sawfly','slug','snail','weevil']
        model.eval()
        
        
        # Load and preprocess the image
        image = Image.open(image_file)
        image = transform(image).unsqueeze(0)

        # Perform the inference
        with torch.no_grad():
            output = model(image)

        # Get the predicted class
        _, predicted = output.max(1)

        # Retrieve the class label
        predicted_label = labels[predicted.item()]
        
        shop_links = [
        "https://www.google.com/maps/place/VENKATESWARA+AGRO+CORPORATION/@13.0506277,80.2011048,17z/data=!3m1!4b1!4m6!3m5!1s0x3a5266c5d434d4c5:0xa6bf24716f6292b6!8m2!3d13.0506277!4d80.2011048!16s%2Fg%2F1ttp8qyt?entry=ttu",
         "https://www.google.com/maps/dir//Omega+Agro+Service+Center,+No.+1%2F2,+2nd+St,+Sivananda+Nagar,+Ambattur,+Chennai,+Tamil+Nadu+600053/data=!4m6!4m5!1m1!4e2!1m2!1m1!1s0x3a5263864a56f54b:0xab295a5fca955df4?sa=X&ved=2ahUKEwjAr_6l-OOEAxWT3jgGHYrwCp4Q48ADegQIARAA",
        "https://www.google.com/maps/dir//Sundaram+agro+Centre,+5%2F15+arcot+road,+1st+Cross+St,+Karambakkam,+Chennai,+Tamil+Nadu+600116/data=!4m6!4m5!1m1!4e2!1m2!1m1!1s0x3a52611c668a3d99:0x40977c76a2fc0d3a?sa=X&ved=2ahUKEwjAr_6l-OOEAxWT3jgGHYrwCp4Q48ADegQIDhAA",
        "https://www.google.com/maps/dir//New+No.83,+Old,+Deeptha+Agro+Chemicals,+44,+Santhome+High+Rd,+Mylapore,+Chennai,+Tamil+Nadu+600004/data=!4m6!4m5!1m1!4e2!1m2!1m1!1s0x3a52662d89f16de3:0xfbeb0fb37a601cf9?sa=X&ved=2ahUKEwjAr_6l-OOEAxWT3jgGHYrwCp4Q48ADegQIDBAA",
        "https://www.google.com/maps/dir/13.0449408,80.19968/Shop+No.35,+Pest+'+N'+Green+Care+Agro+Services,+L.M:MARUNDHEESWARAR+TEMPLE+NEXT+SIGNAL,+7,+SH+49,+Thiruvanmiyur,+Chennai,+Tamil+Nadu+600041/@13.0141686,80.1884229,13z/data=!3m1!4b1!4m9!4m8!1m1!4e1!1m5!1m1!1s0x3a525d5c0cda78df:0xb9555f18b028758d!2m2!1d80.2595754!2d12.983381?entry=ttu"
        # Add more shop URLs as needed
    ]  

        context = {'corresponding_disease': predicted_label,'shop_linkss':shop_links}

        # Define an array of shop URLs
      
        # Render the template with the context
        return render(request, 'detect.html', context)

    return render(request, 'detect.html')
