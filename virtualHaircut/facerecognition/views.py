from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from . import main_ai

def home(request):
    if request.method == 'POST' and request.FILES['myfile']:
        main_ai.delete_images()
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save('facerecognition/static/images/1.jpg', myfile)
        uploaded_file_url = fs.url('images/'+myfile.name)
        print(uploaded_file_url)
        main_ai.run_landmark_detection()
        return render(request, 'home.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'home.html')