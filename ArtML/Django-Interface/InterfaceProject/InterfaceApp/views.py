
from .forms import ImageForm
from django.shortcuts import render, HttpResponse
from .models import Image
# Create your views here.

def upload(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return HttpResponse('Image Uploaded')
    else:
        form = ImageForm()
    return render(request, 'InterfaceApp/upload.html', {'form': form})

def delete(request, id):
    if request.method == 'POST':
        pi = Image.objects.get(pk=id)
        pi.delete()
        return HttpResponse('Image Deleted')
    else:
        return HttpResponse('Failed to Delete Image')
    
def edit(request, id):
    if request.method == 'POST':
        pi = Image.objects.get(pk=id)
        form = ImageForm(request.POST, request.FILES, instance=pi)
        if form.is_valid():
            form.save()
            return HttpResponse('Image Edited')
    else:
        pi = Image.objects.get(pk=id)
        form = ImageForm(instance=pi)
    return render(request, 'InterfaceApp/edit.html', {'form': form})


def index(request, id):
    if request.method == 'POST':
        pi = Image.objects.get(pk=id)
        form = ImageForm(request.POST, request.FILES, instance=pi)
        if form.is_valid():
            form.save()
            return HttpResponse('Image Edited')
    else:
        pi = Image.objects.get(pk=id)
        form = ImageForm(instance=pi)
    return render(request, 'InterfaceApp/index.html', {'form': form})




