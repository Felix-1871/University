from django.db import models

# Create your models here.
class Image(models.Model):
    id = models.AutoField(primary_key=True)
    image = models.ImageField(upload_to='images/')
    label = models.CharField(max_length=200)


    def __str__(self):
        return self.label
    
    