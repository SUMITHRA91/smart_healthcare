from django.db import models

from django.contrib.auth.models import User
from django.db import models

class HealthHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    disease = models.CharField(max_length=100)
    risk = models.FloatField()
    date = models.DateTimeField(auto_now_add=True)


# Create your models here.
