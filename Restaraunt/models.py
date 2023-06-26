from django.db import models

# Create your models here.
class Restaurant(models.Model):

    resid=models.BigIntegerField()
    cuisine=models.CharField(max_length=20)
    name=models.CharField(max_length=100)
    country=models.CharField(max_length=50)
    city=models.CharField(max_length=50)
    address=models.CharField(max_length=200)
    cost=models.IntegerField()
    currency=models.CharField(max_length=25)
    book=models.CharField(max_length=5)
    delivery=models.CharField(max_length=5)
    rating=models.FloatField()
    reviews=models.IntegerField()
    link=models.CharField(max_length=50)