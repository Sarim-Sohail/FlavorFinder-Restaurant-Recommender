from traceback import print_tb
from django.shortcuts import render,redirect
from django.template import loader
from django.http import HttpResponse,HttpResponseRedirect
import os
from django.db import connection
from soupsieve import select
from Restaraunt.models import Restaurant

countries = Restaurant.objects.values('country').distinct()
country = []
for i in countries:
    #print(i['country'])
    country.append(i['country'])

currencies = Restaurant.objects.values('currency').distinct()
currency = []
for i in currencies:
    #print(i['currency'])
    currency.append(i['currency'])

def index(request):
    
    if request.method=='POST':
        # print("Lol---------------------------")
        predict=request.POST.get("predict")
        # print(predict)
        
        os.system(f"py models/main.py {predict}")
        file = open("models/temp.txt", "r")
        predictions = file.read()
        predictions=predictions.strip()
        predictions=predictions.capitalize()
        # print("-------data-------",data)
        file.close()
        return render(request,'index.html',{"data":predictions,'country':country,'currency':currency})
    else:
        return render(request,'index.html',{"data":" ",'country':country,'currency':currency})

def recommendations(request):
    if request.method=='POST':
        # print("")
        selectcountry = request.POST.get("selectcountry")
        selectcost = request.POST.get("selectcost")
        selectcurrency = request.POST.get("selectcurrency")
        # print(selectcountry)
        # print(selectcost)
        # print(selectcurrency)
        recommendations=request.POST.get("recommendations")
        
        # test=Restaurant.objects.filter(cuisine=recommendations.capitalize())
        #Rating
        if selectcountry=='Country' and selectcost=='Cost' and selectcurrency=='Currency':
            test=Restaurant.objects.filter(cuisine=recommendations.capitalize()).order_by("rating").reverse()
        #Only country
        elif selectcountry!='Country' and selectcost=='Cost' and selectcurrency=='Currency':
            test=Restaurant.objects.filter(country=selectcountry).filter(cuisine=recommendations.capitalize()).order_by("rating").reverse()
        #Only currency
        elif selectcountry=='Country' and selectcost=='Cost' and selectcurrency!='Currency':
            test=Restaurant.objects.filter(currency=selectcurrency).filter(cuisine=recommendations.capitalize()).order_by("rating").reverse()
        #Only cost   
        elif selectcountry=='Country' and selectcost=='Highest to Lowest' and selectcurrency=='Currency':
            test=Restaurant.objects.filter(cuisine=recommendations.capitalize()).order_by("cost").reverse()
        elif selectcountry=='Country' and selectcost=='Lowest to Highest' and selectcurrency=='Currency':
            test=Restaurant.objects.filter(cuisine=recommendations.capitalize()).order_by("cost")
        #Only cost and currency
        elif selectcountry=='Country' and selectcost=='Highest to Lowest' and selectcurrency!='Currency':
            test=Restaurant.objects.filter(currency=selectcurrency).filter(cuisine=recommendations.capitalize()).order_by("cost").reverse()
        elif selectcountry=='Country' and selectcost=='Lowest to Highest' and selectcurrency!='Currency':
            test=Restaurant.objects.filter(currency=selectcurrency).filter(cuisine=recommendations.capitalize()).order_by("cost")
        #Only currency and cost and country
        elif selectcountry!='Country' and selectcost=='Highest to Lowest' and selectcurrency!='Currency':
            test=Restaurant.objects.filter(currency=selectcurrency,cuisine=recommendations.capitalize(),country=selectcountry).order_by("cost").reverse()
        elif selectcountry!='Country' and selectcost=='Lowest to Highest' and selectcurrency!='Currency':
            test=Restaurant.objects.filter(currency=selectcurrency,cuisine=recommendations.capitalize(),country=selectcountry).order_by("cost")
        #Only country and cost
        elif selectcountry!='Country' and selectcost=='Highest to Lowest' and selectcurrency=='Currency':
            test=Restaurant.objects.filter(cuisine=recommendations.capitalize(),country=selectcountry).order_by("cost").reverse()
        elif selectcountry!='Country' and selectcost=='Lowest to Highest' and selectcurrency=='Currency':
            test=Restaurant.objects.filter(cuisine=recommendations.capitalize(),country=selectcountry).order_by("cost")
        #only country and currency
        elif selectcountry!='Country' and selectcost=='Cost' and selectcurrency!='Currency':
            test=Restaurant.objects.filter(cuisine=recommendations.capitalize(),country=selectcountry,currency=selectcurrency).order_by("rating").reverse()
        
        
        

        # print(countries)

        # with connection.cursor() as cursor:
        #     cursor.execute("SELECT * FROM public.\"Restaraunt_restaurant\"(CNIC_id,msg,status,dateTime) VALUES (%s,%s,%s,%s)",(CNIC,resp,"server",dateTime))
        #     cursor.close()
        # test=Restaurant.objects.filter(cuisine=recommendations.capitalize()).order_by("rating").reverse()
        test=test[0:15]
        # for i in test.values():
        #     print(i)
        
        return render(request,'index.html',{"res":test.values(),'country':country,'currency':currency})
    return render(request,'index.html',{"res":" ",'country':country,'currency':currency})

    
    

