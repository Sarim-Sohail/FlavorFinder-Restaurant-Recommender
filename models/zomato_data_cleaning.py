import pandas as pd
import numpy as np
from itertools import chain
import json

if __name__=="__main__":
    zdata=pd.read_csv('Datasets/zomato.csv',encoding = "ISO-8859-1")
    #print(zdata)
    zdata.drop(["Latitude","Longitude","Locality","Locality Verbose","Switch to order menu","Price range"],axis=1,inplace=True)

    print(zdata)
    f=open('Datasets\\train.json')
    wdata=json.load(f)
    # print(wdata)
    # print(type(wdata))

    zallcuisines = zdata['Cuisines'].tolist()
    znewcuisines=[]
    zcuisines=[]
    # print(zallcuisines)
    for i in zallcuisines:
        # print(type(i))
        temp=str(i)
        for j in temp.split(", "):
            znewcuisines.append(j)

    # print(znewcuisines)

    for i in znewcuisines:
        if i not in zcuisines:
            zcuisines.append(i)

    wallcuisines=[d['cuisine'] for d in wdata]
    wcuisines=[]
    # print(wallcuisines)
    for i in wallcuisines:
        if i not in wcuisines:
            wcuisines.append(i)

    zcuisines=[string.lower() for string in zcuisines]

    print(sorted(zcuisines))
    print(sorted(wcuisines))

    tokeep=[]

    for i in zcuisines:
        if i in wcuisines:
            tokeep.append(i)
    # tokeep.append('cajun')
    # tokeep.append('american')
    print(sorted(tokeep))

    american = ['hawaiian', 'latin american', 'steak', 'southern', 'new american', 'western', 'south american','southwestern', 'western']
    indian = ['sri lankan', 'goan', 'kerala', 'bengali', 'lucknowi', 'chettinad', 'maharashtrian', 'andhra', 'kebab','oriya', 'street food', 'kashmiri', 'rajasthani', 'afghani', 'pakistani', 'south indian', 'mughlai','north indian', 'gujarati', 'mithai', 'biryani', 'hyderabadi', 'modern indian', 'bihari', 'curry']
    french = ['bakery', 'belgian']
    italian = ['patisserie', 'pizza', 'turkish pizza']
    chinese = ['tibet', 'cantonese', 'taiwanese']
    japanese = ['ramen', 'teriyaki', 'sushi']
    moroccan = ['middle eastern', 'lebanese', 'mediterranean']
    filipino = ['indonesian']
    spanish = ['tapas']
    irish = ['scottish']
    vietnamese = ['singaporean', 'malaysian', 'burmese']
    brazilian = ['portugese']

    index=zdata.index
    print(len(index))

    for i in index:
        templist = str(zdata.iloc[i]['Cuisines']).lower().split(', ')
        newlist=[]
        # print(i,templist)
        for j in templist:
            if j in american:
                templist = [item.replace(j, 'american') for item in templist]
            if j in indian:
                templist = [item.replace(j, 'indian') for item in templist]
            if j in french:
                templist = [item.replace(j, 'french') for item in templist]
            if j in italian:
                templist = [item.replace(j, 'italian') for item in templist]
            if j in chinese:
                templist = [item.replace(j, 'chinese') for item in templist]
            if j in japanese:
                templist = [item.replace(j, 'japanese') for item in templist]
            if j in moroccan:
                templist = [item.replace(j, 'moroccan') for item in templist]
            if j in filipino:
                templist = [item.replace(j, 'filipino') for item in templist]
            if j in spanish:
                templist = [item.replace(j, 'spanish') for item in templist]
            if j in irish:
                templist = [item.replace(j, 'irish') for item in templist]
            if j in vietnamese:
                templist = [item.replace(j, 'vietnamese') for item in templist]
            if j in brazilian:
                templist = [item.replace(j, 'brazilian') for item in templist]
        templist = ', '.join(templist)
        # print(templist)
        # print('\n')
        zdata.at[i,'Cuisines'] = templist.title()

    # for i in index:
    #     print(i, str(zdata.iloc[i]['Cuisines']))

    index = zdata.index
    print(len(index))

    for i in index:
        templist = str(zdata.iloc[i]['Cuisines']).lower().split(', ')
        # print(i, templist)
        for j in templist:
            if j not in tokeep:
                templist.remove(j)
        for j in templist:
            if j not in tokeep:
                templist.remove(j)
        templist = list(dict.fromkeys(templist))
        # print(templist)
        # print('\n')
        templist = ', '.join(templist)
        zdata.at[i, 'Cuisines'] = templist.title()

    # for i in index:
    #     print(i, str(zdata.iloc[i]['Cuisines']))

    index = zdata.index
    print(len(index))

    todrop=[]
    for i in index:
        templist = str(zdata.iloc[i]['Cuisines']).lower().split(', ')
        check = any(item in templist for item in tokeep)
        if check == False:
            # print(i)
            # print(sorted(templist))
            # print(check)
            todrop.append(i)

    print(len(todrop))
    zdata = zdata.drop(zdata.index[todrop])

    newindex = zdata.index
    print(len(newindex))

    # for i in newindex:
    #     if i==6578:
    #         break
    #     print(i, str(zdata.iloc[i]['Cuisines']))

    countrydata = pd.DataFrame(pd.read_excel("Datasets/Country-Code.xlsx"))

    code_country = {}
    index = countrydata.index
    for i in index:
        code = str(countrydata.iloc[i]['Country Code'])
        country = str(countrydata.iloc[i]['Country'])
        code_country[code] = country

    print(code_country)

    print(zdata['Country Code'].dtype)
    zdata = zdata.astype({"Country Code": str})
    print(type(zdata['Country Code'][0]))

    # index = zdata.index
    # for i in index:
    #     if i == 7863:
    #         break
    #     code = str(zdata.iloc[i]['Country Code'])
    #     for key,value in code_country.items():
    #         if code == key:
    #             #print(i,code, str(value))
    #             zdata.iloc[i]['Country Code'] = str(value)

    # for key,value in code_country.items():
    #     print(type(key),type(value))
    zdata.rename(columns={'Country Code':'Country'}, inplace=True)
    zdata['Country'].replace(code_country,inplace=True)
    # index = zdata.index
    # for i in index:
    #     print(i, str(zdata.iloc[i]['Country Code']))
    # # print(zdata['Country Code'])

    zdata.to_csv('updated_zomato.csv')

