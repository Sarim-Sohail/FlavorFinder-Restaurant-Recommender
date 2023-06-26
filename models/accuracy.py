import json
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
from os.path import exists
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import linear_model
import string
from sklearn import svm
from pathlib import Path
import tkinter as tk
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
import nltk
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# from sklearn import preprocessing
# import numpy as np
# from sklearn.impute import SimpleImputer
# import numpy as np
# from itertools import chain

#model save files
knnfile = exists('models/Trained Models/KNN_Model.pkl')
linearsvmfile = exists('models/Trained Models/Linear_SVM_Model.pkl')
crammersvmfile = exists('models/Trained Models/Crammer_SVM_Model.pkl')
mtnbfile = exists('models/Trained Models/MultinomialNB_Model.pkl')
lrfile = exists('models/Trained Models/Logistic_Regression_Model.pkl')
bernoulinbfile = exists('models/Trained Models/BernouliNB_Model.pkl')
sgdfile = exists('models/Trained Models/SGD_Model.pkl')

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")

def preprocess_userinput(userinput):
    # Remove punctuations
    for character in string.punctuation:
        userinput = userinput.replace(character, ' ')
    userinput.replace("  ", " ")

    # Remove Digits
    userinput = re.sub(r"(\d)", "", userinput)

    # Remove content inside paranthesis
    userinput = re.sub(r'\([^)]*\)', '', userinput)

    # Remove Brand Name
    userinput = re.sub(u'\w*\u2122', '', userinput)

    # Convert to lowercase
    userinput = userinput.lower()

    # Remove Stop Words
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(userinput)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    userinput = ' '.join(filtered_sentence)

    ps = PorterStemmer()
    userinput = word_tokenize(userinput)
    word_ps = []
    for w in userinput:
        word_ps.append(ps.stem(w))
    userinput = ' '.join(word_ps)
    print(userinput)
    return userinput

def knn_prediction(userfeatures,predictions, X_test, y_test):
    # KNN classifier with K-value = 11
    with open('models/Trained Models/KNN_Model.pkl', 'rb') as f:
        neigh = pickle.load(f)
    prediction = neigh.predict(userfeatures)
    y_pred = neigh.predict(X_test)
    print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:11")
    print("KNN Prediction: ", prediction)
    predictions.append(str(prediction))
    return predictions

def linear_svm_prediction(userfeatures,predictions, X_test, y_test):
    # OVA SVM(Grid Search Results: Kernel - Linear, C - 1, Gamma - Auto)
    with open('models/Trained Models/Linear_SVM_Model.pkl', 'rb') as f:
        lin_clf = pickle.load(f)
    prediction = lin_clf.predict(userfeatures)
    y_pred = lin_clf.predict(X_test)
    print("Accuracy of OVA SVM is ", accuracy_score(y_test, y_pred) * 100)
    print("OVA SVM Prediction: ", prediction)
    predictions.append(str(prediction))
    return predictions

def crammer_svm_prediction(userfeatures,predictions, X_test, y_test):
    # SVM by Crammer(Grid Search Results: Gamma - , C - )
    with open('models/Trained Models/Crammer_SVM_Model.pkl', 'rb') as f:
        lin_clf = pickle.load(f)
    prediction = lin_clf.predict(userfeatures)
    y_pred = lin_clf.predict(X_test)
    print("Accuracy of Crammer SVM is ", accuracy_score(y_test, y_pred) * 100)
    print("Crammer SVM Prediction: ", prediction)
    predictions.append(str(prediction))
    return predictions

def multinomialnb_prediction(userfeatures,predictions, X_test, y_test):
    # Implementing OVA Naive Bayes
    with open('models/Trained Models/MultinomialNB_Model.pkl', 'rb') as f:
        clf = pickle.load(f)
    prediction = clf.predict(userfeatures)
    y_pred = clf.predict(X_test)
    print("Accuracy of OVA Naive Bayes is ", accuracy_score(y_test, y_pred) * 100)
    print("OVA Naive Bayes Prediction: ", prediction)
    predictions.append(str(prediction))
    return predictions

def logistic_regression_prediction(userfeatures,predictions, X_test, y_test):
    # Implementing OVA Logistic Regerssion
    with open('models/Trained Models/Logistic_Regression_Model.pkl', 'rb') as f:
        logisticRegr = pickle.load(f)
    prediction = logisticRegr.predict(userfeatures)
    y_pred = logisticRegr.predict(X_test)
    print("Accuracy of OVA Logistic Regression is ", accuracy_score(y_test, y_pred) * 100)
    print("OVA Logistic Regerssion Prediction: ", prediction)
    predictions.append(str(prediction))
    return predictions

def bernoulinb_prediction(userfeatures,predictions, X_test, y_test):
    # Implementing OVA Bernouli Naive Bayes
    with open('models/Trained Models/BernouliNB_Model.pkl', 'rb') as f:
        clf = pickle.load(f)
    prediction = clf.predict(userfeatures)
    y_pred = clf.predict(X_test)
    print("Accuracy of Bernoulli Naive Bayes is ", accuracy_score(y_test, y_pred) * 100)
    print("Bernouli Naive Bayes Prediction: ", prediction)
    predictions.append(str(prediction))
    return predictions

def sgd_prediction(userfeatures,predictions, X_test, y_test):
    # Implmenting SGD classifier
    with open('models/Trained Models/SGD_Model.pkl', 'rb') as f:
        clf=pickle.load(f)
    prediction = clf.predict(userfeatures)
    y_pred = clf.predict(X_test)
    print("Accuracy of SGD is ", accuracy_score(y_test, y_pred) * 100)
    print("SGD Prediction: ", prediction)
    predictions.append(str(prediction))
    return predictions

def get_restaurants(predicted_cuisine):
    data = pd.read_csv(r'models/updated_zomato.csv')
    data = data.drop(data.columns[0], axis=1)

    cuisine = str(predicted_cuisine)

    index = data.index
    restraunts = []

    for i in index:
        templist = str(data.iloc[i]['Cuisines']).lower().split(', ')
        if cuisine in templist:
            restraunts.append(i)

    predicted_restraunts = []

    for res in restraunts:
        empty_dict = {}
        url = 'http://www.zoma.to/r/'
        resID = str(data.iloc[res]['Restaurant ID'])
        url += resID
        empty_dict['Restaurant ID'] = data.iloc[res]['Restaurant ID']
        empty_dict['Restaurant Name'] = data.iloc[res]['Restaurant Name']
        empty_dict['Country'] = data.iloc[res]['Country']
        empty_dict['City'] = data.iloc[res]['City']
        empty_dict['Address'] = data.iloc[res]['Address']
        empty_dict['Cuisines'] = data.iloc[res]['Cuisines']
        empty_dict['Average Cost for two'] = data.iloc[res]['Average Cost for two']
        empty_dict['Currency'] = data.iloc[res]['Currency']
        empty_dict['Has Table booking'] = data.iloc[res]['Has Table booking']
        empty_dict['Has Online delivery'] = data.iloc[res]['Has Online delivery']
        empty_dict['Aggregate rating'] = data.iloc[res]['Aggregate rating']
        empty_dict['Votes'] = data.iloc[res]['Votes']
        empty_dict['Website'] = url
        predicted_restraunts.append(empty_dict)

    return predicted_restraunts

def rating_sort(predicted_restraunts):
    id_ratings = {}

    for restraunt in predicted_restraunts:
        resID = restraunt['Restaurant ID']
        rating = restraunt['Aggregate rating']
        id_ratings[resID] = rating

    sorted_id_ratings = sorted(id_ratings.items(), key=lambda x: x[1], reverse=True)

    toprestrauntids = []
    count = 1
    for i in sorted_id_ratings:
        if count <= 10:
            toprestrauntids.append(i[0])
        count += 1

    toprestraunts = []
    for restraunt in toprestrauntids:
        for res in predicted_restraunts:
            if restraunt == res['Restaurant ID']:
                toprestraunts.append(res)

    return toprestraunts

def average_cost_sort(predicted_restraunts):
    id_costs = {}
    for restraunt in predicted_restraunts:
        resID = restraunt['Restaurant ID']
        cost = restraunt['Average Cost for two']
        id_costs[resID] = cost

    sorted_id_costs = sorted(id_costs.items(), key=lambda x: x[1])
    toprestrauntcostids = []
    count = 1
    for i in sorted_id_costs:
        if count <= 10:
            toprestrauntcostids.append(i[0])
        count += 1

    topcostrestraunts = []
    for restraunt in toprestrauntcostids:
        for res in predicted_restraunts:
            if restraunt == res['Restaurant ID']:
                topcostrestraunts.append(res)

    return topcostrestraunts

def booking_sort(predicted_restraunts,cuisine):
    bookingrestraunts = []
    for restraunt in predicted_restraunts:
        resID = restraunt['Restaurant ID']
        hasbooking = restraunt['Has Table booking']
        if hasbooking == 'Yes':
            bookingrestraunts.append(restraunt)

    if len(bookingrestraunts) == 0:
        print("There are no restaraunts of the", cuisine, "cuisine that has table booking")
    else:

        toprestraunts = rating_sort(bookingrestraunts)
        return toprestraunts

def delivery_sort(predicted_restraunts,cuisine):
    deliveryrestraunts = []
    for restraunt in predicted_restraunts:
        resID = restraunt['Restaurant ID']
        hasdelivery = restraunt['Has Online delivery']
        if hasdelivery == 'Yes':
            deliveryrestraunts.append(restraunt)

    if len(deliveryrestraunts) == 0:
        print("There are no restaraunts of the", cuisine, "cuisine that has online delivery ")
    else:
        toprestraunts = rating_sort(deliveryrestraunts)
        return toprestraunts

def country_sort(predicted_restraunts,cuisine,country):
    countryrestraunts = []
    for restraunt in predicted_restraunts:
        resID = restraunt['Restaurant ID']
        rescountry = restraunt['Country']
        if rescountry == country:
            countryrestraunts.append(restraunt)

    if len(countryrestraunts) == 0:
        print("There are no restraunts of the", cuisine, "cuisine in", country)
    else:
        toprestraunts = rating_sort(countryrestraunts)
        return toprestraunts

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def Predict(ingredients):
    predictions = []
    vectorizer = TfidfVectorizer()
    data = pd.read_csv(r'models/zomato_model.csv')
    X = vectorizer.fit_transform(data['ing_mod'])
    Y=[]
    Y = data['cuisine']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100)
    

    # userinput = 'tumeric,olive,oil,lemon,saffron,tomato,paste,pepper,flour,chickpeas,chicken,stock,warm,water,fresh,ginger,salt,tomatoes,fresh,cilantro,shallots,fresh,parsley'

    userinput = preprocess_userinput(ingredients)

    userfeatures = vectorizer.transform([userinput])

    predictions = knn_prediction(userfeatures, predictions, X_test, y_test)
    predictions = linear_svm_prediction(userfeatures, predictions, X_test, y_test)
    predictions = crammer_svm_prediction(userfeatures, predictions, X_test, y_test)
    predictions = multinomialnb_prediction(userfeatures, predictions, X_test, y_test)
    predictions = logistic_regression_prediction(userfeatures, predictions, X_test, y_test)
    predictions = bernoulinb_prediction(userfeatures, predictions, X_test, y_test)
    predictions = sgd_prediction(userfeatures, predictions, X_test, y_test)

    predictions = ' '.join(predictions)
    predictions = re.sub(r'[^\w\s]', '', predictions)
    predictions = predictions.split(' ')
    frequency_distribution = nltk.FreqDist(predictions)
    predicted_cuisine = frequency_distribution.max()

    print("Top prediction : ", predicted_cuisine)
    file = open("models/temp.txt", "w")
    file.write(predicted_cuisine)
    # print("-------data-------",data)
    file.close()
    # cuz.insert(tk.END, str(predicted_cuisine))

def Recommend(predicted_cuisine):

    predicted_restraunts = get_restaurants(predicted_cuisine.lower())

    toprestraunts = rating_sort(predicted_restraunts)
    #toprestraunts = average_cost_sort(predicted_restraunts)
    #toprestraunts = booking_sort(predicted_restraunts,predicted_cuisine)
    #toprestraunts = delivery_sort(predicted_restraunts,predicted_cuisine)
    # country = 'United States'
    # toprestraunts = country_sort(predicted_restraunts,predicted_cuisine,country)

    # RecommendationDisplay(toprestraunts)
    # for i in toprestraunts:
    #         for key, value in i.items():
    #             print(key, ' : ', value)
    #         print('\n')

if not knnfile or not linearsvmfile or not crammersvmfile or not mtnbfile or not lrfile or not bernoulinbfile or not sgdfile:

    ps = PorterStemmer()
    file = 'Datasets/train.json'
    with open(file) as train_file:
        dict_train = json.load(train_file)

    id_ = []
    cuisine = []
    ingredients = []
    for i in range(len(dict_train)):
        id_.append(dict_train[i]['id'])
        cuisine.append(dict_train[i]['cuisine'])
        ingredients.append(dict_train[i]['ingredients'])

    import pandas as pd
    df = pd.DataFrame({'id':id_,
                       'cuisine':cuisine,
                       'ingredients':ingredients})

    new = []
    for s in df['ingredients']:
        s = ' '.join(s)
        new.append(s)

    df['ing'] = new

    l = []
    for s in df['ing']:
        # Remove punctuations
        s = re.sub(r'[^\w\s]', '', s)

        # Remove Digits
        s = re.sub(r"(\d)", "", s)

        # Remove content inside paranthesis
        s = re.sub(r'\([^)]*\)', '', s)

        # Remove Brand Name
        s = re.sub(u'\w*\u2122', '', s)

        # Convert to lowercase
        s = s.lower()

        # Remove Stop Words
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(s)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        filtered_sentence = []
        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)
        s = ' '.join(filtered_sentence)

        # Remove low-content adjectives

        # Porter Stemmer Algorithm
        words = word_tokenize(s)
        word_ps = []
        for w in words:
            word_ps.append(ps.stem(w))
        s = ' '.join(word_ps)

        l.append(s)
    df['ing_mod'] = l

    df.to_csv('models/zomato_model.csv')

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['ing_mod'])
    print(X.shape())
    cuisine_map={'0':'brazilian', '1':'british', '2':'cajun_creole', '3':'chinese', '4':'filipino', '5':'french', '6':'greek', '7':'indian', '8':'irish', '9':'italian', '10':'jamaican', '11':'japanese', '12':'korean', '13':'mexican', '14':'moroccan', '15':'russian', '16':'southern_us', '17':'spanish', '18':'thai', '19':'vietnamese'}

    Y=[]
    Y = df['cuisine']

    userfeatures = vectorizer.transform(["plain flour, ground pepper, broiler-fryer chicken, tomatoes, ground black zesty italian dressing, thyme, gutter green tomatoes, italian seasoning, meal milk, vegetable oil"])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100)

    predictions = []

    #KNN classifier with K-value = 11
    neigh = KNeighborsClassifier(n_neighbors = 11, weights='uniform', algorithm='auto')
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:11")
    prediction = neigh.predict(userfeatures)
    print("KNN Prediction: ", prediction)
    with open('models/Trained Models/KNN_Model.pkl', 'wb') as f:
        pickle.dump(neigh, f)
    predictions.append(str(prediction))

    # OVA SVM(Grid Search Results: Kernel - Linear, C - 1, Gamma - Auto)
    lin_clf = svm.LinearSVC(C=1)
    lin_clf.fit(X_train, y_train)
    y_pred = lin_clf.predict(X_test)
    print("Accuracy of OVA SVM is ", accuracy_score(y_test, y_pred) * 100)
    prediction = lin_clf.predict(userfeatures)
    print("OVA SVM Prediction: ", prediction)
    with open('models/Trained Models/Linear_SVM_Model.pkl', 'wb') as f:
        pickle.dump(lin_clf, f)
    predictions.append(str(prediction))

    # SVM by Crammer(Grid Search Results: Gamma - , C - )
    lin_clf = svm.LinearSVC(C=1.0, multi_class='crammer_singer')
    lin_clf.fit(X_train, y_train)
    y_pred = lin_clf.predict(X_test)
    print("Accuracy of Crammer SVM is ", accuracy_score(y_test, y_pred) * 100)
    prediction = lin_clf.predict(userfeatures)
    print("Crammer SVM Prediction: ", prediction)
    with open('models/Trained Models/Crammer_SVM_Model.pkl', 'wb') as f:
        pickle.dump(lin_clf, f)
    predictions.append(str(prediction))

    # Implementing OVA Naive Bayes
    clf = MultinomialNB().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy of OVA Naive Bayes is ", accuracy_score(y_test, y_pred) * 100)
    prediction = clf.predict(userfeatures)
    print("OVA Naive Bayes Prediction: ", prediction)
    with open('models/Trained Models/MultinomialNB_Model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    predictions.append(str(prediction))

    # Implementing OVA Logistic Regerssion
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, y_train)
    y_pred = logisticRegr.predict(X_test)
    print("Accuracy of OVA Logistic Regression is ", accuracy_score(y_test, y_pred) * 100)
    prediction = logisticRegr.predict(userfeatures)
    print("OVA Logistic Regerssion Prediction: ", prediction)
    with open('models/Trained Models/Logistic_Regression_Model.pkl', 'wb') as f:
        pickle.dump(logisticRegr, f)
    predictions.append(str(prediction))

    # Implementing OVA Bernouli Naive Bayes
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy of Bernoulli Naive Bayes is ", accuracy_score(y_test, y_pred) * 100)
    prediction = clf.predict(userfeatures)
    print("Bernouli Naive Bayes Prediction: ", prediction)
    with open('models/Trained Models/BernouliNB_Model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    predictions.append(str(prediction))

    # Implementing SGD classifier
    sgd = linear_model.SGDClassifier()
    sgd.fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    print("Accuracy of SGD is ", accuracy_score(y_test, y_pred) * 100)
    prediction = sgd.predict(userfeatures)
    print("SGD Prediction: ", prediction)
    with open('models/Trained Models/SGD_Model.pkl', 'wb') as f:
        pickle.dump(sgd, f)
    predictions.append(str(prediction))

    predictions = ' '.join(predictions)
    predictions = re.sub(r'[^\w\s]', '', predictions)
    predictions = predictions.split(' ')
    frequency_distribution = nltk.FreqDist(predictions)
    predicted_cuisine = frequency_distribution.max()
    print("Top prediction : ", predicted_cuisine)

else:
    # print(sys.argv[1])
    # Predict(sys.argv[1])
    var = "romaine lettuce,black olives,grape tomatoes,garlic,pepper,purple onion,seasoning,garbanzo beans,feta cheese crumbles"
    Predict(var)
    # window = tk.Tk()

    # ingredients = tk.StringVar()
    # prediction = tk.StringVar()

    # window.geometry("1440x1024")
    # window.configure(bg="#222831")

    # canvas = tk.Canvas(
    #     window,
    #     bg="#222831",
    #     height=1024,
    #     width=1440,
    #     bd=0,
    #     highlightthickness=0,
    #     relief="ridge"
    # )
    # canvas.place(x=0, y=0)
    # image_image_1 = PhotoImage(
    #     file=relative_to_assets("image_1.png"))
    # image_1 = canvas.create_image(
    #     720.0,
    #     97.0,
    #     image=image_image_1
    # )
    # canvas.create_text(
    #     450.0,
    #     73.0,
    #     anchor="nw",
    #     text="FLAVOR",
    #     fill="#FFD369",
    #     font=("RoadRage Regular", 40 * -1)
    # )

    # canvas.create_text(
    #     840.0,
    #     73.0,
    #     anchor="nw",
    #     text="HUNT",
    #     fill="#FFD369",
    #     font=("RoadRage Regular", 40 * -1)
    # )

    # canvas.create_text(
    #     575.0,
    #     200.0,
    #     anchor="nw",
    #     text="Enter Ingredients",
    #     fill="#FFFFFF",
    #     font=("RoadRage Regular", 40 * -1)
    # )

    # canvas.create_text(
    #     575.0,
    #     355.0,
    #     anchor="nw",
    #     text="Predicted Cuisine",
    #     fill="#FFFFFF",
    #     font=("RoadRage Regular", 40 * -1)
    # )

    # ing = tk.Text(window, font=('Bahnschrift', 15), width=75, height=3, borderwidth=4, relief="ridge", wrap=tk.WORD)
    # ing.place(x=300.0, y=260)

    # cuz = tk.Text(window, font=('Bahnschrift', 15), width=20, height=1, borderwidth=4, relief="ridge", wrap=tk.WORD)
    # cuz.place(x=610, y=410)

    # predict_button = tk.Button(window, text="Predict", height=1, width=10, font=('Bahnschrift', 12, 'bold'),
    #                            command=lambda: getIngredients())
    # predict_button.place(x=1200, y=275, height=50, width=200)

    # recommend_button = tk.Button(window, text="Recommend", height=1, width=10, font=('Bahnschrift', 12, 'bold'),
    #                              command=lambda: getCuisine())
    # recommend_button.place(x=875, y=410, height=35, width=120)

    # window.mainloop()
#-----------------------------------------------------------------------------------------------------------------------
