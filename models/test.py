import json
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
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
import tkinter as tk
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import preprocessing
import numpy as np
from sklearn.impute import SimpleImputer
import nltk
import numpy as np
from itertools import chain

#model save files
knnfile = exists('Trained Models/KNN_Model.pkl')
linearsvmfile = exists('Trained Models/Linear_SVM_Model.pkl')
crammersvmfile = exists('Trained Models/Crammer_SVM_Model.pkl')
mtnbfile = exists('Trained Models/MultinomialNB_Model.pkl')
lrfile = exists('Trained Models/Logistic_Regression_Model.pkl')
bernoulinbfile = exists('Trained Models/BernouliNB_Model.pkl')
sgdfile = exists('Trained Models/SGD_Model.pkl')

predicted_cuisine = ''

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

    return userinput

def knn_prediction(userfeatures,predictions):
    # KNN classifier with K-value = 11
    with open('Trained Models/KNN_Model.pkl', 'rb') as f:
        neigh = pickle.load(f)
    prediction = neigh.predict(userfeatures)
    print("KNN Prediction: ", prediction)
    predictions.append(str(prediction))
    return predictions

def linear_svm_prediction(userfeatures,predictions):
    # OVA SVM(Grid Search Results: Kernel - Linear, C - 1, Gamma - Auto)
    with open('Trained Models/Linear_SVM_Model.pkl', 'rb') as f:
        lin_clf = pickle.load(f)
    prediction = lin_clf.predict(userfeatures)
    print("OVA SVM Prediction: ", prediction)
    predictions.append(str(prediction))
    return predictions

def crammer_svm_prediction(userfeatures,predictions):
    # SVM by Crammer(Grid Search Results: Gamma - , C - )
    with open('Trained Models/Crammer_SVM_Model.pkl', 'rb') as f:
        lin_clf = pickle.load(f)
    prediction = lin_clf.predict(userfeatures)
    print("Crammer SVM Prediction: ", prediction)
    predictions.append(str(prediction))
    return predictions

def multinomialnb_prediction(userfeatures,predictions):
    # Implementing OVA Naive Bayes
    with open('Trained Models/MultinomialNB_Model.pkl', 'rb') as f:
        clf = pickle.load(f)
    prediction = clf.predict(userfeatures)
    print("OVA Naive Bayes Prediction: ", prediction)
    predictions.append(str(prediction))
    return predictions

def logistic_regression_prediction(userfeatures,predictions):
    # Implementing OVA Logistic Regerssion
    with open('Trained Models/Logistic_Regression_Model.pkl', 'rb') as f:
        logisticRegr = pickle.load(f)
    prediction = logisticRegr.predict(userfeatures)
    print("OVA Logistic Regerssion Prediction: ", prediction)
    predictions.append(str(prediction))
    return predictions

def bernoulinb_prediction(userfeatures,predictions):
    # Implementing OVA Bernouli Naive Bayes
    with open('Trained Models/BernouliNB_Model.pkl', 'rb') as f:
        clf = pickle.load(f)
    prediction = clf.predict(userfeatures)
    print("Bernouli Naive Bayes Prediction: ", prediction)
    predictions.append(str(prediction))
    return predictions

def sgd_prediction(userfeatures,predictions):
    # Implmenting SGD classifier
    with open('Trained Models/SGD_Model.pkl', 'rb') as f:
        clf=pickle.load(f)
    prediction = clf.predict(userfeatures)
    print("SGD Prediction: ", prediction)
    predictions.append(str(prediction))
    return predictions


def get_restaurants(predicted_cuisine):
    data = pd.read_csv(r'updated_zomato.csv')
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
predicted_restraunts = get_restaurants(predicted_cuisine)

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

    for i in toprestraunts:
            for key, value in i.items():
                print(key, ' : ', value)
            print('\n')

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

#defining font and colors for GUI
font_name = "Times New Roman"
blue = "#005555"
lightblue = "#383838"
dblue = "#4D77FF"
black = "#D29D2B"
darkblue = "#051367"
yellow = "#D29D2B"
bb = "#141E27"

#creating GUI window
master_window = tk.Tk()
master_window.geometry("700x550")
master_window.configure(bg=lightblue)
master_window.title("Vector Space Model")


#the title labels for the GUI Window
label1 = tk.Label(master_window, text = "TF-IDF",fg = black, bg = lightblue, font = (font_name,20,"bold"))
label1.place(x=300,y=32)
label2 = tk.Label(master_window, text = "VECTOR SPACE MODEL",fg = black, bg = lightblue, font = (font_name,30,"bold"))
label2.place(x=120,y=60)

#the label to display enter query
label3 = tk.Label(master_window, text = "ENTER QUERY",fg = black, bg = lightblue, font = (font_name,10))
label3.place(x=85,y=200)

#a text box entry to take the query input from user
query_input = tk.Entry()
query_input.place(x=180,y=200,height= 20, width= 350)

#function to store values of query and alpha from user input
def store_query():
    q = query_input.get()
    query_processing(q)
    return q

def query_processing(userinput):
    predictions = []
    vectorizer = TfidfVectorizer()
    data = pd.read_csv(r'zomato_model.csv')
    X = vectorizer.fit_transform(data['ing_mod'])

    #userinput = 'tumeric,olive,oil,lemon,saffron,tomato,paste,pepper,flour,chickpeas,chicken,stock,warm,water,fresh,ginger,salt,tomatoes,fresh,cilantro,shallots,fresh,parsley'

    userinput = preprocess_userinput(userinput)

    userfeatures = vectorizer.transform([userinput])

    predictions = knn_prediction(userfeatures, predictions)
    predictions = linear_svm_prediction(userfeatures, predictions)
    predictions = crammer_svm_prediction(userfeatures, predictions)
    predictions = multinomialnb_prediction(userfeatures, predictions)
    predictions = logistic_regression_prediction(userfeatures, predictions)
    predictions = bernoulinb_prediction(userfeatures, predictions)
    predictions = sgd_prediction(userfeatures, predictions)

    predictions = ' '.join(predictions)
    predictions = re.sub(r'[^\w\s]', '', predictions)
    predictions = predictions.split(' ')
    frequency_distribution = nltk.FreqDist(predictions)
    global predicted_cuisine
    predicted_cuisine = frequency_distribution.max()

    print("Top prediction : ", predicted_cuisine)


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

    df.to_csv('zomato_model.csv')

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['ing_mod'])

    cuisine_map={'0':'brazilian', '1':'british', '2':'cajun_creole', '3':'chinese', '4':'filipino', '5':'french', '6':'greek', '7':'indian', '8':'irish', '9':'italian', '10':'jamaican', '11':'japanese', '12':'korean', '13':'mexican', '14':'moroccan', '15':'russian', '16':'southern_us', '17':'spanish', '18':'thai', '19':'vietnamese'}

    Y=[]
    Y = df['cuisine']

    userfeatures = vectorizer.transform(["plain flour ground pepper broiler-fryer chicken tomatoes ground black zesty italian dressing thyme gutter green tomatoes italian seasoning meal milk vegetable oil"])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100)

    predictions = []

    #KNN classifier with K-value = 11
    neigh = KNeighborsClassifier(n_neighbors = 11, weights='uniform', algorithm='auto')
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:11")
    prediction = neigh.predict(userfeatures)
    print("KNN Prediction: ", prediction)
    with open('Trained Models/KNN_Model.pkl', 'wb') as f:
        pickle.dump(neigh, f)
    predictions.append(str(prediction))

    # OVA SVM(Grid Search Results: Kernel - Linear, C - 1, Gamma - Auto)
    lin_clf = svm.LinearSVC(C=1)
    lin_clf.fit(X_train, y_train)
    y_pred = lin_clf.predict(X_test)
    print("Accuracy of OVA SVM is ", accuracy_score(y_test, y_pred) * 100)
    prediction = lin_clf.predict(userfeatures)
    print("OVA SVM Prediction: ", prediction)
    with open('Trained Models/Linear_SVM_Model.pkl', 'wb') as f:
        pickle.dump(lin_clf, f)
    predictions.append(str(prediction))

    # SVM by Crammer(Grid Search Results: Gamma - , C - )
    lin_clf = svm.LinearSVC(C=1.0, multi_class='crammer_singer')
    lin_clf.fit(X_train, y_train)
    y_pred = lin_clf.predict(X_test)
    print("Accuracy of Crammer SVM is ", accuracy_score(y_test, y_pred) * 100)
    prediction = lin_clf.predict(userfeatures)
    print("Crammer SVM Prediction: ", prediction)
    with open('Trained Models/Crammer_SVM_Model.pkl', 'wb') as f:
        pickle.dump(lin_clf, f)
    predictions.append(str(prediction))

    # Implementing OVA Naive Bayes
    clf = MultinomialNB().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy of OVA Naive Bayes is ", accuracy_score(y_test, y_pred) * 100)
    prediction = clf.predict(userfeatures)
    print("OVA Naive Bayes Prediction: ", prediction)
    with open('Trained Models/MultinomialNB_Model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    predictions.append(str(prediction))

    # Implementing OVA Logistic Regerssion
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, y_train)
    y_pred = logisticRegr.predict(X_test)
    print("Accuracy of OVA Logistic Regression is ", accuracy_score(y_test, y_pred) * 100)
    prediction = logisticRegr.predict(userfeatures)
    print("OVA Logistic Regerssion Prediction: ", prediction)
    with open('Trained Models/Logistic_Regression_Model.pkl', 'wb') as f:
        pickle.dump(logisticRegr, f)
    predictions.append(str(prediction))

    # Implementing OVA Bernouli Naive Bayes
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy of Bernoulli Naive Bayes is ", accuracy_score(y_test, y_pred) * 100)
    prediction = clf.predict(userfeatures)
    print("Bernouli Naive Bayes Prediction: ", prediction)
    with open('Trained Models/BernouliNB_Model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    predictions.append(str(prediction))

    # Implementing SGD classifier
    sgd = linear_model.SGDClassifier()
    sgd.fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    print("Accuracy of SGD is ", accuracy_score(y_test, y_pred) * 100)
    prediction = sgd.predict(userfeatures)
    print("SGD Prediction: ", prediction)
    with open('Trained Models/SGD_Model.pkl', 'wb') as f:
        pickle.dump(sgd, f)
    predictions.append(str(prediction))

    predictions = ' '.join(predictions)
    predictions = re.sub(r'[^\w\s]', '', predictions)
    predictions = predictions.split(' ')
    frequency_distribution = nltk.FreqDist(predictions)
    predicted_cuisine = frequency_distribution.max()
    print("Top prediction : ", predicted_cuisine)

else:
    print('.')


#toprestraunts = rating_sort(predicted_restraunts)
#toprestraunts = average_cost_sort(predicted_restraunts)
#toprestraunts = booking_sort(predicted_restraunts,predicted_cuisine)
#toprestraunts = delivery_sort(predicted_restraunts,predicted_cuisine)
# country = 'United States'
# toprestraunts = country_sort(predicted_restraunts,predicted_cuisine,country)

# for i in toprestraunts:
#         for key, value in i.items():
#             print(key, ' : ', value)
#         print('\n')

#Button to take input the value of alpha and the query from the user
search_button = tk.Button(master_window, command = store_query, text = "Search",height= 1,width = 10,fg = yellow, bg = bb, font = (font_name,10,"bold"))
search_button.place(x=315,y=230)

rating_button = tk.Button(master_window, command = rating_sort, text = "Rating",height= 1,width = 10,fg = yellow, bg = bb, font = (font_name,10,"bold"))
rating_button.place(x=315,y=260)

#labels for printing roll number and name
label5 = tk.Label(master_window, text = "K19-1292",fg = black, bg = lightblue, font = (font_name,10))
label5.place(x=635,y=6)
label6 = tk.Label(master_window, text = "HASSAN JAMIL",fg = black, bg = lightblue, font = (font_name,10))
label6.place(x=595,y=25)

#main window loop of the GUI
master_window.mainloop()