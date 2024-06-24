import streamlit as st
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Load your data
data_true = pd.read_csv('True.csv')
data_fake = pd.read_csv('Fake.csv')

data_fake["class"] = 0
data_true["class"] = 1

data_true_manual_testing = data_true.tail(10)
for i in range(21416, 21316, -1):
    data_true.drop([i], axis=0, inplace=True)

data_fake_manual_testing = data_fake.tail(10)
for i in range(23480, 23470, -1):
    data_fake.drop([i], axis=0, inplace=True)

data_true_manual_testing['class'] = 1
data_fake_manual_testing['class'] = 0

data_merge = pd.concat([data_fake, data_true], axis=0)
data = data_merge.drop(['title', 'subject', 'date'], axis=1)

def preprocess_text(text_data):
    preprocessed_text = []
    for sentence in text_data:
        sentence = re.sub(r'[^\w\s]', '', sentence)
        preprocessed_text.append(' '.join(token.lower() for token in str(sentence).split() if token not in stopwords.words('english')))
    return preprocessed_text

preprocessed_review = preprocess_text(data['text'].values)
data['text'] = preprocessed_review

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

data['text'] = data['text'].apply(wordopt)

x = data['text']
y = data['class']

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Logistic Regression Model
LR = LogisticRegression()
LR.fit(xv_train, y_train)

# Decision Tree Model
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

# Gradient Boosting Classifier Model
GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)

# Random Forest Classifier Model
RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)

def output_label(n):
    return "Fake News" if n == 0 else "Not A Fake News"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return {
        "LR Prediction": output_label(pred_LR[0]),
        "DT Prediction": output_label(pred_DT[0]),
        "GBC Prediction": output_label(pred_GBC[0]),
        "RFC Prediction": output_label(pred_RFC[0])
    }

st.title("Fake News Detection")
st.write("Enter the news text below to check if it's Fake News or Not A Fake News:")

user_input = st.text_area("Enter News Here")

if st.button("Check"):
    if user_input:
        results = manual_testing(user_input)
        st.write("Logistic Regression Prediction: ", results["LR Prediction"])
        st.write("Decision Tree Prediction: ", results["DT Prediction"])
        st.write("Gradient Boosting Classifier Prediction: ", results["GBC Prediction"])
        st.write("Random Forest Classifier Prediction: ", results["RFC Prediction"])
    else:
        st.write("Please enter some news text to check.")
