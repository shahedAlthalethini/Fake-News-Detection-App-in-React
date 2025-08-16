import pandas as pd
import re
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from nltk.corpus import stopwords
from typing import List



def load_and_clean_data():
    processedDataLemma = pd.read_csv('processedDataLemma.csv')
    processedDataLemma = processedDataLemma.drop(
        columns=['full_news', 'full_news_no_stopwords', 'full_news_no_ner', 'full_news_lemma'],
        axis=1
    )
    X = processedDataLemma['full_news_lemma_clean']
    Y = processedDataLemma['Label']
    return X, Y


def vectorize_text(X):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
    vectorizer.fit(X)
    X_vectorized = vectorizer.transform(X)
    return X_vectorized, vectorizer


def split_data(X, Y):
    return train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)


def train_logistic_regression(X_train, Y_train, X_test, Y_test):
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    X_train_prediction = model.predict(X_train)
    # training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    # print('Accuracy score of the training data : ', training_data_accuracy)

    X_test_prediction = model.predict(X_test)
    # test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    # print('Accuracy score of the test data : ', test_data_accuracy)

    # cm = confusion_matrix(Y_test, X_test_prediction)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    #             xticklabels=['Fake', 'Real'],
    #             yticklabels=['Fake', 'Real'])
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title('LogisticRegression Confusion Matrix')
    # plt.show()

    #print(classification_report(Y_test, X_test_prediction, target_names=["Fake", "Real"]))

    pickle.dump(model, open("LogisticRegression_model.pkl", "wb"))


def train_svc(X_train, Y_train, X_test, Y_test, vectorizer):
    model = SVC(kernel='linear', probability=True, class_weight='balanced')
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    X_train_prediction = model.predict(X_train)
    # training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
    # print('Accuracy score of the training data : ', training_data_accuracy)

    X_test_prediction = model.predict(X_test)
    # test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
    # print('Accuracy score of the test data : ', test_data_accuracy)

    # cm = confusion_matrix(Y_test, Y_pred)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    #             xticklabels=['Fake', 'Real'],
    #             yticklabels=['Fake', 'Real'])
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title('SVC Confusion Matrix')
    # plt.show()

    # print(classification_report(Y_test, Y_pred, target_names=["Fake", "Real"]))

    pickle.dump(model, open("svm_model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))


def setup_stopwords():
    return set(stopwords.words('arabic'))


def download_arabic_stopwords():
    url = "https://raw.githubusercontent.com/mohataher/arabic-stop-words/master/list.txt"
    response = requests.get(url)
    return set(response.text.splitlines())


def clean_arabic_text(text):
    arabic_stopwords = download_arabic_stopwords()

    if not isinstance(text, str):
        return ""

    text = re.sub(r'http\S+|www.\S+|pic\.twitter\.com/\S+', '', text)
    text = re.sub(r'<.*?>', '', text)

    emoji_pattern = re.compile("["       
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    text = re.sub(r'[\u064B-\u0652]', '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'[ـ،؛؟!:\.\,\(\)\[\]\{\}"\'«»\-_~…]', '', text)

    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if re.fullmatch(r'[^\u0600-\u06FF]+', line.strip()):
            continue
        line = re.sub(r'[^\u0600-\u06FF\s]', '', line)
        if line.strip():
            cleaned_lines.append(line.strip())

    clean_text = '\n'.join(cleaned_lines)

    clean_words = [word for word in clean_text.split() if word not in arabic_stopwords]
    final_text = ' '.join(clean_words)
    return final_text


def load_model_and_vectorizer():
    model = pickle.load(open("svm_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer


def predict_news(text, model, vectorizer):
    cleaned_text = clean_arabic_text(text)
    vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    confidence = round(probabilities[prediction] * 100, 2)
    label =  generate_message(prediction)
   #label = "Real News" if prediction == 1 else "Fake News"
    return label, confidence

def generate_message(prediction: int) -> str:
    if prediction == 1:
        return (
            "🟢 هذا الخبر يبدو حقيقيًا بناءً على تحليلنا.\n"
            "لكن يُنصح دائمًا بالتحقق من المصدر الرسمي قبل مشاركة الخبر."
        )
    else:
        return (
            "🔴 هذا الخبر يُحتمل أن يكون زائفًا أو مضللًا.\n"
            "يُفضل تجنّب تداوله حتى يتم التأكد من صحته من مصادر موثوقة."
        )

def check_input_reality(input: str)->str:
    X, Y = load_and_clean_data()
    X_vectorized, vectorizer = vectorize_text(X)
    X_train, X_test, Y_train, Y_test = split_data(X_vectorized, Y)
    #train_logistic_regression(X_train, Y_train, X_test, Y_test)
    train_svc(X_train, Y_train, X_test, Y_test, vectorizer)
    model, vectorizer = load_model_and_vectorizer()
    print("النص:", input)
    label, confidence = predict_news(input, model, vectorizer)
    return label

# print(check_input_reality("الفيديو اشتباك مقاومة تداول حساب و صفحة موقع تواصل اجتماعي."))

# def main():
#     X, Y = load_and_clean_data()
#     X_vectorized, vectorizer = vectorize_text(X)
#     X_train, X_test, Y_train, Y_test = split_data(X_vectorized, Y)
#     #train_logistic_regression(X_train, Y_train, X_test, Y_test)
#     train_svc(X_train, Y_train, X_test, Y_test, vectorizer)
#     check_input_reality("الفيديو اشتباك مقاومة تداول حساب و صفحة موقع تواصل اجتماعي.")



# main()
