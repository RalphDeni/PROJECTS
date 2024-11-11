from flask import *
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
import string
import requests 
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english')) 

app = Flask(__name__)
bad_words = pd.read_csv('bad-words.csv')
cv = joblib.load('vector.joblib')
clf = joblib.load('model.joblib')  
 
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

def predict_with_asterisks(user_input):
    t = clean(user_input)
    t=t.lower()
    print(t)
    t = cv.transform([t])
    output = clf.predict(t)
    for word in bad_words['jigaboo']:
        user_input = user_input.replace(word, '*' * len(word))
    print(user_input)
    return user_input

def translate(source_text,source):  
    source_lang = source
    target_lang = 'en'
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q={source_text}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        result_text = data[0][0][0]
        print(result_text)
        return result_text
    else:
        return 'could not translate'
    
def ttranslate(source_text,source):
    source_lang = 'en'
    target_lang = source
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q={source_text}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        result_text = data[0][0][0]
        print(result_text)
        return result_text
    else:
        return 'could not translate'
    
@app.route('/') 
def hello():
    return render_template('index.html',pred='')
@app.route('/', methods=['POST'])
def predict():
    fea = [str(x) for x in request.form.values()]
    s=translate(fea[1],fea[0])
    s=predict_with_asterisks(s)
    s=ttranslate(s,fea[0])
    return render_template('index.html',pred=s)
    
if __name__ == '__main__':
    app.run(debug=True) 
