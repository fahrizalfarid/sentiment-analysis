from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import sys
import re


factory = StemmerFactory()
stemmer = factory.create_stemmer()

factory_stopword = StopWordRemoverFactory()
stopword = factory_stopword.create_stop_word_remover()


# 'Pak Jokowi dilarang hadir besok karena sejumlah pendemo sudah membludak.'
kalimat = ' '.join(sys.argv[1:])
labels = ['Negative','Neutral','Positive']


naive_bayes = pickle.load(open("models/naive_bayes.pkl","rb"))
mlp = pickle.load(open("models/mlp.pkl","rb"))
logistic_regression = pickle.load(open("models/logistic_regression.pkl","rb"))

deep_learning_model = load_model("models/model_3-content-06-0.35.h5")
print(deep_learning_model.summary())


def tokenize(text):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(text)

    text_sequences = tokenizer.texts_to_sequences(text)
    text_padded = pad_sequences(text_sequences,
                                 maxlen=128,
                                 truncating='post',
                                 padding='post')
    return text_padded
    


def getClean(text):
    text = str(text).lower().replace('\\', '').replace('_', ' ') # case folding
    text = re.sub(r'http\S+', " ", text)    # remove urls (punctuation)
    text = re.sub(r'@\w+',' ', text)         # remove mentions
    text = re.sub(r'#\w+', ' ', text)       # remove hastags
    text = re.sub('r<.*?>',' ', text)       # remove html tags
    text = re.sub("(.)\\1{2,}", "\\1", text)
    text = re.sub("\n", "", text)           # remove new line
    
    # Stemming
    text = stemmer.stem(text)
    
    # Stopword
    text = stopword.remove(text)
    return text



cleanSentence = getClean(kalimat)
tokenizeSentence = tokenize(cleanSentence)

naive_bayes_predict = labels[np.int(naive_bayes.predict([cleanSentence]))]
mlp_predict = labels[np.int(mlp.predict([cleanSentence]))]
logistic_regression_predict = labels[np.int(logistic_regression.predict([cleanSentence]))]

deep_learning_predict = labels[
    np.int(
        np.max(deep_learning_model.predict(tokenizeSentence))
    )
]

print('Kalimat : {}, Naive bayes : {}, Multi layer perceptron : {}, Logistic regression : {}, Deep learning : {}'.format(
    cleanSentence, naive_bayes_predict, mlp_predict, logistic_regression_predict, deep_learning_predict
))