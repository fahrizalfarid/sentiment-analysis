import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from tqdm import tqdm
import threading


i = 0

factory = StemmerFactory()
stemmer = factory.create_stemmer()

factory_stopword = StopWordRemoverFactory()
stopword = factory_stopword.create_stop_word_remover()



data_ = pd.read_csv("clean_.csv",delimiter=',')
print(data_.head())



def getClean(text):
    global i
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
    print(i)
    i+=1
    return text


def main():
    data_['title_clean'] = data_['title'].apply(lambda row: getClean(row))
    # data_['content_clean'] = data_['content'].apply(lambda row: getClean(row))
    data_.to_csv('cleaned_data.csv', encoding='utf-8', sep=',', index=False)



a = threading.Thread(target=main)
a.start()
a.join()







