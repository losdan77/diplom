import glob
import numpy as np
import os
import fnmatch
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.vq import whiten
from ruts import DiversityStats
from flask import Flask, render_template, url_for, request, redirect
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'


@app.route("/", methods=["POST", "GET"])
def index():

    texts = []
    sum_stat = 0

    if request.method == 'POST':

        example_text = request.form.get('exampleText')
        example_text = ' '.join(example_text.splitlines())
        texts.append(example_text)

        text_from_file = request.files.getlist('files')

        for text in text_from_file:
            clear_text = ' '.join(text.read().decode().splitlines())
            texts.append(clear_text)


        sentence_tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')
        word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        file_counts = len(fnmatch.filter(os.listdir('./texts'), '*.txt'))

        fvs_lexical = np.zeros((len(texts), 3), np.float64)
        fvs_punct = np.zeros((len(texts), 3), np.float64)
        fvs_avg_MTLD = np.zeros((len(texts), 1), np.float64)

        method_avg_MTLD(texts, fvs_avg_MTLD, sum_stat)
        methods_punctuation_and_lexical(texts,
                                        sentence_tokenizer,
                                        word_tokenizer,
                                        fvs_lexical,
                                        fvs_punct)

        result = print_result(fvs_lexical,
                                fvs_punct,
                                fvs_avg_MTLD,
                                method_syntax(texts))

        result_dict = {}

        for text, res in zip(text_from_file, result[1:]):
            result_dict[f'{text.filename}'] = res

        return render_template("index.html",
                               result=result_dict)

    return render_template("index.html")

def token_to_pos(ch):
    tokens = nltk.word_tokenize(ch)
    return [p[1] for p in nltk.pos_tag(tokens)]

def method_avg_MTLD(texts, fvs_avg_MTLD, sum_stat):
    for e, text in enumerate(texts): #2-й метод на основании среднего всех метрик лексического расстояния
        ds = DiversityStats(text)
        stats = ds.get_stats()

        for key, stat in stats.items():
            sum_stat = sum_stat + stat

        avg_stat = sum_stat/len(stats)
        #print(e,'   ',avg_stat)
        fvs_avg_MTLD[e, 0] = avg_stat
        sum_stat = 0

def methods_punctuation_and_lexical(texts,
                                    sentence_tokenizer,
                                    word_tokenizer,
                                    fvs_lexical,
                                    fvs_punct):
    for e, ch_text in enumerate(texts):#1 метод
        # Лексические и пунктуационные особенности:
        tokens = nltk.word_tokenize(ch_text.lower())
        words = word_tokenizer.tokenize(ch_text.lower())
        sentences = sentence_tokenizer.tokenize(ch_text)
        vocab = set(words)
        words_per_sentence = np.array([len(word_tokenizer.tokenize(s))for s in sentences])

        # Среднее количество слов в предложении
        fvs_lexical[e, 0] = words_per_sentence.mean()
        # Изменение длины предложения
        fvs_lexical[e, 1] = words_per_sentence.std()
        # Лексическое разнообразие
        fvs_lexical[e, 2] = len(vocab) / float(len(words))

        #  Среднее количество запятых
        fvs_punct[e, 0] = tokens.count(',') / float(len(sentences))
        #  Среднее количество точек с запятой
        fvs_punct[e, 1] = tokens.count(';') / float(len(sentences))
        #  Среднее количество двоеточий
        fvs_punct[e, 2] = tokens.count(':') / float(len(sentences))

def method_syntax(texts):
    chapters_pos = [token_to_pos(ch) for ch in texts]
    # count frequencies for common POS types
    pos_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS']
    fvs_syntax = np.array([[ch.count(pos) for pos in pos_list]
                           for ch in chapters_pos]).astype(np.float64)
    # normalise by dividing each row by number of tokens in the chapter
    fvs_syntax /= np.c_[np.array([len(ch) for ch in chapters_pos])]
    return fvs_syntax

def reduce_mass(mass):
    if mass[0] == 1:
        return mass
    else:
        new_mass = []
        new_mass = map(lambda x: (x + 1) % 2, mass)
        return list(new_mass)

def print_result(fvs_lexical,
                 fvs_punct,
                 fvs_avg_MTLD,
                 fvs_syntax):

    result_mass = []

    #Функция выводит результаты кластеризируя их по к-среднему
    km = KMeans(n_clusters=2, init='k-means++', n_init=10, verbose=0)

    mass1 = reduce_mass(km.fit_predict(fvs_lexical))
    mass2 = reduce_mass(km.fit_predict(fvs_punct))
    mass3 = reduce_mass(km.fit_predict(fvs_avg_MTLD))
    mass4 = reduce_mass(km.fit_predict(fvs_syntax))

    zip_mass = list(zip(mass1, mass2, mass3, mass4))

    for sum_mass in zip_mass:
        result_mass.append(sum(sum_mass))

    return result_mass


if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0')
