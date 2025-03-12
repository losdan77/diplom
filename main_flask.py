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
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from flask_sqlalchemy import SQLAlchemy
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///anonim.db'
db = SQLAlchemy(app)
app.config['FLASK_ADMIN_SWATCH'] = 'cerulean'
app.secret_key = os.urandom(24)


class Authors(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(50), nullable = True)
    example_text = db.Column(db.String(500), nullable = True)

    def __str__(self):
        return f'{[self.name, self.example_text]}'


# with app.app_context():
#     db.create_all()
#     db.session.add(Authors(name="example", example_text='123'))
#     db.session.commit()


admin = Admin(app, name='Anonim', template_mode='bootstrap3')
admin.add_view(ModelView(Authors, db.session))


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
        print('avg_ntld', fvs_avg_MTLD)
        print('lexical',fvs_lexical)
        print('punct',fvs_punct)

        result = print_result(fvs_lexical,
                                fvs_punct,
                                fvs_avg_MTLD,
                                method_syntax(texts))

        result_dict = {}

        for text, res in zip(text_from_file, result[1:]):
            result_dict[f'{text.filename}'] = res

        return render_template("index.html",
                               result=result_dict)

    return render_template("index.html", result=result_dict)


@app.route('/help')
def help_page():
    return render_template('help.html')


@app.route('/find_author', methods=["POST", "GET"])
def find_author():
    
    texts = []
    sum_stat = 0

    if request.method == 'POST':

        example_text = request.form.get('exampleText')
        example_text = ' '.join(example_text.splitlines())
        texts.append(example_text)

        text_from_authors = Authors.query.all()
        
        for text in text_from_authors:
            print('---', text)
            clear_text = ' '.join(text.example_text.splitlines())
            texts.append(clear_text)

        print(str(texts))
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
        print('avg_ntld', fvs_avg_MTLD)
        print('lexical',fvs_lexical)
        print('punct',fvs_punct)

        result = print_result(fvs_lexical,
                                fvs_punct,
                                fvs_avg_MTLD,
                                method_syntax(texts))

        result_dict = {}

        for text, res in zip(text_from_authors, result[1:]):
            result_dict[f'{text.name}'] = res

        return render_template("find_author.html",
                               result=result_dict)
    return render_template('find_author.html')


def token_to_pos(ch):
    tokens = nltk.word_tokenize(ch)
    return [p[1] for p in nltk.pos_tag(tokens)]

def method_avg_MTLD(texts, fvs_avg_MTLD, sum_stat):
    for e, text in enumerate(texts): #2-й метод на основании среднего всех метрик лексического расстояния
        ds = DiversityStats(text)
        stats = ds.get_stats()
        print(f'stats=', stats)
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
    print('syntax', fvs_syntax)
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
    print('fit_lexical=', mass1)
    mass2 = reduce_mass(km.fit_predict(fvs_punct))
    print('fit_punct=', mass2)
    mass3 = reduce_mass(km.fit_predict(fvs_avg_MTLD))
    print('fit_avg_mtld=', mass3)
    mass4 = reduce_mass(km.fit_predict(fvs_syntax))
    print('fit_syntax=', mass4)

    zip_mass = list(zip(mass1, mass2, mass3, mass4))

    for sum_mass in zip_mass:
        if sum(sum_mass) > 2:
            result_word = 'да'
        elif sum(sum_mass) < 2:
            result_word = 'нет'
        elif sum(sum_mass) == 2:
            result_word = 'возможно'

        result_mass.append([sum(sum_mass), result_word])


    return result_mass


if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0')
