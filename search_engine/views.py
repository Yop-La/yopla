from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse
import os

import string
import re
import nltk
# librairie pour faire du steeming ( réduire les mots à leurs racines)
from nltk.stem.porter import PorterStemmer
import collections;
import time
from collections import defaultdict
from pickle import dump, load
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import coo_matrix
import numpy as np
import scipy.sparse


import itertools

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

root_folder = os.path.dirname(os.path.abspath(__file__))
# chargement de l'index
with open(root_folder + '/static/search_engine/index.pkl', 'rb') as f:
    index = load(f)

# chargement des textes
with open(root_folder + '/static/search_engine/list_text.pkl', 'rb') as f:
    list_text = load(f)

#chargement du vocs
with open(root_folder + '/static/search_engine/voc.pkl', 'rb') as f:
    vocs = list(load(f))
    vocs = sorted(vocs)

nb_words = len(index.keys())
nb_texts = len(list_text)

def build_mat_index(metric):
    col = []
    row = []
    data = []

    for word, res_index in index.items():

        indiceCol = vocs.index(word)
        for text_res in res_index:
            col.append(indiceCol)
            row.append(text_res['indice_text'])
            if metric == 'presence':
                data.append(1)
            elif metric == 'tf':
                data.append(text_res['tf'])
            elif metric == 'tf_idf':
                data.append(text_res['tf_idf'])

    mat_index = coo_matrix((data, (row, col)), shape=(nb_texts, nb_words))
    return(mat_index)

def toVec(req):
    vec = np.repeat(0, nb_words)
    req = tokenise(req)
    if(len(req) == 0):
        return([])
    list_vocs = list(vocs)

    for word in req:
        indiceCol = vocs.index(word)
        vec[indiceCol] = 1
    return(vec)
# fonction tokenise qui à partir d'un texte retourne une liste de token normalisé
def tokenise(text):
    # split into words
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # stemming of words
    stemmed = [ps.stem(word) for word in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    words = [w for w in words if not w in stop_words]
    return (words)

def search_mat(mat_index,query):
    query = toVec(query)
    if(len(query) == 0):
        return([])
    query = np.array(query)
    print(len(query))
    print(sum(query))
    score_res = list(mat_index.dot(query))
    print(len(score_res))
    print(sum(score_res))
    return(score_res)

# matrice avec comme métrique absence ou présence du mot
sparse_matrix_b = scipy.sparse.load_npz(root_folder + '/static/search_engine/sparse_matrix.npz')
index_mat_pre = scipy.sparse.load_npz(root_folder + '/static/search_engine/index_mat_pre.npz')
index_mat_tf = scipy.sparse.load_npz(root_folder + '/static/search_engine/index_mat_tf.npz')
index_mat_tf_idf = scipy.sparse.load_npz(root_folder + '/static/search_engine/index_mat_tf_idf.npz')


res = search_mat(index_mat_tf,'air france')

def home(request):
    return (render(request, 'search_engine/home.html', {'date': datetime.now()}));




def get_pertient_results(n,mat_index,query):
    res_txt = []
    score_res = search_mat(mat_index,query)
    if(len(score_res) == 0):
        return([])
    for i in range(n):
        pos_res = score_res.index(max(score_res))
        score_res[pos_res] = -1
        res_txt.append(pos_res)
    return(res_txt)





# pour retourner le nombre de match d'un mot
def search_word(word):
    res = []
    word = tokenise(word)
    if (len(word) == 1):
        word = word[0]
        res = index[word]
    return (res)


def search_text(req):
    words = tokenise(req)
    if (len(words) == 0):
        return ([])
    list_res = [search_word(word) for word in words]

    # on récupère les indices des textes correspondant à chaque mot de la requête
    text_positions = [set(map(lambda x: x['indice_text'], res)) for res in list_res]

    # pour récupérer les indices des textes contenant exactement tous les mots de la requête
    inter = text_positions[0].intersection(*text_positions)

    # pour récupérer les indices des textes qui contiennent une partie des mots de la requête

    partial = [text_position - inter for text_position in text_positions]
    partial = itertools.chain(*partial)
    partial = set(partial)
    return {'exact': inter, 'partials': partial, 'list_res': list_res}


def search_phrase(req):
    words = tokenise(req)
    if (len(words) == 0):
        return ([])

    res_search_test = search_text(req)
    exact = res_search_test['exact']

    list_res = res_search_test['list_res']

    # pour ne garder que les résultats avec un exact match
    list_exact_res = list(
        map(lambda res: [text_res for text_res in res if text_res['indice_text'] in exact], list_res))

    res = []
    for i_res in range(len(exact)):
        text_pos = list_exact_res[0][i_res]['indice_text']
        inter_sum = set()
        for j_list_res in range(len(list_exact_res)):
            inter = list_exact_res[j_list_res][i_res]['pos_in_text']
            inter = set(map(lambda text_pos: text_pos - j_list_res, inter))
            if (len(inter_sum) == 0):
                inter_sum = inter
            inter_sum = inter_sum.intersection(inter)

        if (len(inter_sum) != 0):
            res.append(text_pos)
    return (res)


def search(request):
    requete = request.GET.get('requete', None)

    search_option = request.GET.get('search-option', None)

    if(search_option == 'search_phrase'):
        res_index = search_phrase(requete)
        res_index = res_index[1:10]
    elif search_option == 'binaire':
        res_index = get_pertient_results(10,index_mat_pre,requete)
    elif search_option == 'tf':
        res_index = get_pertient_results(10,index_mat_tf,requete)
    else:
        res_index = get_pertient_results(10,index_mat_tf_idf,requete)

    res = []
    for index in res_index:
        res.append(list_text[index])

    data = {

        'search_option': search_option,
        'requete': requete,
        'res': res,
        'res_index': res_index
    }
    return (JsonResponse(data))
