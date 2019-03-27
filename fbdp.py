#All code by Team Vaccinated 3/6/2019.

import numpy as np
import pandas as pd

import re
import nltk

import wordcloud as wc
import matplotlib.pyplot as plt

from nltk.stem.wordnet import wordnet

from bs4 import BeautifulSoup, NavigableString

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import RegexpTokenizer

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class FBDesktopParser():
    def __init__(self, filename='Data/Stop Mandatory Vaccination - Posts.html',
                article_div='mbs',
                article_host_div='_6lz _6mb _1t62 ellipsis',
                article_subtitle_div='_6m7 _3bt9'):
        self.filename = filename
        self.load_soup(filename)
        self.article_div = article_div
        self.article_host_div = article_host_div
        self.article_subtitle_div = article_subtitle_div
    
    def __repr__(self):
        return "FBDesktopParser({})".format(self.filename)
    
    def __str__(self):
        return "FBDesktopParser({})".format(self.filename)

    #Load a beautifulsoup from the html file
    def load_soup(self, filename):
        try:
            self.soup = BeautifulSoup(open(filename, encoding='utf-8'), 'html.parser')
        except Exception as e:
            print(e)

    #Parse a given post's soup and return a dataframe
    def parse_post(self, post_soup):
        data = {}
        data['linked_profiles'] = []
        data['hashtags'] = []

        #Get date published
        data['timestamp'] = post_soup.find('abbr').attrs['title']

        #Get text and links on the post
        data['links'] = []
        data['text'] = ''
        for p in post_soup.find_all('p'):
            if p.a:
                a_tags = p.a.extract()
                for a_tag in a_tags:
                    if not isinstance(a_tag, NavigableString):
                        a_tag = a_tag.text.strip()
                    else:
                        a_tag = str(a_tag)
                    if a_tag.startswith('#'):
                        data['hashtags'].append(a_tag[1:])
                    elif a_tag.startswith('http'):
                        data['links'].append(a_tag)
                    else:
                        data['linked_profiles'].append(a_tag)
            data['text'] += p.text

        #Find if it has an image or not
        img = post_soup.find('img', {'class': 'scaledImageFitWidth img'})
        data['img_src'] = img.attrs['src'] if img and 'src' in img.attrs else None
        data['img-label'] = img.attrs['aria-label'] if img and 'aria-label' in img.attrs else None

        #Get the link info if it exists
        article = post_soup.find('div', class_=self.article_div)
        data['article_name'] = article.text if article else None
        host = post_soup.find('div', class_=self.article_host_div)
        data['article_host'] = host.text if host else None
        subtitle = post_soup.find('div', class_=self.article_subtitle_div)
        data['article_subtitle'] = subtitle.text if subtitle else None

        #Find other profiles if it has linked to them
        data['linked_profiles'].extend(
                [page.text for page in post_soup.find_all('span', class_='fwb')
                if page.text != 'Stop Mandatory Vaccination'])
        return data

    #Generator for a set number of posts
    def parse_posts_generator(self, limit=0):
        for i, post_child in enumerate(self.soup.find_all('div', class_='userContent')):
            if limit != 0 and i >= limit:
                break
            yield self.parse_post(post_child.parent)

    #Parse all posts given a soup
    def parse_posts(self, limit=0):
        self.posts = pd.DataFrame(list(self.parse_posts_generator(limit)))
        return self.posts

    #Extract features from a dataframe
    def extract_features(self, bag_of_words=False, lemmatize=True):
        self.posts['has_article'] = self.posts.article_name.apply(lambda x: x != None)
        self.posts['text_length'] = self.posts.text.apply(len)

        self.posts['num_hashtags'] = self.posts.hashtags.apply(len)
        self.posts['has_text'] = self.posts.text_length.apply(lambda x: x > 0)
        self.posts['num_linked_profiles'] = self.posts.linked_profiles.apply(len)
        self.posts['num_links'] = self.posts.links.apply(len)

        #Extract nltk features-------------------------------------------------------------
        stop_words = set(nltk.corpus.stopwords.words("english")) #Stop words to not consider

        #Tokenize
        self.posts['text_tokenized'] = self.posts.text.apply(nltk.tokenize.word_tokenize)
        self.posts['num_tokens'] = self.posts.text_tokenized.apply(len)

        #Tokenize - no punctuations
        no_punc_tokenizer = RegexpTokenizer(r'\w+')
        self.posts['text_tokenized_filtered'] = self.posts.text.apply(
            lambda words: [word.lower() for word in no_punc_tokenizer.tokenize(words) if word not in stop_words])

        #Tokenize - lemmatize
        if lemmatize:
            lem = nltk.stem.wordnet.WordNetLemmatizer() #Lemmatize words if possible
            def get_wordnet_pos(pos):
                pos = pos[0].upper()
                wordnet_tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
                return wordnet_tag_dict.get(pos, wordnet.NOUN)

            self.posts['text_tokenized_lemmatized'] = self.posts.text_tokenized_filtered.apply(
                lambda words: [lem.lemmatize(word, get_wordnet_pos(pos)) for word, pos in nltk.pos_tag(words)])

        #Collect punc info
        self.posts['num_words'] = self.posts.text_tokenized_filtered.apply(len)
        puncs = [('periods', '.'), ('exclamations', '!'), ('questionms', '?'), ('equals', '='), ('dollars', '$')]
        for name, punc in puncs:
            self.posts['num_' + name] = self.posts.text_tokenized.apply(lambda words: words.count(punc))
            self.posts['percent_' + name] = self.posts['num_' + name] / self.posts.num_tokens

        #Bag of words model if wanted ---------------------------------------------------
        if bag_of_words:
            count_vectorizer = CountVectorizer()
            tfidf_transformer = TfidfTransformer()
            bag_of_words_matrix = tfidf_transformer.fit_transform(count_vectorizer.fit_transform(self.posts.text))
            return self.posts.to_sparse().join(pd.SparseDataFrame(bag_of_words_matrix,
                                    columns=['word_' + x for x in count_vectorizer.get_feature_names()]))
        return self.posts
    
    #Join features to dataset
    def join_features(self, features, **kwargs):
        self.posts = self.posts.join(features, **kwargs)
        return self.posts

    #Build a word cloud
    def make_wordcloud(self, include_vaccine_words=False):
        text = ' '.join(self.posts.text_tokenized_lemmatized.apply(lambda x: ' '.join(x)))
        stop_words = wc.STOPWORDS
        if not include_vaccine_words:
            stop_words = stop_words.union({'vaccine', 'vaccinated',
                                            'vaccination', 'vaccinate'})
        wordcloud = wc.WordCloud(stopwords=stop_words, background_color='white',
                                width=2100, height=1000).generate(text)
        plt.figure(1, figsize=(21, 10))
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()
