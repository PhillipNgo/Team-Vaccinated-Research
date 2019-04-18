#Basic
import os
import math
import emoji
import numpy as np
import pandas as pd
from IPython.display import display, clear_output

#NLP
import nltk
import textstat
import wordcloud as wc
from Liwc import LiwcAnalyzer
from collections import Counter
from itertools import combinations, islice
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#Features
from sklearn.feature_selection import mutual_info_classif

#ML
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

BASE_OPTIONS = {
    'NUM_ARTICLE_HOSTS' : 30,
    'NUM_ARTICLE_DOMAINS' : 30,

    'BIGRAMS' : False,
    'BIGRAM_SAMPLE_SIZE' : 5_000,
    'NUM_BIGRAM_FEATURES_PER_GRP' : 15, #30 Total features

    'TRIGRAMS' : False,
    'TRIGRAM_SAMPLE_SIZE' : 5_000,
    'NUM_TRIGRAM_FEATURES_PER_GRP' : 5, #10 Total features

    'PAIRS' : False, 
    'PAIR_SAMPLE_SIZE' : 1_000,
    'NUM_PAIR_FEATURES_PER_GRP' : 5, #10 Total features

    'TERM_FREQUENCY': False,
    'NUM_WORD_FREQ_FEATURES' : 50,
    'MIN_WORD_DOC_FREQ' : .05, #Float : % of all documents

    'HASHTAGS': False,
    'NUM_HASHTAG_FEATURES' : 20,

    'FEATURE_SCALER' : StandardScaler(with_std=True)
}

def load_and_clean(filename, options={}):
    #Final result dict
    results = {}
    
    #Load data
    dtypes = {
        'article_host': 'str', 
        'article_name': 'str', 
        'article_subtitle': 'str', 
        'hashtags': 'object',
        'img-label': 'str', 
        'img_src': 'str', 
        'linked_profiles': 'object', 
        'links': 'object', 
        'text': 'str', 
        'text_tokenized': 'object', 
        'text_tokenized_filtered': 'object',
        'text_tokenized_lemmatized': 'object', 
        'page_name': 'category', 
        'page_name_adjusted': 'category',
        'text_length': 'int32', 
        'wc': 'int32', 
        'sc': 'int32',
        'sixltr': 'int32'
    }
    print('Loading Data')
    posts = pd.read_csv(filename, index_col=0, dtype=dtypes, 
                             parse_dates=['timestamp'])
    
    #Set options not in base options
    op = BASE_OPTIONS.copy()
    for o in options:
        op[options] = o
        
    #Correct has_text settings
    posts['text'] = posts.text.str.strip()
    posts['text_length'] = posts.text.apply(lambda x: len(x) if type(x) == str else 0)
    posts['has_text'] = posts.text_length.apply(lambda x: x > 0).astype('bool')
    assert posts.has_text.max() == True, "Posts does not have has_text"
    clear_output(wait=False)
        
    #Fix list columns written as string
    print('Fixing Lists') 
    posts = fix_list_columns(posts)
    clear_output(wait=False)
    
    #Number of emojis in text
    print('Adding emojis')
    posts['num_emojis'] = posts.text_tokenized.apply(
        lambda x: len([e for e in x if e in emoji.UNICODE_EMOJI]))
    clear_output(wait=False)

    #Whether post has an image
    print('Adding images')
    posts['has_img'] = ~posts.img_src.isnull()
    clear_output(wait=False)
    
    #Add article hosts and domains
    print('Adding articles')
    posts['article_domain'] = posts.article_host.apply(
        lambda h: get_host_and_domain(h)[1]).astype('category')
    posts['article_host'] = posts.article_host.apply(
        lambda h: get_host_and_domain(h)[0]).astype('category')
    clear_output(wait=False)

    #Fix punc info
    print('Adding punctuation')
    puncs = [('periods', '.'), ('exclamations', '!'), ('questionms', '?'), 
             ('equals', '='), ('dollars', '$')]
    for name, punc in puncs:
        posts['percent_' + name] = posts.text_tokenized.apply(
            lambda words: words.count(punc)) / posts.num_tokens
    clear_output(wait=False)

    #Percent All Caps
    print('Adding all-caps')
    posts['percent_all_caps'] = posts.text_tokenized.apply(
        lambda tokens: [token.isupper() for token in tokens].count(True) / 
                        len(tokens) if len(tokens) else 0)
    clear_output(wait=False)

    #Scale 'num_' features by number of words to reduce dependence on how long the text is
    print('Turning num_ features into percent_ features') 
    skip_percs = {'num_words', 'num_tokens'}
    for nc in [n for n in posts.columns if n.startswith('num_') and n not in skip_percs]:
        percent_column_name = 'percent_' + '_'.join(nc.split('_')[1:])
        if percent_column_name not in posts.columns:
            replacement = (posts[nc] / posts.num_words).apply(
                lambda x: x if not math.isinf(x) else 0)
            posts[percent_column_name] = replacement
        posts.drop(nc, axis=1, inplace=True)
    clear_output(wait=False)

    #Take log of positive, exponential columns [text_length, num_words, n]
    print('Taking log of exponential counts (and of zeros *dab on the hater below)')
    for c in ['text_length', 'num_words', 'num_tokens']:
        posts[c + '_log'] = pd.Series(np.log(posts[c])).replace([np.inf, -np.inf], 0)
        posts.drop(c, axis=1, inplace=True)
    clear_output(wait=False)

    #Clean article hosts based upon popularity
    print('Reducing number of article hosts')
    common_hosts = set(posts.article_host.value_counts().sort_values(
        ascending=False).head(op['NUM_ARTICLE_HOSTS']).values)
    posts['article_host'] = posts.article_host.apply(
        lambda x: x if x in common_hosts else 'other').astype('category')
    results['hosts'] = common_hosts
    clear_output(wait=False)

    #Clean domains based upon most common
    print('Reducing number of article domains')
    common_domains = set(posts.article_domain.value_counts().sort_values(
        ascending=False).head(op['NUM_ARTICLE_DOMAINS']).values)
    posts['article_domain'] = posts.article_domain.apply(
        lambda x: x if x in common_domains else 'other').astype('category')
    results['domains'] = common_domains
    clear_output(wait=False)

    #Filter text_filtered and text_tokenized_lemmatized further
    print('Filtering tokens further')
    stop_words = set(nltk.corpus.stopwords.words("english")).union(
        wc.STOPWORDS).union(common_domains).union({'http', 'https', 'bit', 'ly'})

    def filter_words(words):
        return [word for word in words if 
                (word not in stop_words and not word.isdigit())]
    posts['text_tokenized_filtered'] = \
        posts.text_tokenized_filtered.apply(filter_words)
    posts['text_tokenized_lemmatized'] = \
        posts.text_tokenized_lemmatized.apply(filter_words)
    clear_output(wait=False)

    #Rename liwc features to the same naming scheme
    print('Renaming liwc features to new scheme')
    liwc_counts = ['wc', 'sc', 'wps', 'sixltr']
    liwc_scores = ['dic', 'funct', 'pronoun', 'ppron',
           'i', 'we', 'you', 'shehe', 'they', 'ipron', 'article', 'verbs',
           'auxvb', 'past', 'present', 'future', 'adverbs', 'prep', 'conj',
           'negate', 'quant', 'numbers', 'swear', 'social', 'family',
           'friends', 'humans', 'affect', 'posemo', 'negemo', 'anx', 'anger',
           'sad', 'cogmech', 'insight', 'cause', 'discrep', 'tentat',
           'certain', 'inhib', 'incl', 'excl', 'percept', 'see', 'hear',
           'feel', 'bio', 'body', 'health', 'sexual', 'ingest', 'relativ',
           'motion', 'space', 'time', 'work', 'achiev', 'leisure', 'home',
           'money', 'relig', 'death', 'assent', 'nonflu', 'filler']
    posts.rename(columns={c: 'liwc_counts_' + c for c in liwc_counts}, inplace=True)
    posts.rename(columns={c: 'liwc_scores_' + c for c in liwc_scores}, inplace=True)
    clear_output(wait=False)

    #Bigrams
    if op['BIGRAMS']:
        print('Computing Bigrams')
        posts, bigrams = add_ngrams(posts, 2, 'bigram_', 
                                    sample_size=op['BIGRAM_SAMPLE_SIZE'],
                                    most_common=op['NUM_BIGRAM_FEATURES_PER_GRP'])
        results['bigrams'] = bigrams
        clear_output(wait=False)
        
    #Trigrams
    if op['TRIGRAMS']:
        print('Computing Trigrams')
        posts, trigrams = add_ngrams(posts, 3, 'trigrams_', 
                                     sample_size=op['TRIGRAM_SAMPLE_SIZE'],            
                                     most_common=op['NUM_TRIGRAM_FEATURES_PER_GRP'])
        results['trigrams'] = trigrams
        clear_output(wait=False)
    
    #Pairs
    if op['PAIRS']:
        print('Computing Pairs')
        posts, pairs = add_pairs(posts, sample_size=op['PAIR_SAMPLE_SIZE'], 
                             most_common_pairs=op['NUM_PAIR_FEATURES_PER_GRP'])
        results['pairs'] = pairs
        clear_output(wait=False)
        
    #Term Frequency
    if op['TERM_FREQUENCY']:
        print('Computing TFIDF')
        tfidf = TfidfVectorizer(stop_words=stop_words, 
                                dtype='int32', min_df=op['MIN_WORD_DOC_FREQ'], 
                                max_features=op['NUM_WORD_FREQ_FEATURES'])
        word_vecs = \
            tfidf.fit_transform(posts.text_tokenized_lemmatized.apply(
                lambda words: ' '.join(words)))
        posts = posts.join(pd.DataFrame(word_vecs.toarray(), 
                                        columns=['count_' + c for c in 
                                                 word_vecs.get_feature_names()]))
        assert posts.shape[0] == 89867
        results['terms'] = word_vecs.get_feature_names()
        clear_output(wait=False)

    #Hashtags
    if op['HASHTAGS']:
        print('Computing Hashtags')
        hashtags = set([h[0] for h in hashtags(posts.hashtags, 
                                               most_common=op['NUM_HASHTAG_FEATURES'])])
        for hashtag in hashtags:
            posts['hashtag_' + hashtag] = posts.hashtags.apply(
                lambda x: Counter(x)[hashtag]).astype('int32')
        results['hashtags'] = hashtags
        clear_output(wait=False)
            
    #Article Analysis
    print('Analyzing article name features')
    posts['article_title_length'] = posts.article_name.map(len, na_action='ignore')
    posts['article_readability_fkg'] = posts.article_name.map(textstat.flesch_kincaid_grade, na_action='ignore')
    posts['article_readability_smog'] = posts.article_name.map(textstat.smog_index, na_action='ignore')
    posts['article_readability_gunning_fog'] = posts.article_name.map(textstat.gunning_fog, na_action='ignore')
    liwc = LiwcAnalyzer()
    posts = posts.join(liwc.parse(posts.article_name.fillna('')).add_prefix('article_liwc_'))
    clear_output(wait=False)
    
    #Normalize numerical features
    print('Normalizing features')
    features = posts.select_dtypes(
        ['bool', 'float64', 'float32', 'category', 'int64', 'int32'])
    normalized = features.select_dtypes(['float64', 'float32', 'int64', 'int32'])
    
    #If filling all zero-row zero scores with mean, fill with NaN first
    zero_text_features = normalized[~features.has_text].std().apply(lambda x: np.isnan(x) or x <= 0.001)
    zero_text_features = zero_text_features[zero_text_features].index.values
    zero_article_features = normalized[~features.has_article].std().apply(lambda x: np.isnan(x) or x <= 0.001)
    zero_article_features = zero_article_features[zero_article_features].index.values
    
    normalized.loc[~features.has_text, zero_text_features] = np.nan
    normalized.loc[~features.has_article, zero_article_features] = np.nan
    assert (normalized.loc[~features.has_text, zero_text_features].isnull(
            ).all().all()), "Not all non-text rows are null"
    assert (normalized.loc[~features.has_article, zero_article_features].isnull(
            ).all().all()), "Not all non-article rows are null"
    
    #Scale integer and float columns
    normalized = pd.DataFrame(op['FEATURE_SCALER'].fit_transform(normalized),
                              columns=normalized.columns)
    normalized.fillna(normalized.mean(skipna=True), inplace=True)
    clear_output(wait=False)
    
    #One hot encoding categorical variables
    print('Encoding dummies')
    dummies = pd.get_dummies(features.select_dtypes(['category']), dummy_na=True)
    clear_output(wait=False)
    
    #Final variables left over
    print('Combining results')
    remaining = features.select_dtypes(['uint8', 'bool']).drop('page_name', axis=1)
    
    #Combine all into one
    features = remaining.join(normalized).join(dummies)
    results['X'], results['y'] = features.drop('anti_vax', axis=1), features.anti_vax
    results['features'] = features
    clear_output(wait=False)
    
    return results

def fix_list_columns(posts):
    #Linked_profiles were written as NaN if empty
    posts['linked_profiles'] = posts.linked_profiles.fillna('[]')
    #Change list columns to lists
    list_cols = ['hashtags', 'links', 'linked_profiles', 'text_tokenized', 
                 'text_tokenized_filtered', 'text_tokenized_lemmatized']
    for col in list_cols:
        posts[col] = posts[col].apply(eval)
    return posts

#Host and domain from raw article host
def get_host_and_domain(host):
    if type(host) != str: #Host is NaN
        return (host, host)
    host = host.replace('http://', '').replace('https://', '').split('/')[0]
    #Sometimes | titles can occur, need to strip
    if '|' in host:
        host = host.split('|')[0].lower()
    elif '.' in host:
        host = host.lower()
    else:
        return np.nan, np.nan
    domain = host.split('.')[-1]
    host = '.'.join(host.split('.')[:-1])
    if not domain.isalpha():
        return np.nan, np.nan
    return host, domain

#Get grams of a group posts
def add_ngrams(df, n, prefix, most_common=50, sample_size=1_000):
    def count_ngrams(words, n):
        iters = []
        for i in range(n):
            iters.append(islice(words, i, None))
        return Counter(zip(*iters))
    def ngrams(lemmatized_posts, n):
        return lemmatized_posts.apply(
            lambda words: count_ngrams(words, n)).sum().most_common(most_common)
    
    #Generate sorted set of ngram for each group
    anti_vax_ngrams = set([x[0] for x in ngrams(
        df[df.anti_vax].text_tokenized_lemmatized.sample(n=sample_size), n)])
    normal_ngrams = set([x[0] for x in ngrams(
        df[~df.anti_vax].text_tokenized_lemmatized.sample(n=sample_size), n)])
    #Compute presence of ngram in df 
    df_ngrams = df.text_tokenized_lemmatized.apply(lambda words: count_ngrams(words, n))
    total_ngrams = anti_vax_ngrams.union(normal_ngrams)
    #For each common ngram, count the number of occurrences per post
    for ngram in total_ngrams:
        new_name = prefix + '_'.join(ngram)
        df[new_name] = df_ngrams.apply(
            lambda counter: counter[ngram]).astype('int32')
    return df, total_ngrams

def add_pairs(df, sample_size=30_000, most_common_pairs=20, prefix='pair_'):
    #Generate pairs
    def generate_pairs(words):
        return [tuple(sorted(list(pair))) for pair in combinations(set(words), 2)]
    #Count pairs for group of posts
    def count_pairs(words):
        counter = Counter()
        counter.update(generate_pairs(words))
        return counter
    #Count most common pairs
    def pairs(lemmatized_posts):
        return lemmatized_posts.apply(count_pairs).sum().most_common(most_common_pairs)
    #Generate sorted set of pairs for each group
    anti_vax_pairs = set([pair for pair, count in pairs(
        df[df.anti_vax].text_tokenized_lemmatized.sample(n=sample_size))])
    print('Anti-Vax Pairs Generated')
    normal_pairs = set([pair for pair, count in pairs(
        df[~df.anti_vax].text_tokenized_lemmatized.sample(n=sample_size))])
    print('Normal Pairs Generated')
    #Compute presence of pairs in df
    df_pairs = df.text_tokenized_lemmatized.apply(count_pairs)
    print('Pairs Counted')
    #Add pairs to dataframe
    for pair in anti_vax_pairs.union(normal_pairs):
        df[prefix + pair[0] + '_' + pair[1]] = df_pairs.apply(
            lambda counter: counter[pair]).astype('int32')
    return df, anti_vax_pairs.union(normal_pairs)

#Add counts of most commonly occurring hashtags
def hashtags(hashtags, most_common=20):
    c = Counter()
    for h_list in hashtags:
        c.update(h_list)
    return c.most_common(most_common)
