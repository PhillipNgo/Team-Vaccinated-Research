import pandas as pd
import numpy as np
import re
import LiwcTrie as lt

class LiwcAnalyzer():
    def __init__(self, dictionary_file='Data/LIWC2007dictionary poster.xls'):
        # Read cleaned dictionary file into DataFrame
        self.liwc_df = pd.read_excel(dictionary_file)

        # Create a Trie, liwc_dict, of all terms in the dictionary
        dict_words = [term for term in self.liwc_df.values.flatten() if not pd.isnull(term)]
        self.liwc_dict = lt.LiwcTrieNode('*')
        for word in dict_words:
            lt.add(self.liwc_dict, word)

        # Create a Trie for each category of terms in the dictionary, liwc_cat_map
        self.liwc_cat_map = {}
        for category in self.liwc_df:
            cat_words = self.liwc_df[category].dropna().values
            cat_dict = lt.LiwcTrieNode('*')
            for word in cat_words:
                lt.add(cat_dict, word)
            self.liwc_cat_map[category] = cat_dict


    # Find LIWC values for a pandas Series of text
    # Returns a DataFrame of the values
    def parse(self, data):
        # List of Words, no punctuation
        def words(text):
            return re.sub("[^a-zA-Z\\s]", ' ', re.sub("['.]", '', text)).split()

        # List of Sentences, punction included except end marks (!, ?, .)
        def sentences(text):
            sentences = re.compile("[!?.]+").split(text)
            return [sentence for sentence in sentences if sentence] # filter empty sentences

        w = data.str.lower().apply(words) # Split text into lower case words
        s = data.apply(sentences) # Split text into sentences

        wc = w.apply(len) # word count
        sc = s.apply(len) # sentence count
        wps = wc / sc # words per sentence
        dic = w.apply(lambda words: [lt.find(self.liwc_dict, word) for word in words]).apply(np.count_nonzero) / wc # % of words in dictionary
        sixltr = w.apply(lambda words: [w for w in words if len(w) > 6]).apply(len) # words >6 letters

        # Title and Data lists
        liwc_cols = ['wc', 'sc', 'wps', 'dic', 'sixltr']
        liwc_data = [wc, sc, wps, dic, sixltr]

        # Calculate percentage of words in each LIWC category
        for category in self.liwc_df:
            c = w.apply(lambda words: [lt.find(self.liwc_cat_map[category], word) for word in words]).apply(np.count_nonzero) / wc
            liwc_data.append(c)
            liwc_cols.append(category.lower())

        # Combine the data into a DataFrame
        return pd.concat(liwc_data, axis=1, keys=liwc_cols)
