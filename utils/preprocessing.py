from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
from nltk.corpus import stopwords as stop_words
import warnings
import numpy as np

# ----- Monolingual/Original -----
class WhiteSpacePreprocessing():
    """
    Provides a very simple preprocessing script that filters infrequent tokens from text
    """
    def __init__(self, documents, stopwords_language="english", vocabulary_size=2000, min_len=10, max_len=200):
        """

        :param documents: list of strings
        :param stopwords_language: string of the language of the stopwords (see nltk stopwords)
        :param vocabulary_size: the number of most frequent words to include in the documents. Infrequent words will be discarded from the list of preprocessed documents
        """
        self.documents = documents
        self.stopwords = set(stop_words.words(stopwords_language) + stop_words.words("english"))
        self.vocabulary_size = vocabulary_size
        self.max_len = max_len
        self.min_len = min_len

    def preprocess(self):
        """
        Note that if after filtering some documents do not contain words we remove them. That is why we return also the
        list of unpreprocessed documents.

        :return: preprocessed documents, unpreprocessed documents and the vocabulary list
        """
        print("Preprocessing", len(self.documents), "documents")
        print("Max seq length:", self.max_len)

        # --my changes-- truncate raw articles to the first 200 tokens
        truncated_docs = [' '.join(doc.split()[:self.max_len]) for doc in self.documents]
        self.documents = truncated_docs
        # --end my changes --
        preprocessed_docs_tmp = self.documents
        preprocessed_docs_tmp = [doc.lower() for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [doc.translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 2 and w not in self.stopwords])
                             for doc in preprocessed_docs_tmp]

        vectorizer = CountVectorizer(max_features=self.vocabulary_size, token_pattern=r'\b[a-zA-Z]{2,}\b')
        vectorizer.fit_transform(preprocessed_docs_tmp)
        vocabulary = set(vectorizer.get_feature_names())
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in vocabulary])
                                 for doc in preprocessed_docs_tmp]

        preprocessed_docs, unpreprocessed_docs = [], []
        for i, doc in enumerate(preprocessed_docs_tmp):
            if len(doc.split()) >= self.min_len:
                # if len(doc.split()) > self.max_len:
                #     doc = ' '.join(doc.split()[:self.max_len])
                # if len(self.documents[i].split()) > self.max_len:
                #     self.documents[i] = ' '.join(self.documents[i].split()[:self.max_len])
                preprocessed_docs.append(doc)
                unpreprocessed_docs.append(self.documents[i])

        return preprocessed_docs, unpreprocessed_docs, list(vocabulary)


# ----- Multimodal AND Multilingual Preprocessing -----


class WhiteSpacePreprocessingM3L():
    """
    Provides a very simple preprocessing script for aligned multimodal AND multilingual data
    """
    def __init__(self, documents, image_urls, stopwords_languages, vocabulary_size=2000, min_len=10, max_len=200):
        """

        :param documents: list of lists of strings, e.g. [['good morning', 'thank you'], ['guten morgen', 'danke sie']]
        :param stopwords_language: list  of strings of the languages (see nltk stopwords)
        :param vocabulary_size: the number of most frequent words to include in the documents. Infrequent words will be discarded from the list of preprocessed documents
        """
        self.documents = documents
        self.image_urls = image_urls
        self.num_lang = len(stopwords_languages)
        self.languages = stopwords_languages
        self.stopwords = []
        for lang in self.languages:
            self.stopwords.append(set(stop_words.words(lang)))
        # same vocab_size for all langs for now
        self.vocabulary_size = vocabulary_size
        self.min_len = min_len
        self.max_len = max_len
        # if user has custom stopwords list
        #self.custom_stops = custom_stops

    def preprocess(self):
        """
        Note that if after filtering some documents do not contain words we remove them. That is why we return also the
        list of unpreprocessed documents.

        :return: preprocessed documents, unpreprocessed documents and the vocabulary list
        """
        # --my changes-- truncate raw articles to the first 200 tokens
        for l in range(self.num_lang):
            truncated_docs = [' '.join(doc.split()[:self.max_len]) for doc in self.documents[l]]
            self.documents[l] = truncated_docs
        # --end my changes --

        preprocessed_docs_tmp = []
        vocabulary = []
        for l in range(self.num_lang):
            print("--- lang", l, ":", self.languages[l], "---")
            preprocessed_docs = [doc.lower() for doc in self.documents[l]]
            preprocessed_docs = [doc.translate(
                str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs]
            preprocessed_docs = [' '.join([w for w in doc.split() if len(w) > 2 and w not in self.stopwords[l]])
                                 for doc in preprocessed_docs]
            # if self.custom_stops is not None:
            #     preprocessed_docs = [' '.join([w for w in doc.split() if len(w) > 1 and w not in self.custom_stops[l]])
            #                          for doc in preprocessed_docs]
            vectorizer = CountVectorizer(max_features=self.vocabulary_size, token_pattern=r'\b[a-zA-Z]{2,}\b')
            vectorizer.fit_transform(preprocessed_docs)
            vocabulary_lang = set(vectorizer.get_feature_names())
            print('vocabulary_lang:', len(vocabulary_lang))
            preprocessed_docs = [' '.join([w for w in doc.split() if w in vocabulary_lang]) for doc in preprocessed_docs]
            preprocessed_docs_tmp.append(preprocessed_docs)
            vocabulary.append(list(vocabulary_lang))
            # print('vocab size:', len(vocabulary_lang))


        if self.num_lang == 1:
            preprocessed_data_final = [[]]
            unpreprocessed_data_final = [[]]
            image_urls_final = []

            for i in range(len(preprocessed_docs_tmp[0])):
                doc1 = preprocessed_docs_tmp[0][i]
                image_url = self.image_urls[i]
                if self.min_len <= len(doc1.split()):
                    preprocessed_data_final[0].append(doc1)

                    unpreprocessed_data_final[0].append(self.documents[0][i])

                    image_urls_final.append(image_url)


        elif self.num_lang == 2:

            preprocessed_data_final = [[], []]
            unpreprocessed_data_final = [[], []]
            image_urls_final = []
            # docs must be aligned across languages and modalities (text-image)
            for i in range(len(preprocessed_docs_tmp[0])):
                doc1 = preprocessed_docs_tmp[0][i]
                doc2 = preprocessed_docs_tmp[1][i]
                image_url = self.image_urls[i]
                if self.min_len <= len(doc1.split()) and self.min_len <= len(doc2.split()):
                    preprocessed_data_final[0].append(doc1)
                    preprocessed_data_final[1].append(doc2)

                    unpreprocessed_data_final[0].append(self.documents[0][i])
                    unpreprocessed_data_final[1].append(self.documents[1][i])

                    image_urls_final.append(image_url)

        else:
            # TODO: rewrite in generic form for any number of languages
            raise NonImplementedError("Cannot process number of languages: %s" %self.num_lang)


        # preprocessed_data_final is a list of list of strings (processed articles) and image urls
        # unpreprocessed_data_final is a list of list of strings (original articles) and image urls
        # vocabulary is a list of list of words (separate vocabularies for each language)
        return preprocessed_data_final, unpreprocessed_data_final, vocabulary, image_urls_final


# ----- Multilingual Preprocessing -----

# ----- Hardlink setup for Contextual Model -----

class WhiteSpacePreprocessingMultilingual():
    """
    Provides a very simple preprocessing script for aligned multilingual documents
    """
    def __init__(self, documents, stopwords_languages, vocabulary_size=2000, min_len=10, custom_stops=None, max_len=200):
        """

        :param documents: list of lists of strings, e.g. [['good morning', 'thank you'], ['guten morgen', 'danke sie']]
        :param stopwords_language: list  of strings of the languages (see nltk stopwords)
        :param vocabulary_size: the number of most frequent words to include in the documents. Infrequent words will be discarded from the list of preprocessed documents
        """
        self.documents = documents
        self.num_lang = len(stopwords_languages)
        self.languages = stopwords_languages
        self.stopwords = []
        for lang in self.languages:
            self.stopwords.append(set(stop_words.words(lang)))
        # same vocab_size for all langs for now
        self.vocabulary_size = vocabulary_size
        # min/max article length after preprocessing
        self.min_len = min_len
        self.max_len = max_len
        # if user has custom stopwords list
        self.custom_stops = custom_stops

    def preprocess(self):
        """
        Note that if after filtering some documents do not contain words we remove them. That is why we return also the
        list of unpreprocessed documents.

        :return: preprocessed documents, unpreprocessed documents and the vocabulary list
        """
        # --my changes-- truncate raw articles to the first 200 tokens
        for l in range(self.num_lang):
            truncated_docs = [' '.join(doc.split()[:self.max_len]) for doc in self.documents[l]]
            self.documents[l] = truncated_docs
        # --end my changes --

        preprocessed_docs_tmp = []
        vocabulary = []
        for l in range(self.num_lang):
            print("--- lang", l, ":", self.languages[l], "---")
            preprocessed_docs = [doc.lower() for doc in self.documents[l]]
            preprocessed_docs = [doc.translate(
                str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs]
            preprocessed_docs = [' '.join([w for w in doc.split() if len(w) > 2 and w not in self.stopwords[l]])
                                 for doc in preprocessed_docs]
            if self.custom_stops is not None:
                preprocessed_docs = [' '.join([w for w in doc.split() if len(w) > 2 and w not in self.custom_stops[l]])
                                     for doc in preprocessed_docs]
            vectorizer = CountVectorizer(max_features=self.vocabulary_size, token_pattern=r'\b[a-zA-Z]{2,}\b')
            vectorizer.fit_transform(preprocessed_docs)
            vocabulary_lang = set(vectorizer.get_feature_names())
            print('vocabulary_lang:', len(vocabulary_lang))
            preprocessed_docs = [' '.join([w for w in doc.split() if w in vocabulary_lang]) for doc in preprocessed_docs]
            preprocessed_docs_tmp.append(preprocessed_docs)
            vocabulary.append(list(vocabulary_lang))
            # print('vocab size:', len(vocabulary_lang))

        preprocessed_docs_final = [[], []]
        unpreprocessed_docs_final = [[], []]
        # docs must be aligned across languages
        for i in range(len(preprocessed_docs_tmp[0])):
            doc1 = preprocessed_docs_tmp[0][i]
            doc2 = preprocessed_docs_tmp[1][i]
            if self.min_len <= len(doc1.split()) and self.min_len <= len(doc2.split()):
                # truncate docs if they exceed max_len
                # if len(doc1.split()) > self.max_len:
                #     doc1 = " ".join(doc1.split()[:self.max_len])
                # if len(doc2.split()) > self.max_len:
                #     doc2 = " ".join(doc2.split()[:self.max_len])
                preprocessed_docs_final[0].append(doc1)
                preprocessed_docs_final[1].append(doc2)
                unpreprocessed_docs_final[0].append(self.documents[0][i])
                unpreprocessed_docs_final[1].append(self.documents[1][i])
        # preprocessed_docs_final is a list of list of strings (processed articles)
        # unpreprocessed_docs_final is a list of list of strings (original articles)
        # vocabulary is a list of list of words (separate vocabularies for each language)
        return preprocessed_docs_final, unpreprocessed_docs_final, vocabulary


# ----- Hardlink setup for Contextual Model (improved version that works for any number of languages) -----

class WhiteSpacePreprocessingMultilingualTrue():
    """
    Provides a very simple preprocessing script for aligned multilingual documents
    """
    def __init__(self, documents, stopwords_languages, vocabulary_size=2000, min_len=10, custom_stops=None, max_len=200):
        """

        :param documents: list of lists of strings, e.g. [['good morning', 'thank you'], ['guten morgen', 'danke sie']]
        :param stopwords_language: list  of strings of the languages (see nltk stopwords)
        :param vocabulary_size: the number of most frequent words to include in the documents. Infrequent words will be discarded from the list of preprocessed documents
        """
        self.documents = documents
        self.num_lang = len(stopwords_languages)
        self.languages = stopwords_languages
        self.stopwords = []
        for lang in self.languages:
            self.stopwords.append(set(stop_words.words(lang)))
        # same vocab_size for all langs for now
        self.vocabulary_size = vocabulary_size
        self.min_len = min_len
        self.max_len = max_len
        # if user has custom stopwords list
        self.custom_stops = custom_stops

    def preprocess(self):
        """
        Note that if after filtering some documents do not contain words we remove them. That is why we return also the
        list of unpreprocessed documents.

        :return: preprocessed documents, unpreprocessed documents and the vocabulary list
        """
        # --my changes-- truncate raw articles to the first 200 tokens
        for l in range(self.num_lang):
            truncated_docs = [' '.join(doc.split()[:self.max_len]) for doc in self.documents[l]]
            self.documents[l] = truncated_docs
        # --end my changes --

        preprocessed_docs_tmp = []
        vocabulary = []
        for l in range(self.num_lang):
            print("--- lang", l, ":", self.languages[l], "---")
            preprocessed_docs = [doc.lower() for doc in self.documents[l]]
            preprocessed_docs = [doc.translate(
                str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs]
            preprocessed_docs = [' '.join([w for w in doc.split() if len(w) > 2 and w not in self.stopwords[l]])
                                 for doc in preprocessed_docs]
            if self.custom_stops is not None:
                preprocessed_docs = [' '.join([w for w in doc.split() if len(w) > 2 and w not in self.custom_stops[l]])
                                     for doc in preprocessed_docs]
            vectorizer = CountVectorizer(max_features=self.vocabulary_size, token_pattern=r'\b[a-zA-Z]{2,}\b')
            vectorizer.fit_transform(preprocessed_docs)
            vocabulary_lang = set(vectorizer.get_feature_names())
            print('vocabulary_lang:', len(vocabulary_lang))
            preprocessed_docs = [' '.join([w for w in doc.split() if w in vocabulary_lang]) for doc in preprocessed_docs]
            preprocessed_docs_tmp.append(preprocessed_docs)
            vocabulary.append(list(vocabulary_lang))
            # print('vocab size:', len(vocabulary_lang))

        preprocessed_docs_final = [[] for _ in range(self.num_lang)]
        unpreprocessed_docs_final = [[] for _ in range(self.num_lang)]
        # docs must be aligned across languages
        for i in range(len(preprocessed_docs_tmp[0])):
            docs = [preprocessed_docs_tmp[l][i] for l in range(self.num_lang)]
            length_check = [self.min_len <= len(doc.split()) for doc in docs]
            if sum(length_check) == len(docs):
                for l in range(self.num_lang):
                    preprocessed_docs_final[l].append(docs[l])
                    unpreprocessed_docs_final[l].append(self.documents[l][i])
        # preprocessed_docs_final is a list of list of strings (processed articles)
        # unpreprocessed_docs_final is a list of list of strings (original articles)
        # vocabulary is a list of list of words (separate vocabularies for each language)
        return preprocessed_docs_final, unpreprocessed_docs_final, vocabulary

# ----- Hardlink setup for Non-contextual model (Gibbs-PLTM) -----

class WhiteSpacePreprocessingGibbsMultilingual():
    """
    Provides a very simple preprocessing script for aligned multilingual documents
    """
    def __init__(self, documents, stopwords_languages, vocabulary_size=2000, min_len=10, max_len=200):
        """

        :param documents: list of lists of strings, e.g. [['good morning', 'thank you'], ['guten morgen', 'danke sie']]
        :param stopwords_language: list  of strings of the languages (see nltk stopwords)
        :param vocabulary_size: the number of most frequent words to include in the documents. Infrequent words will be discarded from the list of preprocessed documents
        """
        self.documents = documents
        self.num_lang = len(stopwords_languages)
        self.languages = stopwords_languages
        self.stopwords = []
        for lang in self.languages:
            self.stopwords.append(set(stop_words.words(lang)))
        # same vocab_size for all langs for now
        self.vocabulary_size = vocabulary_size
        # min/max article length after preprocessing
        self.min_len = min_len
        self.max_len = max_len

    def preprocess(self):
        """
        Note that if after filtering some documents do not contain words we remove them. That is why we return also the
        list of unpreprocessed documents.

        :return: preprocessed documents, unpreprocessed documents and the vocabulary list
        """

        # --my changes-- truncate raw articles to the first 200 tokens
        for l in range(self.num_lang):
            truncated_docs = [' '.join(doc.split()[:self.max_len]) for doc in self.documents[l]]
            self.documents[l] = truncated_docs
        # --end my changes --

        preprocessed_docs_tmp = []
        vocabulary = []
        for l in range(self.num_lang):
            print("--- lang", l, ":", self.languages[l], "---")
            preprocessed_docs = [doc.lower() for doc in self.documents[l]]
            preprocessed_docs = [doc.translate(
                str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs]
            preprocessed_docs = [' '.join([w for w in doc.split() if len(w) > 2 and w not in self.stopwords[l]])
                                 for doc in preprocessed_docs]
            vectorizer = CountVectorizer(max_features=self.vocabulary_size, token_pattern=r'\b[a-zA-Z]{2,}\b')
            vectorizer.fit_transform(preprocessed_docs)
            vocabulary_lang = set(vectorizer.get_feature_names())
            print('vocabulary_lang:', len(vocabulary_lang))
            preprocessed_docs = [[w for w in doc.split() if w in vocabulary_lang] for doc in preprocessed_docs]
            preprocessed_docs_tmp.append(preprocessed_docs)
            vocabulary.append(list(vocabulary_lang))
            # print('vocab size:', len(vocabulary_lang))

        preprocessed_docs_final = [[], []]
        # docs must be aligned across languages
        for i in range(len(preprocessed_docs_tmp[0])):
            doc1 = preprocessed_docs_tmp[0][i]
            doc2 = preprocessed_docs_tmp[1][i]
            if self.min_len <= len(doc1) and self.min_len <= len(doc2):
                # truncate docs if they exceed max_len
                if len(doc1) > self.max_len:
                    doc1 = doc1[:self.max_len]
                if len(doc2) > self.max_len:
                    doc2 = doc2[:self.max_len]
                preprocessed_docs_final[0].append(doc1)
                preprocessed_docs_final[1].append(doc2)
        # preprocessed_docs_final is a list of list of strings (processed articles)
        # vocabulary is a list of list of words (separate vocabularies for each language)
        return preprocessed_docs_final, vocabulary

    def preprocess_for_inference(self, vocabulary):
        """
        Preprocess aligned multilingual documents for inference using the given vocabulary

        :return: preprocessed documents using the given vocabulary
        """
        preprocessed_docs_tmp = []
        for l in range(self.num_lang):
            print("--- lang", l, ":", self.languages[l], "---")
            preprocessed_docs = [doc.lower().split() for doc in self.documents[l]]
            preprocessed_docs = [[w for w in doc if w in vocabulary[l]] for doc in preprocessed_docs]
            preprocessed_docs_tmp.append(preprocessed_docs)

        preprocessed_docs_final = [[], []]
        # docs must be aligned across languages
        for i in range(len(preprocessed_docs_tmp[0])):
            doc1 = preprocessed_docs_tmp[0][i]
            doc2 = preprocessed_docs_tmp[1][i]
            if self.min_len <= len(doc1) and self.min_len <= len(doc2):
                # truncate docs if they exceed max_len
                # if len(doc1) > self.max_len:
                #     doc1 = " ".join(doc1[:self.max_len])
                # if len(doc2) > self.max_len:
                #     doc2 = " ".join(doc2[:self.max_len])
                preprocessed_docs_final[0].append(doc1)
                preprocessed_docs_final[1].append(doc2)
        # preprocessed_docs_final is an aligned list of articles (one list per language)
        return preprocessed_docs_final


# # -----  Softlink setup -----
# # Assumptions:
# # language directionality: lang1 -> lang2 (i.e. lang1 doc is associated multiple lang2 docs)
# # num_languages: 2
# class WhiteSpacePreprocessingMultilingualSoftlink():
#     """
#     Provides a very simple preprocessing script for partially-aligned incomparable corpus
#     """
#     def __init__(self, documents, stopwords_languages, vocabulary_size=2000, min_len=10, delimiter='||',
#                  weights=None):
#         """
#
#         :param documents: list of lists of strings, e.g. [['good morning', 'thank you'], ['guten morgen', 'guten tag'],
#         'danke sie', 'danke schön']]
#         :param stopwords_language: list  of strings of the languages (see nltk stopwords)
#         :param vocabulary_size: the number of most frequent words to include in the documents. Infrequent words will be discarded from the list of preprocessed documents
#         """
#         self.documents = documents
#         self.num_lang = len(stopwords_languages)
#         self.languages = stopwords_languages
#         self.stopwords = []
#         for lang in self.languages:
#             self.stopwords.append(set(stop_words.words(lang)))
#         # same vocab_size for all langs for now
#         self.vocabulary_size = vocabulary_size
#         # min/max article length after preprocessing
#         self.min_len = min_len
#         # delimiter between documents for lang2
#         self.delimiter = delimiter
#         # weights for docs in lang2 (matrix of size num_lang1_docs x max_lang2_docset)
#         self.weights = weights
#
#     def preprocess(self):
#         """
#         Note that if after filtering some documents do not contain words we remove them. That is why we return also the
#         list of unpreprocessed documents.
#
#         :return: preprocessed documents, unpreprocessed documents and the vocabulary list
#         """
#         preprocessed_docs_tmp = [[], []]
#         vocabulary = []
#         for l in range(2):
#             print("--- lang", l, ":", self.languages[l], "---")
#             # process documents as normal for lang1
#             # print("raw docs:", len(self.documents[l]))
#             if l == 0:
#                 preprocessed_docs = [doc.lower() for doc in self.documents[l]]
#                 preprocessed_docs = [doc.translate(
#                     str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs]
#                 preprocessed_docs = [' '.join([w for w in doc.split() if len(w) > 1 and w not in self.stopwords[l]])
#                                      for doc in preprocessed_docs]
#                 vectorizer = CountVectorizer(max_features=self.vocabulary_size, token_pattern=r'\b[a-zA-Z]{2,}\b')
#                 vectorizer.fit_transform(preprocessed_docs)
#                 vocabulary_lang = set(vectorizer.get_feature_names())
#                 print('vocabulary_lang:', len(vocabulary_lang))
#                 preprocessed_docs = [' '.join([w for w in doc.split() if w in vocabulary_lang])
#                                      for doc in preprocessed_docs]
#                 preprocessed_docs_tmp[0] = preprocessed_docs
#             # process documents in lang2  where each row has multiple documents
#             else:
#                 preprocessed_docs = [doc.lower().split(self.delimiter) for doc in self.documents[l]]
#                 preprocessed_docs = [[doc.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
#                                       for doc in docset]
#                                      for docset in preprocessed_docs]
#                 preprocessed_docs = [[' '.join([w for w in doc.split() if len(w) > 1 and w not in self.stopwords[l]])
#                                       for doc in doc_set]
#                                      for doc_set in preprocessed_docs]
#                 preprocessed_docs_flatten = [doc for doc_set in preprocessed_docs for doc in doc_set]
#                 vectorizer = CountVectorizer(max_features=self.vocabulary_size, token_pattern=r'\b[a-zA-Z]{2,}\b')
#                 vectorizer.fit_transform(preprocessed_docs_flatten)
#                 vocabulary_lang = set(vectorizer.get_feature_names())
#                 print('vocabulary_lang:', len(vocabulary_lang))
#                 preprocessed_docs = [[' '.join([w for w in doc.split() if w in vocabulary_lang]) for doc in doc_set]
#                                      for doc_set in preprocessed_docs]
#                 preprocessed_docs_tmp[1] = preprocessed_docs
#             vocabulary.append(list(vocabulary_lang))
#
#         preprocessed_docs_final = [[], []]
#         raw_docs_final = [[], []]
#         weights_final = []
#         # docs must be aligned across languages
#         for i in range(len(preprocessed_docs_tmp[0])):
#             # doc1 is a single doc
#             doc1 = preprocessed_docs_tmp[0][i]
#             # doc2_list is a list of docs
#             doc2_list = preprocessed_docs_tmp[1][i]
#             if self.min_len <= len(doc1.split()) :
#                 # # truncate doc1 if they exceed max_len
#                 # if len(doc1.split()) > self.max_len:
#                 #     doc1 = " ".join(doc1.split()[:self.max_len])
#                 # # truncate each doc in docs2 if they exceed max_len
#                 # for d2 in range(len(doc2_list)):
#                 #     if len(doc2_list[d2].split()) > self.max_len:
#                 #         doc2 = " ".join(doc2_list[d2].split()[:self.max_len])
#                 #         doc2_list[d2] = doc2
#                 preprocessed_docs_final[0].append(doc1)
#                 preprocessed_docs_final[1].append(doc2_list)
#                 raw_docs_final[0].append(self.documents[0][i])
#                 raw_docs_final[1].append(self.documents[1][i].lower().split(self.delimiter))
#                 weights_final.append(self.weights[i])
#         weights_final = np.array(weights_final)
#         return preprocessed_docs_final, raw_docs_final, vocabulary, weights_final
