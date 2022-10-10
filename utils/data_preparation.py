import numpy as np
from sentence_transformers import SentenceTransformer
import scipy.sparse
import warnings
from datasets.dataset import CTMDataset, M3LDataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def get_bag_of_words(data, min_length):
    """
    Creates the bag of words
    """
    vect = [np.bincount(x[x != np.array(None)].astype('int'), minlength=min_length)
            for x in data if np.sum(x[x != np.array(None)]) != 0]

    vect = scipy.sparse.csr_matrix(vect)
    return vect


def bert_embeddings_from_file(text_file, sbert_model_to_load, batch_size=200, max_seq_length=128):
    """
    Creates SBERT Embeddings from an input file
    """
    model = SentenceTransformer(sbert_model_to_load)

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    with open(text_file, encoding="utf-8") as filino:
        texts = list(map(lambda x: x, filino.readlines()))

    check_max_local_length(max_seq_length, texts)

    return np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))


def bert_embeddings_from_list(texts, sbert_model_to_load, batch_size=200, max_seq_length=128):
    """
    Creates SBERT Embeddings from a list
    """
    print('bert_embeddings_from_list - texts:', len(texts))
    model = SentenceTransformer(sbert_model_to_load)

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    check_max_local_length(max_seq_length, texts)

    return np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))


def check_max_local_length(max_seq_length, texts):
    max_local_length = np.max([len(t.split()) for t in texts])
    if max_local_length > max_seq_length:
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(f"the longest document in your collection has {max_local_length} words, the model instead "
                      f"truncates to {max_seq_length} tokens.")


def image_embeddings_from_file(image_urls, embedding_file):
    print('image_urls:', len(image_urls))
    if 'resnet' in embedding_file:
        embeddings = pd.read_csv(embedding_file, header=None)
        # embeddings = embeddings.rename({0: 'image_url'}, axis=1) # doesn't work for me
        desired_indices = [embeddings[embeddings[0] == im].index[0] for im in image_urls]
        embeddings = embeddings.iloc[desired_indices]
        embeddings = np.array(embeddings.drop(columns=[0], axis=1))
    else:
        embeddings = pd.read_csv(embedding_file)
        desired_indices = [embeddings[embeddings.image_url == im].index[0] for im in image_urls]
        embeddings = embeddings.iloc[desired_indices]
        embeddings = np.array(embeddings.drop(labels='image_url', axis=1))
    print("image embeddings:", embeddings.shape)
    return embeddings


# ----- Multimodal and Multilingual (M3L) -----

class M3LTopicModelDataPreparation:

    def __init__(self, contextualized_model=None, vocabularies=None, image_emb_file=None):
        self.contextualized_model = contextualized_model
        self.vocabularies = vocabularies
        self.id2token = []
        self.vectorizers = []
        self.label_encoder = None
        self.vocab_sizes = []
        self.image_emb_file = image_emb_file

    def load(self, contextualized_embeddings, bow_embeddings, image_embeddings, id2token):
        return M3LDataset(contextualized_embeddings, bow_embeddings, image_embeddings, id2token)

    # fit is for training data
    def fit(self, text_for_contextual, text_for_bow, image_urls):
        """
        This method fits the vectorizer and gets the embeddings from the contextual model

        :param text_for_contextual: list of list of unpreprocessed documents to generate the contextualized embeddings
        :param text_for_bow: list of list of preprocessed documents for creating the bag-of-words
        :param labels: list of labels associated with each document (optional).

        """
        if self.contextualized_model is None:
            raise Exception("You should define a contextualized model if you want to create the embeddings")

        # TODO: this count vectorizer removes tokens that have len = 1, might be unexpected for the users
        train_image_embeddings = image_embeddings_from_file(image_urls, self.image_emb_file)
        print("train_image_embeddings:", train_image_embeddings.shape)

        num_lang = len(text_for_bow)
        train_bow_embeddings, train_contextualized_embeddings = [], []
        for l in range(num_lang):
            print("----- lang:", l, "-----")
            vectorizer = CountVectorizer(vocabulary=self.vocabularies[l])
            train_bow_embeddings_lang = vectorizer.fit_transform(text_for_bow[l])
            # we use the same SBERT model for both languages (multilingual SBERT)
            if len(self.contextualized_model.split(",")) == 1:
                print('context_model:', self.contextualized_model)
                train_contextualized_embeddings_lang = bert_embeddings_from_list(text_for_contextual[l], self.contextualized_model)
            # or we can use different SBERT models per language (monolingual SBERTs)
            else:
                context_model = self.contextualized_model.split(",")[l].strip()
                print('context_model:', context_model)
                train_contextualized_embeddings_lang = bert_embeddings_from_list(text_for_contextual[l], context_model)
            print('train_bow_embeddings_lang:', train_bow_embeddings_lang.shape)
            train_bow_embeddings.append(train_bow_embeddings_lang)
            train_contextualized_embeddings.append(train_contextualized_embeddings_lang)
            print('train_contextualized_embeddings_lang:', train_contextualized_embeddings_lang.shape)
            vocab_lang = vectorizer.get_feature_names()
            self.vocab_sizes.append(len(self.vocabularies[l]))
            id2token_lang = {k: v for k, v in zip(range(0, len(self.vocabularies[l])), self.vocabularies[l])}
            self.id2token.append(id2token_lang)

        return M3LDataset(train_contextualized_embeddings, train_bow_embeddings, train_image_embeddings, self.id2token, num_lang)

    # transform is for data during inference--dataset is monolingual during inference
    def transform(self, text_for_contextual=None, image_urls=None, lang_index=0):
        """
        This methods create the input for the prediction. Essentially, it creates the embeddings with the contextualized
        model of choice and with trained vectorizer.

        If text_for_bow is missing, it should be because we are using ZeroShotTM
        """
        if self.contextualized_model is None:
            raise Exception("You should define a contextualized model if you want to create the embeddings")

        # get SBERT embeddings for contextualized
        if text_for_contextual is not None:
            if len(self.contextualized_model.split(",")) == 1:
                print('context_model:', self.contextualized_model)
                test_contextualized_embeddings = bert_embeddings_from_list(text_for_contextual, self.contextualized_model)
            else:
                context_model = self.contextualized_model.split(",")[lang_index].strip()
                print('context_model:', context_model)
                test_contextualized_embeddings = bert_embeddings_from_list(text_for_contextual, context_model)
            # create dummy matrix for Bow
            test_bow_embeddings = scipy.sparse.csr_matrix(np.zeros((len(text_for_contextual), 1)))
        else:
            test_contextualized_embeddings = None
        if image_urls is not None:
            test_image_embeddings = image_embeddings_from_file(image_urls, self.image_emb_file)
            # create dummy matrix for Bow
            test_bow_embeddings = scipy.sparse.csr_matrix(np.zeros((len(image_urls), 1)))
        else:
            test_image_embeddings = None

        return M3LDataset(test_contextualized_embeddings, test_bow_embeddings, test_image_embeddings, self.id2token, is_inference=True)


# ---- Original -----

class TopicModelDataPreparation:

    def __init__(self, contextualized_model=None):
        self.contextualized_model = contextualized_model
        self.vocab = []
        self.id2token = {}
        self.vectorizer = None
        self.label_encoder = None

    def load(self, contextualized_embeddings, bow_embeddings, id2token, labels=None):
        return CTMDataset(contextualized_embeddings, bow_embeddings, id2token, labels)

    def fit(self, text_for_contextual, text_for_bow, labels=None):
        """
        This method fits the vectorizer and gets the embeddings from the contextual model

        :param text_for_contextual: list of unpreprocessed documents to generate the contextualized embeddings
        :param text_for_bow: list of preprocessed documents for creating the bag-of-words
        :param labels: list of labels associated with each document (optional).

        """

        if self.contextualized_model is None:
            raise Exception("You should define a contextualized model if you want to create the embeddings")

        # TODO: this count vectorizer removes tokens that have len = 1, might be unexpected for the users
        self.vectorizer = CountVectorizer()

        train_bow_embeddings = self.vectorizer.fit_transform(text_for_bow)
        train_contextualized_embeddings = bert_embeddings_from_list(text_for_contextual, self.contextualized_model)
        self.vocab = self.vectorizer.get_feature_names()
        self.id2token = {k: v for k, v in zip(range(0, len(self.vocab)), self.vocab)}

        if labels:
            self.label_encoder = OneHotEncoder()
            encoded_labels = self.label_encoder.fit_transform(np.array([labels]).reshape(-1, 1))
        else:
            encoded_labels = None

        return CTMDataset(train_contextualized_embeddings, train_bow_embeddings, self.id2token, encoded_labels)

    def transform(self, text_for_contextual, text_for_bow=None, labels=None):
        """
        This methods create the input for the prediction. Essentially, it creates the embeddings with the contextualized
        model of choice and with trained vectorizer.

        If text_for_bow is missing, it should be because we are using ZeroShotTM
        """

        if self.contextualized_model is None:
            raise Exception("You should define a contextualized model if you want to create the embeddings")

        if text_for_bow is not None:
            test_bow_embeddings = self.vectorizer.transform(text_for_bow)
        else:
            # dummy matrix
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn("The method did not have in input the text_for_bow parameter. This IS EXPECTED if you "
                          "are using ZeroShotTM in a cross-lingual setting")

            test_bow_embeddings = scipy.sparse.csr_matrix(np.zeros((len(text_for_contextual), 1)))
        test_contextualized_embeddings = bert_embeddings_from_list(text_for_contextual, self.contextualized_model)

        if labels:
            encoded_labels = self.label_encoder.transform(np.array([labels]).reshape(-1, 1))
        else:
            encoded_labels = None

        return CTMDataset(test_contextualized_embeddings, test_bow_embeddings, self.id2token, encoded_labels)



