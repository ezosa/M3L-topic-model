import pandas as pd
from utils.preprocess_wiki import parse_wikipedia_xml_dump, combine_multilingual_wikipedia_articles

# Preprocessing steps
# Step 1: Parse and clean Wikipedia articles from xml dumps for each language (e.g. EN and DE)
# Wikipedia XML dumps are taken from https://mirror.accum.se/mirror/wikimedia.org/dumps/ (or other mirror sites)
wiki_l1 = "enwiki-20230201-pages-articles.xml"
wiki_l2 = "dewiki-20230201-pages-articles.xml"
parsed_articles_l1 = parse_wikipedia_xml_dump(xml_path=wiki_l1, lang='en')
parsed_articles_l2 = parse_wikipedia_xml_dump(xml_path=wiki_l2, lang='de')
lang1_articles = pd.read_csv(parsed_articles_l1)
lang2_articles = pd.read_csv(parsed_articles_l2)


# Step 2: Here we combine the cleaned articles from step 1 into a dataset of aligned Wikipedia titles, image urls and full articles
# 2.1 (optional)
# aligned_titles used in our experiments is provided in the Git repo (see step 2.3 below)
# If you want to use other language pairs, download an xml from https://linguatools.org/tools/corpora/wikipedia-comparable-corpora/
xml_path = "wikicomp-2014_deen.xml"
# Parse the xml file for the aligned titles and create a csv
aligned_titles_file = align_wikipedia_titles(xml_path, lang_pair='de-en')

# 2.2 (optional)
# image_url is taken from WIT (https://www.kaggle.com/competitions/wikipedia-image-caption/data)
# we match the image_url to the article using the article title (e.g. en_title)
# the result here should be a csv with the columns: en_title, de_title, image_url


# 2.3 Merge cleaned articles with the aligned titles file
aligned_titles_file = "https://github.com/ezosa/M3L-topic-model/blob/master/data/train-titles.csv"
aligned_titles = pd.read_csv(aligned_titles_file)
merged_wiki = combine_multilingual_wikipedia_articles(aligned_titles=aligned_titles,
                                                      lang1_articles=lang1_articles,
                                                      lang2_articles=lang2_articles,
                                                      lang1='en',
                                                      lang2='de')
# the result should be a csv with columns: en_title, en_text, de_title, de_text, image_url (see example https://github.com/ezosa/M3L-topic-model/blob/master/data/train-example.csv)
