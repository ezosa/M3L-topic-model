import pandas as pd
from utils.preprocess_wiki import parse_wikipedia_xml_dump, combine_multilingual_wikipedia_articles

# Preprocessing steps
# 1. Parse and clean Wikipedia XML dump for each language of interest (e.g. EN and DE)
# Wikipedia XML dumps are taken from https://mirror.accum.se/mirror/wikimedia.org/dumps/ (or other mirror sites)
wiki_l1 = "enwiki-20230201-pages-articles.xml"
wiki_l2 = "dewiki-20230201-pages-articles.xml"
parsed_articles_l1 = parse_wikipedia_xml_dump(xml_path=wiki_l1, lang='en')
parsed_articles_l2 = parse_wikipedia_xml_dump(xml_path=wiki_l2, lang='de')
lang1_articles = pd.read_csv(parsed_articles_l1)
lang2_articles = pd.read_csv(parsed_articles_l2)

# 2. Merge cleaned articles with the aligned titles and image_urls
aligned_titles_file = "https://github.com/ezosa/M3L-topic-model/blob/master/data/test-titles.csv"
aligned_titles = pd.read_csv(aligned_titles_file)
merged_wiki = combine_multilingual_wikipedia_articles(aligned_titles=aligned_titles,
                                                      lang1_articles=lang1_articles,
                                                      lang2_articles=lang2_articles,
                                                      lang1='en',
                                                      lang2='de')
# marged_wiki should have the columns en_title, en_text, de_title, de_text, image_url (see example https://github.com/ezosa/M3L-topic-model/blob/master/data/train-example.csv)
