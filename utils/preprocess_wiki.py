import string
import re
import os
import pandas as pd
import xml.etree.ElementTree as ET
import mwparserfromhell


def parse_and_clean_wikicode(raw_content):
    """Strips formatting and unwanted sections from raw page content."""
    wikicode = mwparserfromhell.parse(raw_content)
    # Filters for references, tables, and file/image links.
    re_rm_wikilink = re.compile("^(?:File|Image|Media):", flags=re.IGNORECASE | re.UNICODE)
    def rm_wikilink(obj):
        return bool(re_rm_wikilink.match(str(obj.title)))
    def rm_tag(obj):
        return str(obj.tag) in {"ref", "table"}
    def rm_template(obj):
        return obj.name.lower() in {"reflist", "notelist", "notelist-ua", "notelist-lr", "notelist-ur", "notelist-lg"}
    def try_remove_obj(obj, section):
        try:
            section.remove(obj)
        except ValueError:
            # For unknown reasons, objects are sometimes not found.
            pass
    section_text = []
    # Filter individual sections to clean.
    for section in wikicode.get_sections(flat=True, include_lead=True, include_headings=True):
        for obj in section.ifilter_wikilinks(matches=rm_wikilink, recursive=True):
            try_remove_obj(obj, section)
        for obj in section.ifilter_templates(matches=rm_template, recursive=True):
            try_remove_obj(obj, section)
        for obj in section.ifilter_tags(matches=rm_tag, recursive=True):
            try_remove_obj(obj, section)
        section_text.append(section.strip_code().strip())
    return "\n\n".join(section_text)


def parse_wikipedia_xml_dump(xml_path, lang='de'):
    """ Extracts title and article content from Wikipedia XML dumps (https://dumps.wikimedia.org/) and saves result to CSV"""
    filepath = xml_path
    f = open(filepath, 'r')
    tree = ET.parse(f)
    root = tree.getroot()
    articles = {lang + '_title': [],
                lang + '_text': []}
    for child in root:
        if 'page' in child.tag:
            page = child
            article_title = ""
            for child2 in page:
                if 'title' in child2.tag:
                    title = child2
                    if title.text is not None:
                        article_title = title.text.lower()
                if 'revision' in child2.tag:
                    revision = child2
                    for child3 in revision:
                        if 'text' in child3.tag:
                            txt = child3
                            if txt.text is not None:
                                text = txt.text.lower().strip()
                                # check if article is long enough (by num of chars)
                                if len(text) > 200 and len(article_title) > 0:
                                    articles[lang + '_title'].append(article_title)
                                    articles[lang + '_text': ].append(parse_and_clean_wikicode(text))
    print("Done parsing. ")
    df = pd.DataFrame.from_dict(articles)
    csv_file = filepath[:-4] + ".csv"
    df.to_csv(csv_file, index=False)
    print("Done! Saved Wiki articles as", csv_file, "!")
    return csv_file


def align_wikipedia_titles(xml_path, lang_pair='de-en'):
    """ Extracts titles for aligned multilingual Wikipedia articles (https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/) and saves result to CSV"""
    print("Language pair:", lang_pair.upper())
    languages = lang_pair.split('-')
    lang1 = languages[0]
    lang2 = languages[1]
    df = {lang+"_title": [] for lang in languages}
    wiki = open(xml_path, 'r')
    tree = ET.parse(wiki)
    root = tree.getroot()
    for child in root:
        if len(df[lang1+"_title"]) > 10000:
            break
        if 'article' in child.tag:
            article = child
            lang1_art_title = article.attrib['name'].lower()
            for child2 in article:
                if 'crosslanguage_link' in child2.tag:
                    cross_lang = child2
                    lang_attrib = cross_lang.attrib['language']
                    if lang_attrib == lang2:
                        lang2_art_title = cross_lang.attrib['name'].lower()
                        df[lang1+"_title"].append(lang1_art_title)
                        df[lang2+"_title"].append(lang2_art_title)
    print("Links:", len(df[lang1+"_title"]))
    save_filename = 'wikipairs_titles_'+lang_pair+'.csv'
    df = pd.DataFrame.from_dict(df)
    df.to_csv(save_filename, index=None)
    print("Done! Dumped aligned titles for pair", lang_pair.upper(), "to", save_filename, "!")


def combine_multilingual_wikipedia_articles(aligned_titles, lang1_articles, lang2_articles, lang1='de', lang2='en'):
    """ Merges the aligned multilingual titles and extracted article contents """
    df_merged = lang1_articles.merge(aligned_titles, on=[lang1 + "_title"])
    df_merged = df_merged.merge(lang2_articles, on=[lang2 + "_title"])
    return df_merged


def extract_wikipedia_image_urls(image_tsv='train-00000-of-00005.tsv', lang='en'):
    """ Extracts image urls and article titles for a given language from the WIT dataset"""
    chunksize = 100000
    df_reduced = {'image_url': [],
                  lang + '_title': []}
    with pd.read_csv(image_tsv, sep='\t', chunksize=chunksize) as reader:
        for i, chunk in enumerate(reader):
            df = chunk[chunk.language == lang]
            df_reduced['image_url'].extend(list(df.image_url))
            page_titles = list(df.page_title)
            page_titles = [title.lower() for title in page_titles]
            df_reduced[lang + '_title'].extend(list(page_titles))
    df_reduced = pd.DataFrame.from_dict(df_reduced)
    save_path = image_tsv[:-4] + '-' + lang + '.csv'
    df_reduced.to_csv(save_path, index=False)


wiki_corpus = "/users/zosaelai/project_dir/datasets/wiki/fiwiki-20181001-corpus.xml"
align_wikipedia_titles(xml_path=wiki_corpus, lang_pair='fi-de')