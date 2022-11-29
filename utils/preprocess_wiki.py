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


def parse_wikipedia_xml_dump(xml_path):
    """ Extracts title and article content from Wikipedia XML dumps and saves result to CSV"""
    filepath = xml_path
    f = open(filepath, 'r')
    tree = ET.parse(f)
    root = tree.getroot()
    articles = {'title':[], 'content':[]}
    article_count = 0
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
                                    article_count += 1
                                    articles['title'].append(article_title)
                                    articles['content'].append(parse_and_clean_wikicode(text))
    print("Done parsing.")
    df = pd.DataFrame.from_dict(articles)
    csv_file = filepath[:-4] + ".csv"
    df.to_csv(csv_file, index=False)
    print("Done! Saved Wiki articles as", csv_file, "!")
    return csv_file




