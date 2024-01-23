import nltk
from nltk.corpus import wordnet,stopwords
import contractions
import re
from bs4 import BeautifulSoup

class CleanMsg():
    def __init__(self, project):
        self.project = project
        
    def excute(self, text):
        clean_text = text
        clean_text = clean_text.replace('\n'," ").replace('\r'," ")
        clean_text = nltk.sent_tokenize(clean_text)
        # clean_text = list(map(self.clean_html_and_js_tags, clean_text))
        clean_text = list(map(contractions.fix, clean_text))
        clean_text = list(map(self.lower_text, clean_text))
        clean_text = list(map(self.clean_url, clean_text))
        clean_text = list(map(self.clean_issue_id, clean_text))
        clean_text = list(map(self.clean_change_id, clean_text))
        clean_text = list(map(self.clean_commit_id, clean_text))
        clean_text = list(map(self.clean_hex, clean_text))
        clean_text = list(map(self.normalize_number, clean_text))
        clean_text = list(map(self.split_whitespace, clean_text))
        clean_text = list(map(self.exclude_stopwords, clean_text))
        # clean_text = list(map(self.lemmatize_text, clean_text))
        clean_text = list(map(self.integrate_text, clean_text))
        return clean_text
    
    def lemmatize_term(self, term, pos=None):
        if pos is None:
            synsets = wordnet.synsets(term)
            if not synsets:
                return term
            pos = synsets[0].pos()
            if pos == wordnet.ADJ_SAT:
                pos = wordnet.ADJ
        return nltk.WordNetLemmatizer().lemmatize(term, pos=pos)
    
    def lemmatize_text(self, text):
        cleaned_text = [self.lemmatize_term(word) for word in text]
        return cleaned_text
    
    def integrate_text(self, text):
        text = ' '.join(text)
        return text
    
    def split_whitespace(self, text):
        text_list = text.split(' ')
        text_list = [token for token in text_list if token != '']
        return text_list
    
    def tokenize(self, text):
        tokenized_text = nltk.word_tokenize(text)
        return tokenized_text
    
    def exclude_stopwords(self, text):
        cleaned_text = [word for word in text if not word in set(stopwords.words("english"))]
        return cleaned_text
    
    def clean_html_and_js_tags(self, text):
        soup = BeautifulSoup(text, 'html.parser')
        [x.extract() for x in soup.findAll(['script', 'style'])]
        cleaned_text = soup.get_text()
        cleaned_text = ''.join(cleaned_text.splitlines())
        return cleaned_text
    
    def lower_text(self, text):
        return text.lower()
    
    def normalize_number(self, text):
        """
        pattern = r'\d+'
        replacer = re.compile(pattern)
        result = replacer.sub('0', text)
        """
        # 連続した数字を0で置換
        replaced_text = re.sub(r'\d+', '[NUM]', text)
        return replaced_text

    def clean_hex(self, text):
        pattern = r'0x[0-9a-fA-F]+'
        clean_text = re.sub(pattern, '[NUM]', text)
        return clean_text
    
    def clean_commit_id(self, text):
        pattern = r'[0-9a-f]{40}'
        clean_text = re.sub(pattern, '[COMMITID]', text)
        return clean_text
    
    def clean_change_id(self, text):
        pattern = r'Change-Id.*(I[0-9a-f]{40})'
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            change_id_to_replace = match.group(1)
            result_text = text.replace(change_id_to_replace, '[CHANGEID]')
            return result_text
        else:
            return text
    
    def clean_url(self, text):
        clean_text = re.sub(r'http\S+', '[URL]', text)
        return clean_text
    
    def clean_issue_id(self, text):
        if self.project == 'qt':
            pattern = r'QTBUG-\d+'
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                change_id_to_replace = match.group(0)
                result_text = text.replace(change_id_to_replace, '[ISSUEID]')
                return result_text
            else:
                return text
        elif self.project == 'openstack':
            pattern = r'bug(|.*\b)(\d{6,})'
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                change_id_to_replace = match.group(2)
                result_text = text.replace(change_id_to_replace, '[ISSUEID]')
                return result_text
            else:
                return text
    