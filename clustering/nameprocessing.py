import re

class PreProcessingText:
    _vowels = ['a', 'e', 'i', 'o', 'u']
    _special_symbols = ['!','.',':', '-', '+', '_', '(', ')', '*', '&', '#', ';', '?', '>', '<', '%', '{', '}', '=', ',', ']', '[', '`', '\'']

        
     
    def abbreviation(self, name, is_username = False):
        name_processed = self.preprocess(name)
        abb = []
        abbreviation = all(char not in self._vowels for c in name_processed for char in c)
        if is_username or (abbreviation and is_username == False) :
            abb = [c for c in name_processed for char in c if c.strip() != '']
        else:
            abb = [c[0] for c in name_processed.split(' ') if c != ' ' and c[0].strip() != '']

        return abb, abbreviation

    def preprocess(self, name):
        phrases = name.strip()
        phrases = "".join(c for c in name if c not in self._special_symbols)
        phrases = "".join(c for c in phrases if not re.search(r'^/[/]?', c))

        # # # #remove ________ in text
        phrases = "".join(c for c in phrases if not re.search(r'_+', c))

        # # # #split string 123tert to 123 tert
        phrases = "".join(c for word in phrases for c in re.split(r'([0-9]*)([a-zA-Z\'0-9]+)', word) if c)
        # # #phrases #remove measure weight/digits in sentence
        phrases = "".join(word for word in phrases if not re.search(r'^\d+\.*\s?\d*\s?[mg]*$', word))
        phrases = "".join(c for c in phrases if not re.search(r'^mg', c))
        
        phrases = re.sub(' +',' ',phrases)
        return phrases.lower().strip()

class PostProcessing:

    def __init__(self, _pre):
        self._pre = PreProcessingText()

    def mergecluster(self, cluster1, cluster2):
        raise NotImplementedError("Subclass must implement abstract method")

    def check_equal_name(self, name1, name2):
        raise NotImplementedError("Subclass must implement abstract method")

class PostNameClusterProcessing(PostProcessing):

    def __init__(self, _pre):
        super().__init__(_pre)

    def check_equal_name(self, name1, name2):
        name1_processed = self._pre.preprocess(name1)
        name2_processed = self._pre.preprocess(name2)
        abb_name1, is_abb1 = self._pre.abbreviation(name1)
        abb_name2, is_abb2 = self._pre.abbreviation(name2)

        if is_abb1 == False and is_abb2 == False:
            if set(name1_processed.split(' ')).issubset(name2_processed.split(' ')) or set(name2_processed.split(' ')).issubset(name1_processed.split(' ')):
                return True
            return False

        if len(abb_name2) > 1 and len(abb_name1) > 1:
            if set(abb_name2).issubset(abb_name1) or set(abb_name1).issubset(abb_name2):
                return True

        return False

    def mergecluster(self, cluster1, cluster2):
        return any(self.check_equal_name(element1.hasName[0], element2.hasName[0]) for element1 in cluster1 for element2 in cluster2)

class PostProfessionClusterProcessing(PostProcessing):

    def __init__(self, _pre):
        super().__init__(_pre)

    def check_equal_name(self, name1, name2):
        name1_processed = self._pre.preprocess(name1)
        name2_processed = self._pre.preprocess(name2)
        abb_name1, is_abb1 = self._pre.abbreviation(name1)
        abb_name2, is_abb2 = self._pre.abbreviation(name2)

        if is_abb1 == False and is_abb2 == False:
            if set(name1_processed.split(' ')).issubset(name2_processed.split(' ')) or set(name2_processed.split(' ')).issubset(name1_processed.split(' ')):
                return True
            return False

        if len(abb_name2) > 1 and len(abb_name1) > 1:
            if set(abb_name2).issubset(abb_name1) or set(abb_name1).issubset(abb_name2):
                return True

        return False
    def mergecluster(self, cluster1, cluster2):
        return any(self._pre.check_equal_name(element1.jobName[0], element2.jobName[0]) for element1 in cluster1 for element2 in cluster2)

