import re

class PreProcessingText:
    _vowels = ['a', 'e', 'i', 'o', 'u']
    _special_symbols = ['!','.',':', '-', '+', '_', '(', ')', '*', '&', '#', ';', '?', '>', '<', '%', '{', '}', '=', ',', ']', '[', '`', '\'']

        
     
    def abbreviation(self, name, is_username = False):

        if is_username:
            return name, False

        name_processed = self.preprocess(name)
        abb = []
        isconsonant = all(char not in self._vowels for c in name_processed for char in c)
        len_name = name.strip().split()       

        #len is large than 1 mean name is not abbreviation, or name with length = 1 (example: Xue, Freddy)
        if (len(len_name) == 1 and (name.strip() != name.strip().upper()) and isconsonant == False) or len(len_name) > 1:
            abb = [c[0] for c in name_processed.split() if c != ' ' and c[0].strip() != '']


        #all consonant mean name is abbreviation or process for name with length = 1 with\
        # all upper (for both name = 1(XUE, FREDDY..) and name is abbreviation ready)
        elif isconsonant == True or (len(len_name) == 1 and name.strip() == name.strip().upper()):
            abb = [c for c in name_processed for char in c if c.strip() != '']



        return abb, isconsonant

    def preprocess(self, name, is_username = False):
        phrases = name.strip()
        if is_username:
            return phrases

        phrases = re.findall(r"[\w']+|[.,!?;\/+]", phrases.lower())
        phrases = [c for word in phrases for c in re.split(r'\'s', word) if c]
        phrases = [c for c in phrases if c not in self._special_symbols]
        phrases = [c for c in phrases if not re.search(r'^/[/]?', c)]

        # # # #remove ________ in text
        phrases = [c for c in phrases if not re.search(r'_+', c)]

        # # # #split string 123tert to 123 tert
        phrases = [c for word in phrases for c in re.split(r'([0-9]*)([a-zA-Z\'0-9]+)', word) if c]
        # # #phrases #remove measure weight/digits in sentence
        phrases = [word for word in phrases if not re.search(r'^\d+\.*\s?\d*\s?[mg]*$', word)]
        phrases = [c for c in phrases if not re.search(r'^mg', c)]
        phrases = " ".join(phrases)
        
        phrases = re.sub(' +',' ',phrases)
        return phrases.strip()

class PostProcessing:

    def __init__(self):
        self._pre = PreProcessingText()

    def mergecluster(self, cluster1, cluster2):
        return all(self.check_equal_name(element1.name, element2.name) for element1 in cluster1 for element2 in cluster2)
        

    # check abbrev is abbreviation of text
    def is_abbrev(self, abbrev, text):
        abbrev = abbrev.lower()
        text=text.lower()
        words=text.split()
        if not abbrev:
            return True

        if abbrev and not text:
            return False

        if abbrev[0] != text[0]:
            return False
        else:
            return (self.is_abbrev(abbrev[1:],' '.join(words[1:])))

    # check abbrev is abbreviation of text
    def check_equal_name(self, name1, name2):
        name1_processed = self._pre.preprocess(name1)
        name2_processed = self._pre.preprocess(name2)
        abb_name1, is_abb1 = self._pre.abbreviation(name1)
        abb_name2, is_abb2 = self._pre.abbreviation(name2)

        if is_abb1 == False and is_abb2 == False:
            if set(name1_processed.split()).issubset(name2_processed.split()) or set(name2_processed.split()).issubset(name1_processed.split()):
                return True
            return False

        if len(abb_name2) > 1 and len(abb_name1) > 1:
            if abb_name1 == abb_name2:
                return True

        return False

    #do merge by condition
    def check_equal(self, name1, name2):
        name1_processed = self._pre.preprocess(name1)
        name2_processed = self._pre.preprocess(name2)
        len_name1 = name1.strip().split()
        len_name2 = name2.strip().split()

        if len(len_name1) == 1 and self.is_abbrev(name1_processed, name2_processed):
            return True

        if len(len_name2) == 1 and self.is_abbrev(name2_processed, name1_processed):
            return True

        return self.check_equal_name(name1, name2)

    def splitcluster(self, clusters):
        result = []
        for index, cluster in clusters.items():
            lencls = len(cluster)
            if lencls > 1:
                for i in range(lencls):
                    try:
                        if len(cluster[i].hasName[0].split()) > 1:
                            if any(self.check_equal(cluster[i].hasName[0], cluster[j].hasName[0]) == False for j in range(lencls) if i != j):
                                result.append([cluster[i]])
                                del cluster[i]
                        else:
                            if all(self.check_equal(cluster[i].hasName[0], cluster[j].hasName[0]) == False for j in range(lencls) if i != j):
                                result.append([cluster[i]])
                                del cluster[i]
                    except:
                        pass
        result.extend(clusters.values())
        lenrsl = len(result)
        results = {i: [] for i in range(lenrsl)}
        for i in range(lenrsl):
            results[i].extend(result[i])
        return results, lenrsl

#post clustering for doctors and patients name
class PostNameClusterProcessing(PostProcessing):

    def mergecluster(self, cluster1, cluster2):
        is_valid = [True]
        is_merged = []
        for element2 in cluster2:
            if len(element2.hasName[0].split()) == 1:
                is_merged.append(any(self.check_equal(element1.hasName[0], element2.hasName[0]) for element1 in cluster1))
            else:
                merge_valid = all(self.check_equal(element1.hasName[0], element2.hasName[0]) for element1 in cluster1)
                if not merge_valid:
                    return False
                is_merged.append(merge_valid)
                # is_valid.append(any(self.check_equal(cluster1[0].hasName[0], element2.hasName[0]) == True for element2 in cluster2))

            
        return any(is_merged)



#post clustering for professions
class PostProfessionClusterProcessing(PostProcessing):

    def splitcluster(self, clusters):
        result = []
        for index, cluster in clusters.items():
            lencls = len(cluster)
            if lencls > 1:
                for i in range(lencls):
                    try:
                        if (any(self.check_equal(cluster[i].jobName[0], cluster[j].jobName[0]) == False for j in range(lencls) if cluster[i] and cluster[j] and i != j)):
                            result.append([cluster[i]])
                            del cluster[i]
                    except:
                        pass
        result.extend(clusters.values())
        lenrsl = len(result)
        results = {i: [] for i in range(lenrsl)}
        for i in range(lenrsl):
            results[i].extend(result[i])
        return results, lenrsl

    def mergecluster(self, cluster1, cluster2):
        return all(self.check_equal(element1.jobName[0], element2.jobName[0]) for element1 in cluster1 for element2 in cluster2)

#post clustering for hospital
class PostHospitalClusterProcessing(PostProcessing):

    def check_equal_name(self, name1, name2):
        name1_processed = self._pre.preprocess(name1)
        name2_processed = self._pre.preprocess(name2)
        abb_name1, is_abb1 = self._pre.abbreviation(name1)
        abb_name2, is_abb2 = self._pre.abbreviation(name2)

        
        if set(name1_processed.split()).issubset(name2_processed.split()) or set(name2_processed.split()).issubset(name1_processed.split()):
            return True

        if len(abb_name2) > 1 and len(abb_name1) > 1:
            if abb_name1 == abb_name2:
                return True

        return False

    def mergecluster(self, cluster1, cluster2):
        return all(self.check_equal(element1.hasName[0], element2.hasName[0]) for element1 in cluster1 for element2 in cluster2)


class PostLocationClusteringProcessing(PostProcessing):
    def splitcluster(self, clusters):
        result = []
        for index, cluster in clusters.items():
            lencls = len(cluster)
            if lencls > 1:
                for i in range(lencls):
                    try:
                        if (any(self.check_equal(cluster[i].hasLocation[0], cluster[j].hasLocation[0]) == False for j in range(lencls) if cluster[i] and cluster[j] and i != j)):
                            result.append([cluster[i]])
                            del cluster[i]
                    except:
                        pass
        result.extend(clusters.values())
        lenrsl = len(result)
        results = {i: [] for i in range(lenrsl)}
        for i in range(lenrsl):
            results[i].extend(result[i])
        return results, lenrsl

    def mergecluster(self, cluster1, cluster2):
        return all(self.check_equal(element1.hasLocation[0], element2.hasLocation[0]) for element1 in cluster1 for element2 in cluster2)


    # check abbrev is abbreviation of text
    def is_abbrev(self, abbrev, text):
        abbrev = abbrev.lower()
        text = text.lower()
        words = text.split()

        if not abbrev:
            return True

        if abbrev and not text:
            return False

        if abbrev[0]!=text[0]:
            return False
        else:
            return (self.is_abbrev(abbrev[1:],' '.join(words[1:])) or
                    any(self.is_abbrev(abbrev[1:],text[i+1:])
                        for i in range(len(words[0]))))

if __name__ == '__main__':
    p = PostNameClusterProcessing()
    print(p.check_equal('Quindarrius', 'Ervin, Quindarrius'))

    pre = PreProcessingText()
    print(pre.abbreviation("MI", False))

