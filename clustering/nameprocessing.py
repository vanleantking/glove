import re

class PostClusterProcessing:
    _vowels = ['a', 'e', 'i', 'o', 'u']
    _special_symbols = ['!','.',':', '-', '\'s', '+', '_', '(', ')', '*', '&', '#', ';', '?', '>', '<', '%']
        
     
    def abbreviation(self, name):
        name_processed = self.preprocess(name)
        abb = []
        abbreviation = all(char not in self._vowels for c in name_processed for char in c)
        if abbreviation:
            abb = [c for c in name_processed for char in c if c.strip() != '']
        else:
            try:
                abb = [c[0] for c in name_processed.split(' ') if c[0].strip() != '']
            except: print(name_processed)

        return abb, abbreviation

    def check_equal_name(self, name1, name2):
        name1_processed = self.preprocess(name1)
        name2_processed = self.preprocess(name2)
        abb_name1, is_abb1 = self.abbreviation(name1)
        abb_name2, is_abb2 = self.abbreviation(name2)

        if is_abb1 == False and is_abb2 == False:
            if set(name1_processed.split(' ')).issubset(name2_processed.split(' ')) or set(name2_processed.split(' ')).issubset(name1_processed.split(' ')):
                return True
            return False

        if len(abb_name2) > 1 and len(abb_name1) > 1:
            if set(abb_name2).issubset(abb_name1) or set(abb_name1).issubset(abb_name2):
                return True


        return False

    def preprocess(self, name):
        phrases = "".join(c for c in name if c not in self._special_symbols)
        phrases = "".join(c for c in phrases if c not in ('!','.',':', '-', '+', '_', '(', ')', '*', '&', '#', ';', '?', '>', '<', '%', '{', '}', '=', ',', ']', '[', '`', '\''))
        phrases = re.sub(' +',' ',phrases)
        return phrases.lower()

    def mergecluster(self, cluster1, cluster2):
        return any(self.check_equal_name(element1.hasName[0], element2.hasName[0]) for element1 in cluster1 for element2 in cluster2)


# if __name__ == '__main__':
    # np = PostClusterProcessing()
    # print(np.abbreviation('knox'))
    # print(np.abbreviation('kang'))
    # print(np.abbreviation('aa aa'))
    # print(np.abbreviation('tr rt s'))
    # print(np.abbreviation('eer fr'))
    # print(np.abbreviation('JK'))
    # print('Yechiel'.split(' '))
    # print(set('knox'.split(' ')).issubset('kang'.split(' ')) or set('kang'.split(' ')).issubset('knox'.split(' ')))
    # print(set('knox'.split(' ')).issubset('Hope Kincaid'.split(' ')) or set('Hope Kincaid'.split(' ')).issubset('knox'.split(' ')))
    # print(set('kang'.split(' ')).issubset('Hope Kincaid'.split(' ')) or set('Hope Kincaid'.split(' ')).issubset('kang'.split(' ')))
    # np.check_equal_name('Ida Xayachack', 'Xayachack, Ida')

