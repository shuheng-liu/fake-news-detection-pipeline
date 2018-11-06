import nltk
from itertools import chain
from gensim.corpora import Dictionary
from gensim.models.doc2vec import TaggedDocument

# in case some packages are not properly installed
nltk.download('gutenberg')
nltk.download('reuters')
nltk.download('stopwords')
nltk.download("punkt")


class DocumentSequence:
    def __init__(self, raw_docs, clean=False, sw=None, punct=None):
        """
        an instance featuring difference representations of a doc sequence

        public methods are:
            self.get_dictionary()
            self.get_tokenized()
            self.get_tagged()
            self.get_bow()

        :param raw_docs: numpy.ndarray[str]
            each string for a document
        :param clean: bool
            whether to clean stopwords and punctuations
        :param sw: list[str]
            list of stopwords, only works if `clean` is True, default is empty
        :param punct: str
            string of punctuations, only works if `clean` is True, default is empty

        """
        self.raw_docs = raw_docs
        self._set_tokenized(clean=clean, sw=sw, punct=punct)
        self._set_tagged()

    def _set_tokenized(self, clean=False, sw=None, punct=None):
        """
        set self._tokenized to list[list[str]]: each string for a token
        :param clean: bool, whether to clean stopwords and punctuations
        :param sw: list[str], list of stopwords, only works if `clean` is True, default is empty
        :param punct: str, string of punctuations, only works if `clean` is True, default is empty
        """
        print("converting raw docs into tokens")

        # lower-casing all documents in the first step
        self._tokenized = [nltk.word_tokenize(doc.lower()) for doc in self.raw_docs]

        if clean:  # if clean is set to True, stopwords and punctuations are removed
            print("cleaning up stopwords and punctuations")
            # hashing stopwords and punctuations speeds up look-up computation
            if sw is None:  # default value of sw is None, corresponding to an empty list
                sw = []
            if punct is None:  # default value of punct is None, corresponding to an empty list
                punct = []
            skip_tokens = set(chain(sw, punct))
            print("all tokens to be skipped are: {}".format(skip_tokens))
            # retain only meaningful tokens, while preserving the structure
            self._tokenized = [[token for token in doc if token not in skip_tokens] for doc in self._tokenized]

    def _set_tagged(self):
        """set self._set_tagged to list[TaggedDocument] each TaggedDocument has a tag of [index]"""
        print("listing tagged documents in memory")
        self._tagged = [TaggedDocument(doc, tags=[index]) for index, doc in enumerate(self._tokenized)]

    def _set_dictionary(self):
        """stores the dictionary of current corpus"""
        self._dictionary = Dictionary(self._tokenized)

    def _set_bow(self):
        """set self._bow to list[list[tuple]], where each tuple is (word_id, word_frequency)"""
        if not hasattr(self, '_dictionary'):  # check whether dictionary is set or not
            print("dictionary is not set for {}, setting dictionary automatically".format(self))
            self._set_dictionary()
        self._bow = [self._dictionary.doc2bow(doc) for doc in self._tokenized]

    def get_dictionary(self):
        """getter for class attribute dictionary"""
        if not hasattr(self, "_dictionary"):  # self._dictionary is only computed once
            self._set_dictionary()

        # the previous method is only called once
        return self._dictionary

    dictionary = property(get_dictionary)

    def get_tokenized(self):
        """getter for tokenized documents, cleaned as desired"""
        return self._tokenized

    tokenized = property(get_tokenized)

    def get_tagged(self):
        """getter for list of TaggedDocuments"""
        return self._tagged

    tagged = property(get_tagged)

    def get_bow(self):
        """getter for bag-of-words representation of documents"""
        if not hasattr(self, '_bow'):  # self._bow is only computed lazily
            self._set_bow()

        # the previous method is only called once
        return self._bow

    bow = property(get_bow)
