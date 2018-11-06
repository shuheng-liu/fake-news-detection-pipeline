import numpy as np
from gensim.models import TfidfModel, Word2Vec, Doc2Vec, KeyedVectors
from scipy.sparse import csr_matrix
from .document_sequence import DocumentSequence


def normalized(arr):
    """
    normalize the input and return it
    :param arr: numpy.ndarray, or a scalar
        if numpy.ndarray, it is L2-normalized and returned
        if scalar, 1 is returned (even for 0 as input)
    :return: L2-normalized ndarray, or a scalar
    """
    if isinstance(arr, (int, float)):  # if input is scalar
        # any nonzero scalar is normalized to 1
        # therefore so should 0, by methods of continuation
        return 1

    # get the norm of the array
    norm = np.linalg.norm(arr, ord=1)
    if norm == 0:
        norm = np.finfo(arr.dtype).eps
    return arr / norm


def get_onehot_arr(place, dim, put_value=1.):
    """
    get a `dim` dimensional one-hot vector, with `place`-th entry being `put_value` and dtype being np.float32
    e.g.:
        >>> get_onehot_arr(3, 5, 1.3)
        np.ndarray([0, 0, 0, 1.3, 0], dtype=np.float32)
    :param place: the place to put a non-zero value
    :param dim: the length of the vector
    :param put_value: the value to be put
    :return: a `dim` dimensional one-hot vector, with `place`-th entry being `put_value` and dtype being np.float32
    """
    if place >= dim or place < 0:
        print("Invalid input: place = {}, dim = {}".format(place, dim))
    ans = np.zeros(dim, dtype=np.float32)
    np.put(ans, place, put_value)
    return ans


class DocumentEmbedder:
    def __init__(self, docs: DocumentSequence, pretrained_word2vec=None):
        """
        This class features interfaces to different methods of computing document embeddings.
        Supported embedding mechanisms are:
            Dov2Vec:                               see self.get_doc2vec()
            Naive Doc2Vec:                         see self.get_naive_doc2vec()
            One-Hot Sum:                           see self.get_onehot()
            Attention is all you need              To be implemented
            FastText                               To be implemented

        :param docs: a DocumentSequence instance
        :pretrained_word2vec: path to pretrained word2vec model, in .bin format
        """
        self.docs = docs
        self.pretrained = pretrained_word2vec

    def _set_word2vec(self):
        if self.pretrained is None:
            raise ValueError("Pretrained word2vec path is not specified during instantiation")
        self._w2v = KeyedVectors.load_word2vec_format(self.pretrained, binary=True)

    def _set_doc2vec(self, vector_size=300, window=5, min_count=5, dm=1, epochs=20):
        # instantiate a Doc2Vec model, setting pretrained GoogleNews Vector
        self._d2v = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, dm=dm, epochs=epochs,
                            pretrained=self.pretrained)
        # build vocabulary from corpus
        self._d2v.build_vocab(self.docs.get_tagged())

        # somehow, the training won't start automatically, and must be manually started
        self._d2v.train(self.docs.get_tagged(), total_examples=self._d2v.corpus_count, epochs=epochs)

        # list document embeddings by order of their tags
        self._d2v_embedding = np.stack(self._d2v.docvecs[index] for index in range(len(self.docs.get_tagged())))

    def _set_naive_doc2vec(self, normalizer='l2'):
        if not hasattr(self, '_w2v'):  # load pretrained word2vec lazily
            self._set_word2vec()

        dim = self._w2v.vector_size

        # The naive doc2vec method first adds up word embeddings in a document, then performs normalization
        # supported normalizers are l2, mean and None

        if normalizer == 'l2':  # normalization by L2 norm
            self._naive_d2v_embedding = [
                normalized(np.sum(self._w2v[tok] if tok in self._w2v else np.zeros(dim) for tok in doc))
                for doc in self.docs.get_tokenized()
            ]

        elif normalizer == "mean":  # normalization by number of tokens
            self._naive_d2v_embedding = [
                np.sum(self._w2v[tok] if tok in self._w2v else np.zeros(dim) for tok in doc) / max(len(doc), 1)
                for doc in self.docs.get_tokenized()
            ]

        else:  # not using normalization at all
            self._naive_d2v_embedding = [
                np.sum(self._w2v[tok] if tok in self._w2v else np.zeros(dim) for tok in doc)
                for doc in self.docs.get_tokenized()
            ]

        # convert list of naive doc2vec embeddings into numpy.ndarray
        self._naive_d2v_embedding = \
            np.stack(emb if isinstance(emb, np.ndarray) and emb.shape[0] == dim else np.zeros(dim)
                     for emb in self._naive_d2v_embedding)

    def _set_tfidf(self):
        self._tfidf = TfidfModel(corpus=self.docs.get_bow())
        self._tfidf_score = [[(index, score) for index, score in self._tfidf[doc]] for doc in self.docs.get_bow()]

    def _set_onehot(self, scorer='tfidf'):
        # The dimension of one hot vectors is equal to the number of tokens, i.e., dictionary size
        dim = len(self.docs.get_dictionary())

        if scorer == 'tfidf':  # if using tf-idf scorer, try to compute tf-idf lazily
            if not hasattr(self, '_tfidf_score'):  # the tf-idf score is computed only once
                self._set_tfidf()
            self._onehot_embedding = [np.sum(get_onehot_arr(word_id, dim, tfidf_score) for word_id, tfidf_score in doc)
                                      for doc in self._tfidf_score]

        elif scorer == 'count':  # if using raw counts, the weight of each vector is its term frequency
            self._onehot_embedding = [np.sum(get_onehot_arr(word_id, dim, word_count) for word_id, word_count in doc)
                                      for doc in self.docs.get_bow()]

        else:  # if scorer is not specified, use raw count as default option
            print("scorer not specified, using raw count")
            self._onehot_embedding = [np.sum(get_onehot_arr(word_id, dim, word_count) for word_id, word_count in doc)
                                      for doc in self.docs.get_bow()]

        # convert list of one-hot SUM vectors into sparse matrix
        self._onehot_embedding = csr_matrix(np.stack(self._onehot_embedding))

    # TODO implement setter and getter for fastText
    def _fast_text(self):
        raise NotImplementedError("To be implemented: fast_text")

    # TODO implement setter and getter for attention-is-all-you-need
    def _attention(self):
        raise NotImplementedError("To be implemented: attention-is-all-you-need")

    def get_onehot(self, scorer='tfidf'):
        """
        get the sum of one-hot embeddings weighted by a scorer in each document
        Note: tokens not included in pretrained GoogleNews vectors will be assigned 0 as their embedding

        :param scorer: str, either 'tfidf' or 'count'
            if 'tfidf' the one-hot vectors are weighted by their term frequency and log(inverse document frequency)
            if 'count' the one-hot vectors are weighted by their raw count
        :return: a list of document embeddings, vector size = number of tokens
        """
        if not hasattr(self, '_onehot_embedding'):
            self._set_onehot(scorer=scorer)

        return self._onehot_embedding

    onehot = property(get_onehot)

    def get_doc2vec(self, vectors_size=300, window=5, min_count=5, dm=1, epochs=20):
        """
        get the doc2vec embeddings with word vectors pretrained on GoogleNews task
        :param vectors_size: size for document embeddings, should be 300 if using GoogleNews pretrained word vectors
        :param window: number of tokens to be include in both directions
        :param min_count: lower threshold for a token to be included
        :param dm: using distributed memory or not
            if 1, use distributed memory
            if 0, use distributed bag of words
        :param epochs: number of epochs for training, usually < 20
        :return: a list of document embeddings, vector size can be specified
        """
        if vectors_size != 300:
            print("Warning: pretrained Google News vecs have length 300, got vec-size={} ".format(vectors_size))

        if not hasattr(self, '_d2v_embedding'):
            self._set_doc2vec(vector_size=vectors_size, window=window, min_count=min_count, dm=dm, epochs=epochs)

        return self._d2v_embedding

    doc2vec = property(get_doc2vec)

    def get_naive_doc2vec(self, normalizer='l2'):
        """
        get the naive doc2vec embeddings, which is obtained from summing word vectors and normalizing by a metric
        :param normalizer: str or None
            if 'l2', the sum of word vectors are normalized to fall on the surface of an d-dimensional ball
            if 'mean', the sum of word vectors are divided by the number of words
            if None or otherwise, the sum of word vectors are not normalized (unit normalizer)
        :return: a list of document embeddings, vector size equal to pretrained word vector size
        """
        if not hasattr(self, '_naive_d2v_embedding'):
            self._set_naive_doc2vec(normalizer=normalizer)

        return self._naive_d2v_embedding

    naive_doc2vec = property(get_naive_doc2vec)

    def get_tfidf_score(self):
        if not hasattr(self, "_tfidf_score"):
            self._set_tfidf()

        return self._tfidf_score

    tfidf = property(get_tfidf_score)
