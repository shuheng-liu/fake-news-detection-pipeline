import os
import pickle as pkl


class EmbeddingLoader:
    def __init__(self, parent_dir="embeddings"):
        """
        inits the loader and save the directory where embeddings are saved
        :param parent_dir: the directory where all embeddings are saved
        """
        assert os.path.isdir(parent_dir), "{} is not a valid directory".format(parent_dir)
        self.parent_dir = parent_dir

    @staticmethod
    def get_file(path):
        try:
            with open(path, "rb") as f:
                item = pkl.load(f)
        except FileNotFoundError as e:
            print("unable to load {}, see stack trace below".format(path))
            print("double check that you have the file save {}".format(path))
            print(e)
            return None

        return item

    @staticmethod
    def get_onehot_filename(corpus="title", scorer="count", normalize=False):
        return "{}-onehot(scorer={}{}).pkl".format(corpus, scorer, ", normalized" if normalize else "")

    @staticmethod
    def get_d2v_filename(corpus="title", vec_size=300, win_size=13, min_count=5, dm=False, epochs=100):
        return "{}-d2v(vecsize={}, winsize={}, mincount={}, {}, epochs={}).pkl".format(
            corpus, vec_size, win_size, min_count, "dm" if dm else "dbow", epochs
        )

    @staticmethod
    def get_nd2v_filename(corpus="title", normalizer=None):
        return "{}-nd2v(normalizer={}).pkl".format(corpus, normalizer)

    def get_onehot(self, corpus="title", scorer="count", normalize=False):
        """
        returns the onehot sum matrix
        :param corpus: str, either "title" or "text"
        :param scorer: str, either "count" or "tfidf"
        :param normalize: bool, if set to True, normalized embeddings are returned
        :return: scipy.sparse.csr_matrix, as the one-hot sum vector of the corpus
        """
        assert corpus in ["title", "text"], "`corpus` must be either 'title' or 'text'"
        assert scorer in ["count", "tfidf"], "`scorer` must be either 'count' or 'tfidf'"
        assert isinstance(normalize, bool), "`normalize` must be a bool"
        # filename = "{}-onehot(scorer={}{}).pkl".format(corpus, scorer, ", normalized" if normalize else "")
        filename = EmbeddingLoader.get_onehot_filename(corpus=corpus, scorer=scorer, normalize=normalize)
        return EmbeddingLoader.get_file(os.path.join(self.parent_dir, filename))

    def get_d2v(self, corpus="title", vec_size=300, win_size=13, min_count=5, dm=False, epochs=100):
        """
        returns the d2v embeddings matrix
        :param corpus: str, either "title" or "text"
        :param vec_size: length of vector, default=300, best left untouched
        :param win_size: wndow_size, default=13, only win_size=13 is computed so far
        :param min_count: min_count to be included in dictionary, only min_count=5, 25, 50 are computed so far
        :param dm: int or bool, denotes whether use DM or DBOW
        :param epochs: number of epochs, only epochs=100 is computed so far
        :return: numpy.ndarray, shape=(n_docs, n_dims), as the d2v embeddings matrix
        """
        assert corpus in ["title", "text"], "`corpus` must be either 'title' or 'text'"
        # filename = "{}-d2v(vecsize={}, winsize={}, mincount={}, {}, epochs={}).pkl".format(
        #     corpus, vec_size, win_size, min_count, "dm" if dm else "dbow", epochs
        # )
        filename = EmbeddingLoader.get_d2v_filename(corpus=corpus, vec_size=vec_size, win_size=win_size,
                                                    min_count=min_count, dm=dm, epochs=epochs)
        return EmbeddingLoader.get_file(os.path.join(self.parent_dir, filename))

    def get_nd2v(self, corpus="title", normalizer=None):
        """
        returns the naive d2v embeddings matrix
        :param corpus: str, either "title" or "text"
        :param normalizer: str, either "l2", "mean" or None
        :return: numpy.ndarray, shape=(n_docs, n_dims), as the naive d2v embeddings matrix
        """
        assert corpus in ["title", "text"], "`corpus` must be either 'title' or 'text'"
        assert normalizer is None or normalizer in ["l2", "mean"], "`normalizer` must be 'l2', 'mean' or None"
        # filename = "{}-nd2v(normalizer={}).pkl".format(corpus, normalizer)
        filename = EmbeddingLoader.get_nd2v_filename(corpus=corpus, normalizer=normalizer)
        return EmbeddingLoader.get_file(os.path.join(self.parent_dir, filename))

    def get_label(self):
        """
        returns the labels, if you have a 'label.pkl' nested under `self.parent_dir`
        :return: numpy.ndarray, shape=(n_docs,), as the label vector (0 for REAL and 1 for FAKE)
        """
        return EmbeddingLoader.get_file(os.path.join(self.parent_dir, "label.pkl"))

# if __name__ == '__main__':
#     loader = EmbeddingLoader("embeddings/")
#     d2v = loader.get_d2v()
