import os
import sys
import argparse
import heapq
import pandas as pd
import pickle as pkl
from embedding_utils import EmbeddingLoader
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.model_selection._search import BaseSearchCV


def print_cv_result(result, n):
    if isinstance(result, BaseSearchCV):
        result = result.cv_results_

    scores = result['mean_test_score']
    params = result['params']

    if n < 0:
        n = len(scores)

    print("Cross Validation result in descending order: (totalling {} trials)".format(n))
    for rank, candidate, in enumerate(heapq.nlargest(n, zip(scores, params), key=lambda tup: tup[0])):
        print("rank {}, score = {}\n hyperparams = {}".format(rank + 1, *candidate))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="parent dir to load embeddings")
    parser.add_argument("--output", required=True, help="parent dir to dump search results")
    # uses python reflection to dynamically load model
    parser.add_argument("--classifier", required=True,
                        help="classifier to use, must be existent under model/, such as model/KNN.py")
    parser.add_argument("--corpus", default="title", help="title, text, or concatenated")
    parser.add_argument("--embedding", default="d2v",
                        help="embeddings model to use, must be one of [d2v, nd2v, onehot], default is d2v")
    parser.add_argument("--n_iter", default=100, type=int, help="number of trials to run during cross-validation. "
                                                                "default=100. This is NOT epochs to train d2v")
    parser.add_argument("--n_jobs", default=1, type=int, help="number of cpu workers to run in parallel")
    parser.add_argument("--cv", default=5, type=int, help="number of folds for cross-validation, default=5")
    # hyperparameters for doc2vec
    parser.add_argument("--vec_size", default=300, type=int,
                        help="size of vectors, default is 300, recommended to be left untouched")
    parser.add_argument("--win_size", default=13, type=int,
                        help="window size, used if model is d2v, default = 13")
    parser.add_argument("--min_count", default=5, type=int,
                        help="min count for inclusion in dict, used if model is d2v, default = 5")
    parser.add_argument("--dm", action="store_true",
                        help="whether to use DM or DBOW, used if model is d2v, default is DBOW")
    parser.add_argument("--epochs", default=100, type=int,
                        help="number of epochs to train the model for, used if model is d2v, default = 100. This is "
                             "NOT the epochs for RandomizedSearch")
    # hyperparameters for naive doc2vec
    parser.add_argument("--normalizer", default=None,
                        help="normalizer for naive doc2vec, either l2 or mean, default is None")
    # hyperparameters for one-hot
    parser.add_argument("--scorer", default="count",
                        help="scorer function for one-hot, either tfidf or count, default is count")

    opt = parser.parse_args()
    print(opt)

    loader = EmbeddingLoader(opt.input)

    # filename is saved for dumping CV results later
    if opt.embedding == "d2v":
        filename = loader.get_d2v_filename(corpus=opt.corpus, vec_size=opt.vec_size, win_size=opt.win_size,
                                           min_count=opt.min_count, dm=opt.dm, epochs=opt.epochs)
        embeddings = loader.get_d2v(corpus=opt.corpus, vec_size=opt.vec_size, win_size=opt.win_size,
                                    min_count=opt.min_count, dm=opt.dm, epochs=opt.epochs)
    elif opt.embedding == "nd2v":
        filename = loader.get_nd2v_filename(corpus=opt.corpus, normalizer=opt.normalizer)
        embeddings = loader.get_nd2v(corpus=opt.corpus, normalizer=opt.normalizer)
    elif opt.embedding == "onehot":
        filename = loader.get_onehot_filename(corpus=opt.corpus, scorer=opt.scorer, normalize=opt.normalize is not None)
        embeddings = loader.get_onehot(corpus=opt.corpus, scorer=opt.scorer, normalize=opt.normalize is not None)
    else:
        print("unrecognized embedding method: {}; proceed with d2v as fall back".format(opt.embedding))
        filename = loader.get_d2v_filename(corpus=opt.corpus, vec_size=opt.vec_size, win_size=opt.win_size,
                                           min_count=opt.min_count, dm=opt.dm, epochs=opt.epochs)
        embeddings = loader.get_d2v(corpus=opt.corpus, vec_size=opt.vec_size, win_size=opt.win_size,
                                    min_count=opt.min_count, dm=opt.dm, epochs=opt.epochs)

    labels = loader.get_label()

    seed = 0
    embeddings_train, embeddings_test, labels_train, labels_test = \
        train_test_split(embeddings, labels, test_size=0.25, random_state=seed, stratify=labels)

    # import the target file
    try:
        module = __import__(opt.classifier)
    except ModuleNotFoundError as e:
        print(e)
        print("There is no such file, double check that you have a `model/{}.py`".format(opt.classifier))
        print("If you have checked and the problem persist, make sure to run this script from ROOTDIR instead of "
              "ROOTDIR/model, your code should look like `python model/hypertune.py ...`")
        sys.exit(0)

    # get the model from the target file
    try:
        model = getattr(module, "model")
    except AttributeError as e:
        print(e)
        print("There is no `model` attribute in `model/{}.py`".format(opt.classifier))
        print("Make sure to include a variable named `model` in your file")
        sys.exit(0)

    # get the hyperparameters to be trained
    try:
        param_dist = getattr(module, "param_dist")
    except AttributeError as e:
        print(e)
        print("There is no `param_dist` attribute in `model/{}.py`".format(opt.classifier))
        print("Make sure to include a variable named `param_dist` in your file")
        sys.exit(0)

    verbose = opt.cv * opt.n_iter
    searcher = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=opt.n_iter, scoring='f1', cv=opt.cv,
                                  verbose=verbose, random_state=seed, error_score=0, return_train_score=False,
                                  n_jobs=opt.n_jobs)
    searcher.fit(embeddings_train, labels_train)

    print("best: {}\n{}\n{}\n{}".format(searcher.best_index_, searcher.best_score_, searcher.best_estimator_,
                                        searcher.best_params_))
    # The following line is meant for floydhub renderer to grep
    print('{"metric": "highest_val", "value": %f}' % searcher.best_score_)

    results = pd.DataFrame(searcher.cv_results_)

    filename_classifier = opt.classifier
    dump_filename = "{}-{}".format(opt.classifier, filename)
    with open(os.path.join(opt.output, dump_filename), "wb") as f:
        pkl.dump(results, f)

    print_cv_result(results, n=-1)

    # uses all training samples to refit the model
    searcher.best_estimator_.fit(embeddings_train, labels_train)
    test_score = searcher.best_estimator_.score(embeddings_test, labels_test)
    print("Final test score of the best performing model: {}".format(test_score))

    # The following line is meant for floydhub renderer to grep
    print('{"metric": "test", "value": %f}' % test_score)
