import pandas as pd
import numpy as np
import argparse
import pickle as pkl
import os
from tools import DocumentEmbedder, DocumentSequence
from nltk.corpus import stopwords
from string import punctuation

if __name__ == '__main__':
    # control arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="path to read csv file")
    parser.add_argument("--output", required=True,
                        help="dir to dump embeddigns, NOT including filename, created if non-existent")
    parser.add_argument("--column", default="title",
                        help="which column to select from the csv file, default is `title`")
    parser.add_argument("--model", default="d2v",
                        help="model to use, must be one of [d2v, nd2v, onehot, fasttext], default is d2v")
    # hyperparameters for doc2vec
    parser.add_argument("--vec_size", default=300,
                        help="size of vectors, default is 300, recommended to be left untouched")
    parser.add_argument("--pretrained", default=None,
                        help="path to word2vec model pretrained on Google News, used if model is d2v or nd2v")
    parser.add_argument("--win_size", default=5, type=int,
                        help="window size, used if model is d2v, default = 5")
    parser.add_argument("--min_count", default=5, type=int,
                        help="min count for inclusion in dict, used if model is d2v, default = 5")
    parser.add_argument("--dm", action="store_true",
                        help="whether to use DM or DBOW, used if model is d2v, default is DBOW")
    parser.add_argument("--epochs", default=20, type=int,
                        help="number of epochs to train the model for, used if model is d2v, default = 20")
    # hyperparameters for naive doc2vec
    parser.add_argument("--normalizer", default=None,
                        help="normalizer for naive doc2vec, either l2 or mean, default is None")
    # hyperparameters for one-hot
    parser.add_argument("--scorer", default="count",
                        help="scorer function for one-hot, either tfidf or count, default is count")

    opt = parser.parse_args()
    print(opt)

    df = pd.read_csv(opt.input)
    raw_docs = df[opt.column].values

    docs = DocumentSequence(raw_docs, clean=True, sw=stopwords.words('english'), punct=punctuation)
    embedder = DocumentEmbedder(docs, pretrained_word2vec=opt.pretrained)

    print('computing embeddings')

    model = opt.model  # type: str
    if model.lower() == 'd2v':
        out_name = "d2v(vecsize={}, winsize={}, mincount={}, {}, epochs={}).pkl".format(
            opt.vec_size, opt.win_size, opt.min_count, "dm" if opt.dm else "dbow", opt.epochs
        )
        embeddings = embedder.get_doc2vec(vectors_size=int(opt.vec_size),
                                          window=int(opt.win_size),
                                          min_count=int(opt.min_count),
                                          dm=int(opt.dm),
                                          epochs=int(opt.epochs))

    elif model.lower() == "nd2v":
        out_name = "nd2v(normalizer={}).pkl".format(opt.normalizer)
        embeddings = embedder.get_naive_doc2vec(normalizer=opt.normalizer)

    elif model.lower() == "onehot":
        out_name = "onehot(scorer={}).pkl".format(opt.scorer)
        embeddings = embedder.get_onehot(scorer=opt.scorer)

    elif model.lower() == "fasttext":
        out_name = "fasttext().pkl"
        embeddings = embedder._fast_text()  # not yet implemented

    else:
        print("unrecognized model, using naive doc2vec as fallback")
        out_name = "nd2v(normalizer={}).pkl".format(opt.normalizer)
        embeddings = embedder.get_naive_doc2vec(normalizer=opt.normalizer)

    if isinstance(embeddings, list):  # if the embeddings is in a list, stack them into a 2-D numpy array
        embeddings = np.stack(embeddings)

    print("embeddings computed")

    # dump the embedding matrix on disk
    try:
        os.makedirs(opt.output)
    except FileExistsError:
        print("Parent Dir Existent")
    finally:
        out_name = opt.column + "-" + out_name
        out_path = os.path.join(opt.output, out_name)
        with open(out_path, "wb") as f:
            print("storing embeddings in {}".format(out_path))
            pkl.dump(embeddings, f)
            print("embeddings stored")
