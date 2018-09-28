import numpy as np
from model.hypertuned_models import classifiers
from sklearn.model_selection import cross_val_score, train_test_split
from embedding_loader import EmbeddingLoader

loader = EmbeddingLoader("pretrained")
emb = loader.get_d2v(corpus="concat", win_size=23, epochs=500)
labels = loader.get_label()

emb_train, emb_test, lab_train, lab_test = train_test_split(emb, labels, stratify=labels, test_size=0.25,
                                                            random_state=58)

if __name__ == '__main__':
    for clf in classifiers:
        scores = cross_val_score(clf, emb_train, lab_train, n_jobs=-1, cv=5)
        print(clf.__class__.__name__, ":", np.mean(scores), scores)
        clf.fit(emb_train, lab_train)
        print(clf.score(emb_test, lab_test))
