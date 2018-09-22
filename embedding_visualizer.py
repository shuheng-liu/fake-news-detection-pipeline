import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def visualize_embeddings(embedding_values, label_values, embedding_name="doc_vec", points_to_show=None):
    """
    function for visualize embeddings with tensorboard.
    MUST run in command line "tensorboard --logdir visual/" and visit localhost:6006 to see the visualization
    :param embedding_values: np.ndarray, in the shape of [n_docs, n_dims]
    :param label_values: np.ndarray, in the shape of [n_docs]
    :param embedding_name: name for the embeddings, spaces are auto-deleted
    :param points_to_show: maximum number of points to show
    :return: None
    """
    TENSORBOARD_ROOT = 'visual'  # home directory for running tensorboard server
    embedding_name.replace(" ", "")  # the `embedding_name` is later used as a tf.scope_name; it mustn't contain spaces
    METADATA_PATH = os.path.join(TENSORBOARD_ROOT, 'metadata.tsv')  # place to save metadata

    assert isinstance(embedding_values, np.ndarray), "{} is not a npndarray".format(embedding_values)
    assert isinstance(label_values, np.ndarray), "{} is not a npndarray".format(label_values)

    if points_to_show is not None:
        points_to_show = min(points_to_show, len(embedding_values), len(label_values))
        embedding_values = embedding_values[:points_to_show]
        label_values = label_values[:points_to_show]

    embedding_var = tf.Variable(embedding_values, name=embedding_name)  # instantiate a tensor to hold embedding values
    summary_writer = tf.summary.FileWriter(TENSORBOARD_ROOT)  # instantiate a writer to write summaries
    config = projector.ProjectorConfig()  # `config` maintains arguments for write embeddings and save them on disk
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # Specify where you find the metadata
    embedding.metadata_path = "metadata.tsv"  # XXX this step might introduce error, see the printed message below

    print("WARNING: potential error due to tensorboard version conflicts")
    print("currently setting metadata_path to {}. Due to tensorboard version reasons, if prompted 'metadata not found' "
          "when visiting tensorboard server page, please manually edit metadata_path in projector_config.pbtxt to {} "
          "or the absolute path for `metadata.tsv` and restart tensorboard".format(embedding.metadata_path,
                                                                                   METADATA_PATH))
    print("If your tensorboard version is 1.7.0, you probably should not worry about this")

    # call the following method to visualize embeddings
    projector.visualize_embeddings(summary_writer, config)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # initialize the `embedding_var`
        saver = tf.train.Saver()  # instantiate a saver for this session
        saver.save(sess, os.path.join(TENSORBOARD_ROOT, "model.ckpt"), 1)

    # write metadata (i.e., labels) for emebddings; this is how tensorboard knows labels of different embeddings
    with open(METADATA_PATH, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(label_values):
            f.write("%d\t%d\n" % (index, label))

    print("Embeddings are available now. Please start your tensorboard server with commandline "
          "`tensorboard --logdir visual` and visit http://localhost:6006 to see the visualization")


if __name__ == '__main__':
    from embedding_loader import EmbeddingLoader

    loader = EmbeddingLoader("embeddings")
    visualize_embeddings(embedding_values=loader.get_d2v(corpus="text", win_size=23, dm=False, epochs=500),
                         embedding_name="text",
                         label_values=loader.get_label(),
                         points_to_show=3000)
    visualize_embeddings(embedding_values=loader.get_d2v(corpus="title", win_size=23, dm=False, epochs=500),
                         embedding_name="title",
                         label_values=loader.get_label(),
                         points_to_show=3000)
