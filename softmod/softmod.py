from six.moves import xrange
from collections import Counter
from itertools import chain
import numpy as np
import tensorflow as tf


class SoftMod(object):

    def __init__(self, n_com, edge_list, weights=None, learning_rate=0.1,
                 lambda_phi=0.001, threads=8):
        self.n_com = n_com
        elist = edge_list
        self.n_edge = ne = len(elist)
        self.edge_list = elist
        self.n_node = nn = max(chain.from_iterable(elist)) + 1
        if weights is None:
            weights = np.ones(ne)
        self.weights = weights.astype(np.float32)
        self.sum_weight = weights.sum(dtype=np.float32)
        deg = {n: 0 for n in xrange(nn)}
        for n1, n2 in elist:
            deg[n1] += weights[n1]
            deg[n2] += weights[n2]
        self.degrees = [v for _, v in sorted(deg.items())]
        self.learning_rate = learning_rate
        self.lambda_phi = lambda_phi
        self.threads = threads
        self._setup_graph()

    def optimize(self, max_iter=100, logdir=None, stop_threshold=0.001):
        sess = self.sess
        if logdir:
            writer = tf.train.SummaryWriter(logdir, sess.graph_def)
        sess.run(self.init_op)
        pre_loss = 1.0
        for step in xrange(max_iter):
            loss, sm, _ = sess.run([self.loss, self.sm, self.opt_op])
            if logdir:
                writer.add_summary(sm, step)
            if np.abs(loss - pre_loss) < stop_threshold:
                break
            pre_loss = loss
        mod = sess.run(self.mod)
        theta = sess.run(self.Theta)
        print("Modularity:", mod)
        print("Loss:", loss)
        return theta, mod

    def get_soft_community(self):
        Theta = self.sess.run(self.Theta)
        return Theta

    def get_hard_communirty(self, Theta=None):
        if Theta is None:
            Theta = self.sess.run(self.Theta)
        hard_com = np.argmax(Theta, axis=1)
        return hard_com

    def calculate_mod_for_theta(self, Theta=None):
        sess = self.sess
        if Theta:
            mod = sess.run(self.mod_given,
                           feed_dict={self.Theta_given: Theta})
        else:
            mod = sess.run(self.mod)
        return mod

    def _setup_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            elist = self.edge_list + [(j, i) for i, j in self.edge_list]
            weights = np.append(self.weights, self.weights)
            sum_weight = self.sum_weight
            nn = self.n_node
            nc = self.n_com
            deg = self.degrees
            lr = self.learning_rate
            lambda_phi = self.lambda_phi

            with tf.name_scope("adj_mat"):
                self.X = X = tf.sparse_to_dense(sparse_values=weights,
                                       sparse_indices=elist,
                                       output_shape=[nn, nn])

            K = tf.constant(deg, shape=[nn,1], name="deg_vec")
            self.K_2 = K_2 = tf.matmul(K, K, transpose_b=True)
            sum_weight = tf.constant(sum_weight, name="sum_weight")

            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
            self.Phi = Phi = tf.get_variable(name="Phi", shape=[nn, nc],
                                             initializer=initializer)

            Phi_2 = tf.pow(Phi, 2)
            #Phi_2 = tf.log(1+tf.exp(tf.pow(Phi, 2)))
            with tf.name_scope("Theta"):
                self.Theta = Theta = self.normalize_along_row(Phi_2)
            self.Theta_2 = Theta_2 = tf.matmul(Theta, Theta, transpose_b=True)

            mod = tf.reduce_sum((X - (K_2 / (2*sum_weight)))*Theta_2) / \
                                (2*sum_weight)
            self.mod = mod
            self.phi_norm = phi_norm = tf.reduce_sum(Phi_2) / nn
            self.loss = loss = -mod + lambda_phi*phi_norm
            self.opt_op = tf.train.AdamOptimizer(lr).minimize(loss)

            self.Theta_given = Theta_g = tf.placeholder("float", shape=[nn, nc],
                                                        name="Theta_given")
            Theta_g2 = tf.matmul(Theta_g, Theta_g, transpose_b=True)
            mod_given = tf.reduce_sum((X - K_2 / (2*sum_weight))*Theta_g2) / \
                                      (2*sum_weight)
            self.mod_given = mod_given

            tf.histogram_summary("Phi", Phi)
            tf.histogram_summary("Theta", Theta)
            tf.scalar_summary("mod", mod)
            tf.scalar_summary("regularlization", phi_norm)
            tf.scalar_summary("loss", loss)
            self.sm = tf.merge_all_summaries()

            self.init_op = tf.initialize_all_variables()

            config = tf.ConfigProto(inter_op_parallelism_threads=self.threads,
                                    intra_op_parallelism_threads=self.threads)
            self.sess = tf.Session(config=config)

    @staticmethod
    def normalize_along_row(input_):
        rowsum = tf.expand_dims(tf.reduce_sum(input_, 1), 1)
        return input_ / rowsum
