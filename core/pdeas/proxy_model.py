import csv
import glob
import os
import os.path as osp

import numpy as np
import tensorflow as tf

from core.utils.log_util import create_logger

logger = create_logger()


class Controller(tf.keras.Model):
    """
    Controller, used to predict the model given model and score.
    """

    def __init__(self, input_max, estimator_max, units=64, embedding_dim=16, random_state=None):
        """
        :param input_max: max index of the input
        :param estimator_max: max index of the estimator index
        :param units: #units of the LSTM
        :param embedding_dim: params of the Embedding Layer
        """
        super().__init__()
        self.input_max = input_max
        self.estimator_max = estimator_max
        self.units = units
        self.embedding_dim = embedding_dim
        self.random_state = random_state

        self.build_model()

    def build_model(self):
        """
        build proxy model(currently a LSTM neural network)
        :return:
        """
        self.input_embedding = tf.keras.layers.Embedding(self.input_max + 1, self.embedding_dim)
        self.estimator_embedding = tf.keras.layers.Embedding(self.estimator_max + 1, self.embedding_dim)

        if tf.test.is_gpu_available():
            self.rnn = tf.keras.layers.CuDNNLSTM(self.units, return_state=True)
        else:
            self.rnn = tf.keras.layers.LSTM(self.units, return_state=True)

        self.rnn_score = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, states=None, training=None, mask=None):
        input = inputs[:, 0::2]
        estimator = inputs[:, 1::2]

        if states is None:  # initialize the state vectors
            states = self.rnn.get_initial_state(inputs)
            states = [tf.to_float(state) for state in states]

        # map the sparse inputs and operators into dense embeddings
        embed_inputs = self.input_embedding(input)
        embed_ops = self.estimator_embedding(estimator)

        # concatenate the embeddings
        embed = tf.concat([embed_ops, embed_inputs], axis=-1)  # concatenate the embeddings

        # run over the LSTM
        out = self.rnn(embed, initial_state=states)
        out, h, c = out  # unpack the outputs and states

        # get the predicted validation accuracy
        score = self.rnn_score(out)

        return [score, [h, c]]


class ProxyModel:
    """
    cooperate with search space of the controller, main feature:
    1 initialize search space
    2 initialize controller
    3 train controller, generate new models
    4 finetune controller, update existing models of the search space
    """

    def __init__(self, sp, K=8, train_iterations=10, reg_param=0.0001, controller_cells=64, embedding_dim=16,
                 restore_controller=False, saved_path=None, strategy='best', random_state=None):
        """
        Controller manager.
        :param sp: Search Space
        :param K: best K models
        :param train_iterations: max train iterations
        :param reg_param: regularization params
        :param controller_cells: #cell of lstm units
        :param embedding_dim: dimension of the embedding layer
        :param restore_controller: whether restore the controller
        :param saved_path: results saved path
        :param strategy: Used to control how to grow to a new level, default best. Possible values: best, random,
        best_vs_random, random_vs_best, best_vs_worst, if has some random, the random rate is 0.8
        :param random_state: random state, currently not used
        """
        self.sp = sp  # type: CPSpace

        self.b_ = 1
        self.B = sp.B
        self.K = K
        self.embedding_dim = embedding_dim

        self.train_iterations = train_iterations
        self.controller_cells = controller_cells
        self.reg_strength = reg_param
        self.restore_controller = restore_controller
        self.saved_path = saved_path
        self.strategy = strategy
        self.random_state = random_state

        self.children_history = None
        self.score_history = None

        self.build_proxy_model()

    def get_models(self, top_k=None):
        """
        get top-k child
        :param top_k: beam search K
        :return: top-k models
        """
        models = self.sp.history
        if top_k is not None:
            models = self.sp.history[:top_k]
        return models

    def build_proxy_model(self):
        """
        set RNN controller by the input, can also be used to save and restore controller
        """
        if tf.test.is_gpu_available():
            device = '/gpu:0'
        else:
            device = '/cpu:0'
        # device='/cpu:0'
        self.device = device

        self.global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(0.001, self.global_step,
                                                   500, 0.98, staircase=True)
        # controller
        with tf.device(device):
            self.controller = Controller(self.sp.inputs_embedding_max,
                                         self.sp.operator_embedding_max,
                                         self.controller_cells,
                                         self.embedding_dim, )

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # save model
        self.saver = tf.train.Checkpoint(controller=self.controller,
                                         optimizer=self.optimizer,
                                         global_step=self.global_step)
        # restore model
        if self.restore_controller:
            RESULTS_SAVED_PATH = glob.glob('results*')[0]
            path = tf.train.latest_checkpoint(osp.join(RESULTS_SAVED_PATH, 'weights'))

            if path is not None and tf.train.checkpoint_exists(path):
                logger.info("load controller save point")
                self.saver.restore(path)

    def loss(self, real_acc, rnn_scores):
        """
        compute loss function to training controller
        method:
        MSE of the real validation acc and predict acc, using L2 regularization
        """
        # RNN predicted acc
        rnn_score_loss = tf.losses.mean_squared_error(real_acc, rnn_scores)

        # regularization of model
        params = self.controller.trainable_variables
        reg_loss = tf.reduce_sum([tf.nn.l2_loss(x) for x in params])

        total_loss = rnn_score_loss + self.reg_strength * reg_loss

        return total_loss

    def finetune_proxy_model(self, rewards):
        """
        finetune controller by the model and its performance
        :param rewards: performance of the model
        """
        children = np.array(self.sp.history, dtype=np.object)  # take all the children
        rewards = np.array(rewards, dtype=np.float32)
        loss = 0

        if self.children_history is None:
            self.children_history = [children]
            self.score_history = [rewards]
            batchsize = rewards.shape[0]
        else:
            self.children_history.append(children)
            self.score_history.append(rewards)
            batchsize = sum([data.shape[0] for data in self.score_history])

        train_size = batchsize * self.train_iterations
        for current_epoch in range(self.train_iterations):
            for dataset_id in range(self.b_):
                children = self.children_history[dataset_id]
                scores = self.score_history[dataset_id]
                ids = np.array(list(range(len(scores))))
                np.random.shuffle(ids)
                for id, (child, score) in enumerate(zip(children[ids], scores[ids])):
                    child = child.tolist()
                    state_list = self.sp.entity_encode_child(child)
                    state_list = np.concatenate(state_list, axis=-1).astype('int32')

                    with tf.device(self.device):
                        state_list = tf.convert_to_tensor(state_list)

                        with tf.GradientTape() as tape:
                            rnn_scores, states = self.controller(state_list, states=None)
                            acc_scores = score.reshape((1, 1))

                            total_loss = self.loss(acc_scores, rnn_scores)

                        grads = tape.gradient(total_loss, self.controller.trainable_variables)
                        grad_vars = zip(grads, self.controller.trainable_variables)

                        self.optimizer.apply_gradients(grad_vars, self.global_step)

                    loss += total_loss.numpy().sum()
        return np.mean(loss)

    def update_search_space(self):
        """
        Updates the children from the intermediate products for the next generation
        of larger number of cells in each cell
        """
        if self.b_ + 1 <= self.B:
            self.b_ += 1
            models_scores = []
            # iterate through all the intermediate children
            logger.info(f"End cell number = {self.b_ - 1}")
            logger.info("Begin to predict the model performance.")
            for i, intermediate_child in enumerate(self.sp.generate_new_models(self.b_)):
                state_list = self.sp.entity_encode_child(intermediate_child)
                state_list = np.concatenate(state_list, axis=-1).astype('int32')
                state_list = tf.convert_to_tensor(state_list)
                # score the child
                score, _ = self.controller(state_list, states=None)
                score = score[0, 0].numpy()
                # preserve the child and its score
                models_scores.append([intermediate_child, score])
                scores_dir = osp.join(self.saved_path, 'scores')
                if not osp.exists(scores_dir):
                    os.makedirs(scores_dir)
                with open(osp.join(scores_dir, f'score_{self.b_}.csv'), mode='a+', newline='') as f:
                    writer = csv.writer(f)
                    data = [score]
                    data.extend(intermediate_child)
                    writer.writerow(data)
                if (i + 1) % 500 == 0:
                    logger.info(f"Predict {i + 1} models, current model score = {score:.2f}")
            # account for case where there are fewer children than K
            if self.K is not None:
                children_count = min(self.K, len(models_scores))
            else:
                children_count = len(models_scores)
            # take only the K highest scoring children for next iteration
            children = []
            if self.strategy == 'best':
                # sort the children according to their score
                models_scores = sorted(models_scores, key=lambda x: x[1], reverse=True)
                for i in range(children_count):
                    children.append(models_scores[i][0])
            elif self.strategy == 'random':
                choices = np.random.choice(len(models_scores), children_count, replace=False)
                for choice in choices:
                    children.append(models_scores[choice][0])
            elif self.strategy == 'best_vs_random':
                # sort the children according to their score
                models_scores = sorted(models_scores, key=lambda x: x[1], reverse=True)
                rate = 0.8
                best_count = int(rate * children_count)
                random_count = children_count - best_count
                random_indexes = np.random.choice(range(best_count, len(models_scores)), random_count, replace=False)
                for i in range(best_count):
                    children.append(models_scores[i][0])
                for random_index in random_indexes:
                    children.append(models_scores[random_index][0])
            elif self.strategy == 'random_vs_best':
                # sort the children according to their score
                models_scores = sorted(models_scores, key=lambda x: x[1], reverse=True)
                # take only the K highest scoring children for next iteration
                rate = 0.2
                best_count = int(rate * children_count)
                random_count = children_count - best_count
                random_indexes = np.random.choice(range(best_count, len(models_scores)), random_count, replace=False)
                for i in range(best_count):
                    children.append(models_scores[i][0])
                for random_index in random_indexes:
                    children.append(models_scores[random_index][0])
            elif self.strategy == 'best_vs_worst':
                # sort the children according to their score
                models_scores = sorted(models_scores, key=lambda x: x[1], reverse=True)
                rate = 0.8
                before = int(rate * children_count)
                for i in range(before):
                    children.append(models_scores[i][0])
                for i in range(before, children_count):
                    children.append(models_scores[len(models_scores) - 1 - i][0])
            else:
                raise ValueError("Not supported strategy, possible like 'best','random','best_vs_random',"
                                 "'random_vs_best','best_vs_worst.")
            # save these children for next round
            self.sp.update_children(children)
            logger.info("Predict model performance finished.\n")
