import numpy as np
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from collections import Counter
import time

class SelfWord2Vec:
    def __init__(self, file_path, mode = 'train', freq = 5):
        self.preprocess(file_path, freq)
        self.build_model()
        if mode == 'train':
            self.train()

    def preprocess(self, file_path, freq):
        with open(file_path) as f:
            text = f.read()

        # 对文本中的符号进行替换
        text = text.lower()
        text = text.replace('.', ' <PERIOD> ')
        text = text.replace(',', ' <COMMA> ')
        text = text.replace('"', ' <QUOTATION_MARK> ')
        text = text.replace(';', ' <SEMICOLON> ')
        text = text.replace('!', ' <EXCLAMATION_MARK> ')
        text = text.replace('?', ' <QUESTION_MARK> ')
        text = text.replace('(', ' <LEFT_PAREN> ')
        text = text.replace(')', ' <RIGHT_PAREN> ')
        text = text.replace('--', ' <HYPHENS> ')
        text = text.replace('?', ' <QUESTION_MARK> ')
        # text = text.replace('\n', ' <NEW_LINE> ')
        text = text.replace(':', ' <COLON> ')
        words = text.split()

        # 删除低频词，减少噪音影响
        word_counts = Counter(words)
        self.words = [word for word in words if word_counts[word] > freq]

        # 构建映射表
        self.vocab = set(words)
        self.vocab_to_int = {w: c for c, w in enumerate(self.vocab)}
        self.int_to_vocab = {c: w for c, w in enumerate(self.vocab)}

        print("total words: {}".format(len(self.words)))
        print("unique words: {}".format(len(set(self.words))))

        # 对原文本进行vocab到int的转换
        self.int_words = [self.vocab_to_int[w] for w in self.words]

        t = 1e-5 # t值
        threshold = 0.8 # 剔除概率阈值

        # 统计单词出现频次
        int_word_counts = Counter(self.int_words)
        total_count = len(self.int_words)
        # 计算单词频率
        word_freqs = {w: c/total_count for w, c in int_word_counts.items()}
        # 计算被删除的概率
        prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in int_word_counts}
        # 对单词进行采样
        self.words = [w for w in self.int_words if prob_drop[w] < threshold]
        print('after cut by freq, len: ' + str(len(self.words)))

    def build_model(self):
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            self.inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
            self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')

        self.vocab_size = len(self.int_to_vocab)
        self.embedding_size = 200 # 嵌入维度

        with self.train_graph.as_default():
            # 嵌入层权重矩阵
            self.embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1, 1))
            # 实现lookup
            self.embed = tf.nn.embedding_lookup(self.embedding, self.inputs)

        self.n_sampled = 100

        with self.train_graph.as_default():
            softmax_w = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.vocab_size))

            # 计算negative sampling下的损失
            self.loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, self.labels, self.embed, self.n_sampled, self.vocab_size)

            self.cost = tf.reduce_mean(self.loss)
            self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

        with self.train_graph.as_default():
            # 随机挑选一些单词
            valid_size = 16
            valid_window = 100
            # 从不同位置各选8个单词
            valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
            valid_examples = np.append(valid_examples,
                                       random.sample(range(1000,1000+valid_window), valid_size//2))

            # 验证单词集
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            # 计算每个词向量的模并进行单位化
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding), 1, keep_dims=True))
            normalized_embedding = self.embedding / norm
            # 查找验证单词的词向量
            valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
            # 计算余弦相似度
            self.valid_similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))

    def train(self):
        self.epochs = 10 # 迭代轮数
        self.batch_size = 1000 # batch大小
        self.window_size = 10 # 窗口大小
        with self.train_graph.as_default():
            saver = tf.train.Saver() # 文件存储

        with tf.Session(graph=self.train_graph) as sess:
            iteration = 1
            loss = 0
            sess.run(tf.global_variables_initializer())

            for e in range(1, self.epochs+1):
                batches = self.get_batches(self.words, self.batch_size, self.window_size)
                start = time.time()
                #
                for x, y in batches:

                    feed = {self.inputs: x,
                            self.labels: np.array(y)[:, None]}
                    train_loss, _ = sess.run([self.cost, self.optimizer], feed_dict=feed)

                    loss += train_loss

                    if iteration % 100 == 0:
                        end = time.time()
                        print("Epoch {}/{}".format(e, self.epochs),
                              "Iteration: {}".format(iteration),
                              "Avg. Training loss: {:.4f}".format(loss/100),
                              "{:.4f} sec/batch".format((end-start)/100))
                        loss = 0
                        start = time.time()

                    # 计算相似的词
                    if iteration % 1000 == 0:
                        # 计算similarity
                        sim = self.valid_similarity.eval()
                        for i in range(len(self.valid_examples)):
                            valid_word = self.int_to_vocab[self.valid_examples[i]]
                            top_k = 8 # 取最相似单词的前8个
                            nearest = (-sim[i, :]).argsort()[1:top_k+1]
                            log = 'Nearest to [%s]:' % valid_word
                            for k in range(top_k):
                                close_word = self.int_to_vocab[nearest[k]]
                                log = '%s %s,' % (log, close_word)
                            print(log)

                    iteration += 1

            save_path = saver.save(sess, "checkpoints/text8.ckpt")
            print('save model to ' + save_path)

    def get_targets(self, words, idx, window_size=5):
        '''
        获得input word的上下文单词列表

        参数
        ---
        words: 单词列表
        idx: input word的索引号
        window_size: 窗口大小
        '''
        target_window = np.random.randint(1, window_size+1)
        # 这里要考虑input word前面单词不够的情况
        start_point = idx - target_window if (idx - target_window) > 0 else 0
        end_point = idx + target_window
        # output words(即窗口中的上下文单词)
        targets = set(words[start_point: idx] + words[idx+1: end_point+1])
        return list(targets)

    def get_batches(self, words, batch_size, window_size=5):
        '''
        构造一个获取batch的生成器
        '''
        n_batches = len(words) // batch_size

        # 仅取full batches
        words = words[:n_batches*batch_size]

        for idx in range(0, len(words), batch_size):
            x, y = [], []
            batch = words[idx: idx+batch_size]
            for i in range(len(batch)):
                batch_x = batch[i]
                batch_y = self.get_targets(batch, i, window_size)
                # 由于一个input word会对应多个output word，因此需要长度统一
                x.extend([batch_x]*len(batch_y))
                y.extend(batch_y)
            yield x, y
