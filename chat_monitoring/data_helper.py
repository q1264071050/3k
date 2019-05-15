from gensim.models.word2vec import Word2Vec
import jieba
import re
import numpy as np
import os
import multiprocessing
import pickle
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定一个GPU


def clean_str(line):
    """
    该函数的作用是去掉一个字符串中的所有非中文字符
    :param line:
    :return: 返回处理后的字符串
    """
    line = re.sub(r"voice.*?content=(.*?)>",'',line)#语音
    #line = re.sub(r"[^\u4e00-\u9fff]", "", line)  # 去掉所有的数字
    line=re.sub(r"\\<e>\\d*?\\<\/e>",'',line)#去掉表情
    line = line.replace(" ", "")
    if line != "":
        line = list(jieba.cut(line))
        return line


def load_data_and_labels(positive_data_file, negative_data_file):
    positive_examples = read_and_clean_zh_file(positive_data_file)
    negative_examples = read_and_clean_zh_file(negative_data_file)
    x_text = positive_examples + negative_examples
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return x_text, y

def load_test_data(data_file):
    test_data = read_and_clean_zh_file(data_file)
    return test_data
def read_and_clean_zh_file(input_file, output_cleaned_file=None):
    lines = list(open(input_file, "rb").readlines())
    result = []
    for line in lines:
        tmp = clean_str(line.decode('utf-8'))
        if tmp is not None:
            result.append(tmp)
    lines = result
    if output_cleaned_file is not None:
        with open(output_cleaned_file, 'w') as f:
            for line in lines:
                f.write((line + '\n').encode('utf-8'))
    return lines


def padding_sentence(sentences, padding_token='UNK', padding_sentence_length=None):
    """
    该函数的作用是 按最大长度Padding样本
    :param sentences: [['今天','天气','晴朗'],['你','真','好']]
    :param padding_token: padding 的内容，默认为'UNK'
    :param padding_sentence_length: 以5为例
    :return: [['今天','天气','晴朗','UNK','UNK'],['你','真','好','UNK'，'UNK']]
    """
    if padding_sentence_length is not None:
        max_padding_length = padding_sentence_length
    else:
        max_padding_length = max([len(sentence) for sentence in sentences])

    for i, sentence in enumerate(sentences):
        if len(sentence) < max_padding_length:
            sentence.extend([padding_token] * (max_padding_length - len(sentence)))
        else:
            sentences[i] = sentence[:max_padding_length]
    return sentences, max_padding_length


def word2vector(sentences, embedding_size=50, min_count=3, window=5,
                embedding_file='./embedding1.model'):
    print('-------word2vector------------')
    train_model = Word2Vec(sentences=sentences, size=embedding_size,
                           min_count=min_count, window=window)
    train_model.save(embedding_file)
    return train_model


def embedding_sentences(sentences, embedding_size=64, window=5, min_count=3, file_to_load=None, file_to_save=None):
    if file_to_load is not None:
        w2vModel = Word2Vec.load(file_to_load)
    else:
        w2vModel = Word2Vec(sentences, size=embedding_size, window=window, min_count=min_count, workers=multiprocessing.cpu_count())
        if file_to_save is not None:
            w2vModel.save(file_to_save)
    all_vectors = []
    embeddingDim = w2vModel.vector_size  # embeddingDim=128
    embeddingUnknown = [0 for i in range(embeddingDim)]
    # 一封邮件的向量为每个字向量的拼接
    for sentence in sentences:
        this_vector = []
        for word in sentence:
            if word in w2vModel.wv.vocab:
                this_vector.append(w2vModel[word])
            else:
                this_vector.append(embeddingUnknown)
        all_vectors.append(this_vector)
    return all_vectors

def generate_word2vec_files(input_file, output_model_file, output_vector_file, size = 50, window = 5, min_count = 0):
    start_time = time.time()

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model = Word2Vec(LineSentence(input_file), size = size, window = window, min_count = min_count, workers = multiprocessing.cpu_count())
    model.save(output_model_file)
    model.wv.save_word2vec_format(output_vector_file, binary=False)

    end_time = time.time()
    print("used time : %d s" % (end_time - start_time))

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    '''
    Generate a batch iterator for a dataset
    '''
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
        # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_idx : end_idx]

def saveDict(input_dict, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(input_dict, f)
def loadDict(dict_file):
    output_dict = None
    with open(dict_file, 'rb') as f:
        output_dict = pickle.load(f)
    return output_dict