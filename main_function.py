import numpy as np
import re
import jieba # 结巴分词
# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
import warnings

from tensorflow.python.keras.saving import load_model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.optimizers import Adam
# 进行训练和测试样本的分割
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# 忽略警告
warnings.filterwarnings("ignore")
# 测试加载预训练模型
cn_model = KeyedVectors.load_word2vec_format('sgns.zhihu.bigram',binary=False, unicode_errors="ignore")
# print(cn_model.similarity('橘子', '橙子'))
# print(cn_model.most_similar(positive=['大学'], topn=10))

# 我们数据集有4000条评论
# 只使用前50000个中文词做测试----目前作为测试，生产过程可以全部使用
num_words = 50000
# 词向量数----该值是基于sgns.zhihu.bigram中的维度来设定
embedding_dim = 300
# 输入的最大维度值----该值是代表所有被处理的评论的词数。
max_tokens = 236
# 建立一个权重的存储点
path_checkpoint = 'checkpoint.h5'
# 建立模型
model = Sequential()
def execute():
    # 现在我们将所有的评价内容放置到一个list里
    train_texts_orig = []
    # 文本所对应的labels, 也就是标记
    train_target = []
    with open("positive_samples.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            dic = eval(line)
            train_texts_orig.append(dic["text"])
            train_target.append(dic["label"])

    with open("negative_samples.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            dic = eval(line)
            train_texts_orig.append(dic["text"])
            train_target.append(dic["label"])

    # 进行分词和tokenize
    # train_tokens是一个长长的list，其中含有4000个小list，对应每一条评价
    train_tokens = []
    for text in train_texts_orig:
        # 去掉标点
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
        # 结巴分词
        cut = jieba.cut(text)
        # 结巴分词的输出结果为一个生成器
        # 把生成器转换为list
        cut_list = [i for i in cut]
        for i, word in enumerate(cut_list):
            try:
                # 将词转换为索引index
                cut_list[i] = cn_model.vocab[word].index
            except KeyError:
                # 如果词不在字典中，则输出0
                cut_list[i] = 0
        train_tokens.append(cut_list)


    # 初始化embedding_matrix，之后在keras上进行应用
    embedding_matrix = np.zeros((num_words, embedding_dim))
    # embedding_matrix为一个 [num_words，embedding_dim] 的矩阵
    # 维度为 50000 * 300
    for i in range(num_words):
        embedding_matrix[i, :] = cn_model[cn_model.index2word[i]]
    embedding_matrix = embedding_matrix.astype('float32')

    # 进行padding和truncating， 输入的train_tokens是一个list
    # 由于我们设定了最大的文字量是max_tokens 所以要用pad_sequences 进行一次削减
    # 返回的train_pad是一个numpy array
    train_pad = pad_sequences(train_tokens, maxlen=max_tokens,
                              padding='pre', truncating='pre')
    # 超出五万个词向量的词用0代替
    train_pad[train_pad >= num_words] = 0

    # 准备target向量，前2000样本为1，后2000为0
    train_target = np.array(train_target)


    X_train = train_pad
    y_train = train_target
    # 模型第一层为embedding
    model.add(Embedding(num_words,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_tokens,
                        trainable=False))
    model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
    model.add(LSTM(units=16, return_sequences=False))
    # 输出1个特征  指定激活函数 sigmoid  如果使用二元分类需要使用sigmoid函数
    model.add(Dense(1, "sigmoid"))
    # 我们使用adam以0.001的learning rate进行优化
    optimizer = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])


    # 回调函数
    # 动态储存我们的数据模型 save_weights_only需要改成false 不然只保存权重
    checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss',
                                 verbose=1, save_weights_only=False,
                                 save_best_only=True)
    # 定义early stoping如果3个epoch内validation loss没有改善则停止训练
    earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    # 自动降低learning rate
    lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                     factor=0.1, min_lr=1e-8, patience=0,
                                     verbose=1)
    # 定义callback函数
    callbacks = [
        earlystopping,
        checkpoint,
        lr_reduction
    ]

    # 开始训练并导入回调函数
    # 执行20次
    model.fit(X_train, y_train,
              epochs=20,
              validation_split=0.1,
              batch_size=128,
              callbacks=callbacks)

# 用来将tokens转换为文本
def reverse_tokens(tokens):
    text = ''
    for i in tokens:
        if i != 0:
            text = text + cn_model.index2word[i]
        else:
            text = text + ' '
    return text
# 预测
# 封装成方法
def predict_sentiment(text):
    print(text)
    # 去标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
    # 分词
    cut = jieba.cut(text)
    cut_list = [ i for i in cut ]
    # tokenize
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.vocab[word].index
            if cut_list[i] >= num_words:
                cut_list[i] = 0
        except KeyError:
            cut_list[i] = 0
    # padding
    tokens_pad = pad_sequences([cut_list], maxlen=max_tokens,
                           padding='pre', truncating='pre')
    # 预测
    result = model.predict(tokens_pad)
    # 打印输出我们的分类类别
    #
    # print("最终分类为"+model.predict_classes(tokens_pad))
    coef = result[0][0]
    if coef >= 0.5:
        print('是一例正面评价','output=%.2f'%coef)
    else:
        print('是一例负面评价','output=%.2f'%coef)

if __name__ == '__main__':
    # 测试模型
    execute()
    # model = load_model(path_checkpoint)
    test_list = [
        '酒店设施不是新的，服务态度很不好',
        '酒店卫生条件非常不好',
        '床铺非常舒适',
        "服务很到位，但小姐姐不漂亮",
        '房间很凉，不给开暖气',
        '房间很凉爽，空调冷气很足',
        '酒店环境不好，住宿体验很不好',
        '房间隔音不到位',
        '晚上回来发现没有打扫卫生',
        "真垃圾",
        "非常好",
        '因为过节所以要我临时加钱，比团购的价格贵'
    ]
    for text in test_list:
        predict_sentiment(text)