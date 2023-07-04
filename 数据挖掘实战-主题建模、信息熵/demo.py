from gensim import corpora
from gensim.models import LdaModel
import numpy as np

# 准备一些英文文本数据
texts = ["This is an example sentence about topic modeling.",
         "Topic modeling can be used for text analysis.",
         "LDA is a popular algorithm for topic modeling."]

# 文本预处理

# 文本特征提取
dictionary = corpora.Dictionary([text.split() for text in texts])
corpus = [dictionary.doc2bow(text.split()) for text in texts]

# LDA主题建模
lda_model = LdaModel(corpus, num_topics=6, id2word=dictionary)

# 计算主题的信息熵
topic_entropies = []
for topic in lda_model.show_topics():
    topic_words = [word for word, prob in lda_model.show_topic(topic[0])]
    word_probs = [prob for word, prob in lda_model.show_topic(topic[0])]
    entropy = np.sum(-np.array(word_probs) * np.log2(word_probs))
    topic_entropies.append((topic[0], topic_words, entropy))

# 打印主题及其对应的信息熵
for topic_entropy in topic_entropies:
    print("Topic {}: {}".format(topic_entropy[0], topic_entropy[1]))
    print("Entropy: {}".format(topic_entropy[2]))
    print()