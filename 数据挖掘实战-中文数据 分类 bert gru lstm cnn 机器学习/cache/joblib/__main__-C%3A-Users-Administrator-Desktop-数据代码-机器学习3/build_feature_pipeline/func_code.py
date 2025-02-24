# first line: 20
@memory.cache
def build_feature_pipeline():
    return FeatureUnion([
        ('word_tfidf', TfidfVectorizer(
            ngram_range=(1, 2),  # 减少n-gram范围
            max_features=8000,  # 减少特征数量
            sublinear_tf=True,
            min_df=5  # 增加最小词频
        )),
        ('char_tfidf', TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 4),  # 减少字符n-gram范围
            max_features=5000  # 减少特征数量
        ))
    ])
