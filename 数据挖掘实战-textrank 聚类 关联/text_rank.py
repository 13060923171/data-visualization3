from IPython.display import Image
import stylecloud
import pandas as pd


def wordcould(x):
    df = pd.read_csv('聚类结果.csv')
    new_df = df[df['聚类结果'] == x]
    word = []
    for t in new_df['text_rank']:
        for j in str(t).split(" "):
            word.append(j)
    word_str = ' '.join(word)
    stylecloud.gen_stylecloud(text=word_str, max_words=100,
                              collocations=False,
                              background_color='#B3B6B7',
                              font_path='simhei.ttf',
                              icon_name='fas fa-star',
                              size=500,
                              palette='matplotlib.Inferno_9',
                              output_name='./data/聚类{}_text_rank_词云图.png'.format(x))
    Image(filename='./data/聚类{}_text_rank_词云图.png'.format(x))


if __name__ == '__main__':
    for i in range(0,6):
        wordcould(i)