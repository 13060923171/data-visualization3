import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv('prec_jieba.csv')
df2 = pd.read_csv('prec_hmm.csv')
df3 = pd.read_csv('prec_crf.csv')
jr_prec, jr_rec, jr_f_score = df1['Precision'][0],df1['Recall'][0],df1['FScore'][0]
hr_prec, hr_rec, hr_f_score = df2['Precision'][0],df2['Recall'][0],df2['FScore'][0]
cr_prec, cr_rec, cr_f_score = df3['Precision'][0],df3['Recall'][0],df3['FScore'][0]
# 对比并可视化
labels = ['Precision', 'Recall', 'F-Score']
jieba_metrics = [jr_prec, jr_rec, jr_f_score]
hmm_metrics = [hr_prec, hr_rec, hr_f_score]
crf_metrics = [cr_prec, cr_rec, cr_f_score]

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots()
rects1 = ax.barh(x - width/2, jieba_metrics, width, label='Jieba')
rects2 = ax.barh(x + width/2, hmm_metrics, width, label='HMM')
rects3 = ax.barh(x + 3*width/2, crf_metrics, width, label='CRF')

ax.set_xlabel('Metrics')
ax.set_title('Segmentation Metrics by Algorithm')
ax.set_yticks(x)
ax.set_yticklabels(labels)
ax.legend()

fig.tight_layout()
plt.savefig('Metrics.png')

