import pandas as pd
import numpy as np

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")

from scipy.stats import pearsonr


def data_process():
    df = pd.read_excel('比特币数据.xlsx')
    df['日期'] = pd.to_datetime(df['日期']).dt.date

    df1 = pd.DataFrame()
    df1['Date'] = df['日期']
    df1['Close'] = df['收盘价']

    data = pd.read_excel('new_data.xlsx')
    # 按日期分组并采样
    data['发布时间'] = pd.to_datetime(data['发布时间']).dt.date
    # 按日期和情感类别统计次数
    result = (
        data.groupby(["发布时间", "情感类别"])  # 按日期和类别分组
          .size()                     # 统计每组数量
          .unstack(fill_value=0)      # 将类别转为列，缺失值填0
          .reset_index()              # 将日期从索引转为列
    )

    # 只保留 LABEL_0 和 LABEL_1（排除其他可能存在的类别）
    result = result[["发布时间", "LABEL_0", "LABEL_1"]]

    new_df = pd.merge(df1,result,left_on=['Date'],right_on=['发布时间'])
    new_df = new_df.drop(['发布时间'],axis=1)
    new_df.to_excel('情感数据.xlsx',index=False)

    return new_df


def main1():
    df = data_process()
    # 计算情感指标
    df["Sentiment_Ratio"] = df["LABEL_1"] / (df["LABEL_0"] + df["LABEL_1"])  # 正面情感占比
    df["Sentiment_Net"] = df["LABEL_1"] - df["LABEL_0"]  # 净情感数量
    # 优化2：分箱处理
    df["Sentiment_Net_Bin"] = pd.cut(
        df["Sentiment_Net"],
        bins=[-np.inf, -150, 0, 150, np.inf],
        labels=["Strong Negative", "Weak Negative", "Weak Positive", "Strong Positive"]
    )

    # 计算相关系数
    corr_ratio = df[["Close", "Sentiment_Ratio"]].corr(method="pearson").iloc[0, 1]
    corr_net = df[["Close", "Sentiment_Net"]].corr(method="pearson").iloc[0, 1]

    # 显著性检验（p值）
    _, p_ratio = pearsonr(df["Close"], df["Sentiment_Ratio"])
    _, p_net = pearsonr(df["Close"], df["Sentiment_Net"])

    print(f"Correlation (Sentiment Ratio vs Close): {corr_ratio:.3f}, p-value: {p_ratio:.3f}")
    print(f"Correlation (Sentiment Net vs Close): {corr_net:.3f}, p-value: {p_net:.3f}")

    # 可视化
    plt.figure(figsize=(12, 6))

    # 股价与情感占比的散点图
    plt.subplot(1, 2, 1)
    sns.regplot(x="Sentiment_Ratio", y="Close", data=df, scatter_kws={"alpha": 0.5})
    plt.title("Sentiment Ratio vs Close Price")

    # 股价与净情感的散点图
    plt.subplot(1, 2, 2)
    sns.regplot(x="Sentiment_Net", y="Close", data=df, scatter_kws={"alpha": 0.5})
    plt.title("Sentiment Net vs Close Price")

    plt.tight_layout()
    plt.savefig('Sentiment vs Close Price.png')

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 绘制股价（上方子图）
    ax1.plot(df["Date"], df["Close"], color="blue", label="Close Price", marker="o")
    ax1.set_ylabel("Close Price")
    ax1.legend()

    # 绘制情感占比（下方子图）
    ax2.plot(df["Date"], df["Sentiment_Ratio"] * 100, color="red", label="Sentiment Ratio (%)", linestyle="--")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sentiment Ratio (%)")
    ax2.legend()

    # 调整布局
    plt.xticks(rotation=45)
    plt.suptitle("Close Price and Sentiment Ratio Over Time (Subplots)")
    plt.tight_layout()
    plt.savefig('Close Price and Sentiment Ratio Over Time (Subplots).png')
    plt.show()


if __name__ == '__main__':
    main1()

