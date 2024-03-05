import torch
from transformers import BertTokenizer
from transformers import BertModel
from torch import nn
import pandas as pd
from tqdm import tqdm


#定义bert分类器模型
class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        #加载bert模型
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        #定义dropout
        self.dropout = nn.Dropout(dropout)
        #定义线性层
        self.linear = nn.Linear(768, 5)
        #定义relu激活函数
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        #进行bert模型的正向传播计算
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        #进行dropout层的正向传播计算
        dropout_output = self.dropout(pooled_output)
        #进行线性层传播计算
        linear_output = self.linear(dropout_output)
        #进行relu激活函数的正向传播计算
        final_layer = self.relu(linear_output)
        return final_layer

# 加载BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# # 加载保存的模型
model = BertClassifier()
model.load_state_dict(torch.load("full_model.pkl"))
model.eval()


# 定义一个函数，用于对输入文本进行分类预测
def predict_class(text):
    # 对文本进行分词和转换
    inputs = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")

    # 获取input_ids和attention_mask
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # 使用模型进行预测
    with torch.no_grad():
        output = model(input_ids, attention_mask)

    # 获取预测结果
    _, predicted_class = torch.max(output, dim=1)

    return predicted_class.item()



# # 要进行分类预测的文本数据
# text_data = [
#     "这是一个测试句子",
#     "这个产品真是太棒了",
#     "我不喜欢这个电影",
#     "包装很好 商家态度好 不卖临期药 比药店便宜",
#     "包装很好 商家态度好 不卖临期药 配送员态度好 物流很快",
#     "货送到很快"
#
# ]
# # 对每个文本数据进行分类预测，并输出结果
# for text in text_data:
#     predicted_class = predict_class(text)
#     print(f"文本: {text}，预测分类: {predicted_class}")

df = pd.read_excel('test.xlsx')
text_data = list(df['评价内容'])
# 对每个文本数据进行分类预测，并输出结果
predicted_list = []
for text in tqdm(text_data):
    predicted_class = predict_class(text)
    predicted_list.append(predicted_class)

df['评论类型'] = predicted_list
df.to_excel('new_data.xlsx',index=False)