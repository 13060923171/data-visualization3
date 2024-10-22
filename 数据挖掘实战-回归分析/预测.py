import joblib
import pandas as pd

best_model_name = 'Gradient Boosting'

# 加载保存的对象
loaded_le = joblib.load('label_encoder.pkl')
loaded_scaler = joblib.load('scaler.pkl')
loaded_model = joblib.load(f'{best_model_name}_model.pkl')

def floor_process(x):
    if x == '低楼层':
        return 0
    if x == '中楼层':
        return 1
    if x == '高楼层':
        return 2

area_ = float(input('请输入面积大小：'))
orient_ = input('请输入朝向：')
city_ = input('请输入所在城市：')
district_ = input('请输入所在城市区域：')
floor_ = input('请输入所在楼层高度范围：')
floor1_ = floor_process(floor_)

def safe_label_transform(le, value):
    # 如果值不在已知类别中，则返回一个默认值
    if value not in le.classes_:
        return -1  # 或者其他处理方式
    else:
        return le.transform([value])[0]

# 对新数据进行编码和标准化
try:
    new_data = pd.DataFrame({
        'district': [safe_label_transform(loaded_le, district_)],
        'area': [area_],
        'orient': [safe_label_transform(loaded_le, orient_)],
        'floor': [floor1_],  # 楼层已经是数值，不需要 LabelEncoder
        'city': [safe_label_transform(loaded_le, city_)]
    })
except Exception as e:
    print(f"Error during transforming input: {e}")
    new_data = pd.DataFrame({
        'district': [-1],
        'area': [area_],
        'orient': [-1],
        'floor': [floor1_],
        'city': [-1]
    })

new_data_scaled = loaded_scaler.transform(new_data)

# 使用加载的模型进行预测
predicted_price = loaded_model.predict(new_data_scaled)
print(f"预测的房价是: {predicted_price[0]}")