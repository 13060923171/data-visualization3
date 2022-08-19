# 模型介绍

XGBoost参数介绍：

n_estimators：想要建造的树的数量

learning_rate:我们的模型学习数据模式的速率，在每一轮之后，它缩小特征权重以达到最佳最优值

max_depth：确定每棵树在任何提升汇合中允许生长的深度

colsample_bytree：每棵树使用的特征百分比

gamma：指定进行拆分所需的最小损失减少



案例

```
param_grid = {
	"n_estimators":[100,150,200],
	"learning_rate":[0.01,0.05,0.1],
	"max_depth":[3,4,5,6,7],
	"colsample_bytree":[0.6,0.7,1],
	"gamma":[0.0,0.1,0.2]
}
booster_grid_search = GridSearchCV(booster,param_grid,cv=3,n_jobs=-1)
booster_grid_search.fit(x_train,y_train)
print('best param:{0}\n best score:{1}'.format(booster_grid_search.best_params_,booster_grid_search.best_score_))
```

