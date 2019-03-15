#!/usr/bin/env python
# coding: utf-8

# # 新一轮空值填充和模型训练
# ## 1 数据填充

# In[1]:


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')   # 忽略matplot的警告
warnings.simplefilter('ignore')     # 忽略sklearn的警告

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

train = pd.read_csv("./data/data.csv")
# 删除噪声
train.drop(train[(train["GrLivArea"]>4000) & (train["SalePrice"]<300000)].index, inplace=True)
train.drop(train[(train["OverallQual"]<5) & (train["SalePrice"]>200000)].index, inplace=True)

print(train.shape)



# 按Neighborhood的种类分组，各组的中位数和均值如下：
neighborhood_group=train.groupby("Neighborhood")
lot_medians=neighborhood_group["LotFrontage"].median()
lot_mean=neighborhood_group["LotFrontage"].mean()

train["LotAreaCut"] = pd.qcut(train.LotArea,10)
#train['LotFrontage'] = train.groupby(['LotAreaCut', 'Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
train['LotFrontage'] = train.groupby(['LotAreaCut', 'Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.mean()))

# 查看LotFrontage是空值的行的Neighborhood的值
train[train["LotFrontage"].isnull()]["Neighborhood"]


# 相关性比较大，可添加一列SqrtLotArea
train['SqrtLotArea']=np.sqrt(train["LotArea"])
filter_LotFrontage = train['LotFrontage'].isnull()
# train.LotFrontage[filter_LotFrontage] = 0.6*train.SqrtLotArea[filter_LotFrontage]

# ### 1.2 MasVnrtype与MasVnrArea的填充
# - MasVnrtype：Masonry veneer type，砖石镶板种类
#        BrkCmn：Brick Common
#        BrkFace：Brick Face
#        CBlock：Cinder Block
#        None：None
#        Stone：Stone
# - MasVnrArea：Masonry veneer area in square feet，砖石镶板面积

# 可见相关性并不是很大，可以填充众数
filter_MasVnrArea = train['MasVnrArea'].isnull()   # 好像全部都是不是空啊
train.MasVnrArea[filter_MasVnrArea] = 0.0

filter_MasVnrType = train["MasVnrType"].isnull()
train.MasVnrType[filter_MasVnrType] = 'None'

# ### 1.3 Electrical的填充
# - Electrical：Electrical system，电气系统。
#     - SBrkr：Standard Circuit Breakers & Romex
#     - FuseA：Fuse Box over 60 AMP and all Romex wiring (Average)	
#     - FuseF：60 AMP Fuse Box and mostly Romex wiring (Fair)
#     - FuseP：60 AMP Fuse Box and mostly knob & tube wiring (poor)
#     - Mix：Mixed

# 直接填充众数
filter_Electrical = train["Electrical"].isnull()
train["Electrical"][filter_Electrical] = "SBrkr"

# ### 1.4 Alley(小巷子)的填充
# - Alley: Type of alley access，胡同通道的类型。
#     - Grvl：Gravel
#     - Pave：Paved
#     - NA ：No alley access 没有小巷
# - EDA后半段考虑删掉
# - 80%以上为空

# 填充None
train["Alley"] = train["Alley"].fillna("None")

# ### 1.5 BaseMent群填充
# - TotalBsmtSF是一个完整的关于BaseMent的列，可以拿出来与SalePrice进行相关性分析
# 
# - BsmtQual: Evaluates the height of the basement，地下室的高度。
#     - Ex: Excellent (100+ inches)
#     - Gd: Good (90-99 inches)
#     - TA: Typical (80-89 inches)
#     - Fa: Fair (70-79 inches)
#     - Po: Poor (<70 inches
#     - NA: No Basement
#        
# - BsmtCond: Evaluates the general condition of the basement，地下室的一般状况。
#     - Ex：Excellent
#     - Gd：Good
#     - TA：Typical - slight dampness allowed
#     - Fa：Fair - dampness or some cracking or settling
#     - Po：Poor - Severe cracking, settling, or wetness
#     - NA：No Basement
# 
# - BsmtExposure: Refers to walkout or garden level walls,罢工或花园级地下室的墙壁,裸露的地下室部分。
#     - Gd：Good Exposure
#     - Av：Average Exposure (split levels or foyers typically score average or above)	
#     - Mn：Mimimum Exposure
#     - No：No Exposure
#     - NA：No Basement
# 
# - BsmtFinType1: Rating of basement finished area，地下室成品面积质量。
#     - GLQ：Good Living Quarters
#     - ALQ：Average Living Quarters
#     - BLQ：Below Average Living Quarters
#     - Rec：Average Rec Room
#     - LwQ：Low Quality
#     - Unf：Unfinshed
#     - NA：No Basement
#        
# - BsmtFinSF1: Type 1 finished square feet，1型方形脚。
# 
# - BsmtFinType2: Rating of basement finished area (if multiple types)，第二个完成区域的质量（如果存在）。
#     - GLQ：Good Living Quarters
#     - ALQ：Average Living Quarters
#     - BLQ：Below Average Living Quarters
#     - Rec：Average Rec Room
#     - LwQ：Low Quality
#     - Unf：Unfinshed
#     - NA：No Basement
# 
# - BsmtFinSF2: Type 2 finished square feet，2型完成的平方英尺。
# 
# - BsmtUnfSF: Unfinished square feet of basement area，未完成的地下室面积。
# 
# - TotalBsmtSF: Total square feet of basement area，地下室面积的平方英尺。
# - 可见相关性比较高
basement_cols = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']
# 离散的列填充None,数值的列的已经是0了。
for col in basement_cols:
    if 'FinSF' not in col:
        train[col]=train[col].fillna("None")

# ### 1.6 FireplaceQu填充
# - FireplaceQu: Fireplace quality， 壁炉质量。
#     - Ex：Excellent - Exceptional Masonry Fireplace
#     - Gd：Good - Masonry Fireplace in main level
#     - TA：Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
#     - Fa：Fair - Prefabricated Fireplace in basement
#     - Po：Poor - Ben Franklin Stove
#     - NA：No Fireplace
# 相关性不大，直接填充None
train["FireplaceQu"]=train["FireplaceQu"].fillna("None")

# ### 1.7 Garage群的填充
# 
# - GarageType: Garage location，车库位置。
#     - 2Types: More than one type of garage
#     - Attchd: Attached to home
#     - Basment: Basement Garage
#     - BuiltIn: Built-In (Garage part of house - typically has room above garage)
#     - CarPort: Car Port
#     - Detchd: Detached from home
#     - NA: No Garage
# 
# - GarageYrBlt: Year garage was built，年建车库。
# 
# - GarageFinish: Interior finish of the garage，车库内部装修。
#     - Fin: Finished
#     - RFn: Rough Finished	
#     - Unf: Unfinished
#     - NA: No Garage
# 
# - GarageCars: Size of garage in car capacity，车库的车库容量。
# 
# - GarageArea: Size of garage in square feet，平方英尺车库大小。
# 
# - GarageQual: Garage quality，车库质量。
#     - Ex: Excellent
#     - Gd: Good
#     - TA: Typical/Average
#     - Fa: Fair
#     - Po: Poor
#     - NA: No Garage
# 
# - GarageCond: Garage condition，车库条件。
#     - Ex: Excellent
#     - Gd: Good
#     - TA: Typical/Average
#     - Fa: Fair
#     - Po: Poor
#     - NA: No Garage

garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']
for col in garage_cols:
    if train[col].dtype == np.object:
        train[col]=train[col].fillna("None")
    else:
        train[col]=train[col].fillna(0)

# ### 1.8 PoolQC的填充
# - Pool quality,  游泳池质量
#     - Ex：Excellent
#     - Gd：Good
#     - TA：Average/Typical
#     - Fa：Fair
#     - NA：No Pool
# 基本都是空值
train.PoolQC = train.PoolQC.fillna("None")

# ### 1.9 Fence的填充
# - Fence: Fence quality，栅栏质量。
#     - GdPrv：Good Privacy
#     - MnPrv：Minimum Privacy
#     - GdWo：Good Wood
#     - MnWw：Minimum Wood/Wire
#     - NA：No Fence
# 因为总共有5个类型，然而可以统计出来的类型一共只有4类，因此可以断定最后一类NA用空值代替
train["Fence"]=train["Fence"].fillna("None")

# ### 1.10 MiscFeature的填充
# - Miscellaneous feature not covered in other categories，其余特征。
# - Elev：Elevator
# - Gar2：2nd Garage (if not described in garage section)
# - Othr：Other
# - Shed：Shed (over 100 SF)
# - TenC：Tennis Court
# - NA：None

# 因为MiscFeature本来有5类，然而只统计出4类，因此可以断定第五类NA是空值
train["MiscFeature"]=train["MiscFeature"].fillna("None")

train = train.drop(['SqrtLotArea'], axis=1)
train = train.drop(['LotAreaCut'], axis=1)

print("After fill shape:", train.shape)

# 保存数据
train.to_csv('./data/train_2.csv')


# ## 2 模型训练部分
# - 数据分析完成了，需要开始机器学习训练了。
# - 需要考虑各种单独的模型的各自表现后，筛选出表现较好的模型，并进行集成学习。单独的模型包括：
# 
# - 1.线性回归类模型
#     - 1.1 朴素线性回归
#     - 1.2 基于L1的线性回归   
#     - 1.3 基于L2的线性回归 
#     - 1.4 ElasticNet（弹性网络）
#     
# - 2.树回归-CART
#     - 2.1 CART
#     - 2.2 RF
#     - 2.3 AdaBoost
#     - 2.4 GBDT--XGBoost--lightGBM
#     
# - 3.SVM类型
#     - 3.1SVR
#     
# - 4.神经网络
#     - 4.1FC神经网络
# 
# - 5.集成学习
#     - 5.1 Stacking Ensemble

print("Start fiting...")

new_train = pd.read_csv("./data/train_2.csv")
y = new_train["SalePrice"]
train1 = new_train.drop(["Id", "SalePrice"], axis=1)
X = pd.get_dummies(train1).reset_index(drop=True)
if 'Unnamed: 0' in X.columns:
    X =X.drop(['Unnamed: 0'], axis=1)

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=123)


def benchmark(model):
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    logrmse = np.sqrt(mean_squared_error(np.log(y_test), np.log(pred)))
    print("RMSE: {} \nLOGRMSE: {}".format(rmse, logrmse))
    return rmse, logrmse


#  XGboost
xg_reg=xgb.XGBRegressor(objective='reg:linear',
                        colsample_bytree=0.7,
                        learning_rate=0.01,
                        max_depth=3,
                        n_estimators=3000,
                        subsample=0.7,
                        reg_alpha=0.0006,
                        nthread=6)
xg_reg.fit(X_train,y_train)

benchmark(xg_reg)







