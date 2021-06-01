#!/usr/bin/env python
# coding: utf-8

# ## 0. 라이브러리 정의 

# In[78]:


# 데이터 처리 라이브러리
import pandas as pd
import numpy as np

# 분석 알고리즘 decision tree 구현 라이브러리
# 설명력은 좀 낮지만 정확도 높음 
from sklearn.tree import DecisionTreeRegressor

# 과거 데이터를 8:2 ,7:3 으로 자동으로 나누어주는 라이브러리 
from sklearn.model_selection import train_test_split

# 라벨인코더 -> 문자를 숫자로 맵핑시켜준다. 두개는 0,1 세게는 0, 1, 2 네개는 0, 1 ,2 ,3
from sklearn.preprocessing import LabelEncoder

# 분석regression ; 대표 평가지표 ( MAE, RMSE )
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# In[3]:


# from sklearn.ensemble import RandomForestRegressor
# 설명력이 뛰어남

# from sklearn.linear_model import LinearRegression
# 연속적 예측


# In[34]:


# CSV 파일을 읽어 DataFrame 변수에 저장하기
featuresData =  pd.read_csv("../dataset/feature_regression_example.csv")


# ## 1.  데이터 전처리

# ### 1-1. 타입 통합 / 특성 숫자 컬럼 추가

# ### 1-1-1. 데이터 타입 통합

# In[35]:


featuresData.info()


# In[36]:


# 주의할 사항은 모든 컬럼에 대해서 고정시키는 걸 고려하는 게 나을 수도 있다 
featuresData.QTY = featuresData.QTY.astype(float)


# ### 1-1-2. 특성값 숫자 컬럼 변경

# ##### 이유 : 머신러닝 특성은 숫자형 지원 해결. Y ->1  N ->0 과 같이 변경 

# In[37]:


featuresData.columns


# In[12]:


featuresData.PROMOTION.drop_duplicates()


# In[38]:


featuresData["HOLIDAY_NEW"] = np.where(featuresData.HOLIDAY == "Y" ,1,0)


# In[39]:


featuresData["PROMOTION_NEW"] = np.where(featuresData.PROMOTION == "Y" ,1,0)


# In[6]:


# 아래와 같은 모양도 가능 
# pd.DataFrame( featuresData.HOLIDAY.drop_duplicates()).reset_index().rename(columns={index:"HOLIDAY_NEW"})


# In[24]:


## labelencoder 이용, 알아서 변환해준다.

## 라벨인코더 인스턴스 만들고
holiEn = LabelEncoder()

## 변환 실행
featuresData["HOLIDAY_LABEL_EN"] = holiEn.fit_transform(featuresData.HOLIDAY)


# ### 1-2. 특성선정 / 데이터 분리

# #### 1-2-1. 특성선정 

# In[40]:


corrDf = featuresData.corr()


# In[41]:


standardLimit = 0.5


# In[28]:


# 반응하는 변수들 
corrDf.loc[ (abs(corrDf.QTY) > standardLimit) & (corrDf.QTY !=1)]


# In[42]:


# standard limit에 따라  features가 바뀔 수 있음
features = list(corrDf.loc[ (abs(corrDf.QTY) > standardLimit) & (corrDf.QTY !=1)].index)


# In[43]:


features


# In[44]:


label = ["QTY"]


# In[ ]:





# ### 1-2-2. 데이터 분리

# In[50]:


standardIndex = 0.8


# In[51]:


featuresData.shape


# In[52]:


sortKey = ["REGIONID", "ITEM","YEARWEEK"]


# In[53]:


sortedData = featuresData.sort_values(sortKey, ignore_index = True)


# In[54]:


selectedIndex = int (list(sortedData.shape) [0] * standardIndex )   
# 정렬하고 80 % 에 있는 인댁스의 번호로 분리시킨다 


# In[57]:


yearweekStd = sortedData.loc[selectedIndex].YEARWEEK


# In[62]:


# 훈련데이터와 테스트 데이터를 문제지와 정답지로 정의해서 구분한다 (문제, 정답 , 문제, 정답)
trainingDataFeatures =sortedData.loc[sortedData.YEARWEEK <= yearweekStd, features]
trainingDataLabel =sortedData.loc[sortedData.YEARWEEK <= yearweekStd, label]
testDataFeatures =sortedData.loc[sortedData.YEARWEEK > yearweekStd, features]
testDataLabel =sortedData.loc[sortedData.YEARWEEK > yearweekStd, label]


# In[63]:


trainingDataFeatures


# ### 2. 모델 적용

# ### 2-1. 모델적용

# ### 2-2-1. 학습

# In[65]:


model = DecisionTreeRegressor(random_state =10)


# In[67]:


# x는 문제지 y는 결과물 
model.fit(X= trainingDataFeatures, y= trainingDataLabel) 


# ### 3. 예측

# In[70]:


predictValue = model.predict(testDataFeatures)


# In[73]:


predictDf =  pd.DataFrame(list(predictValue), columns = ["PREDICT"])


# ### 4. 데이터 정리 

# In[81]:


validateDf = pd.concat([testDataLabel.reset_index(drop=True),predictDf], axis=1)


# In[82]:


validateDf


# In[ ]:


### 5. 정확도 검증


# In[85]:


mae = mean_absolute_error( y_true= validateDf.QTY,
                 y_pred= validateDf.PREDICT)


# In[86]:


rmse = np.sqrt ( mean_squared_error(y_true= validateDf.QTY,
                 y_pred= validateDf.PREDICT))


# In[87]:


mae


# In[ ]:


### 장점 
### decision tree => 과거의 경험치 그대로 반영한다. 변동성이 큰 데이터에서 강하다. 설명력이 강하다 
### 단점
### 오버피팅 => 과거에 집착이 오짐
### 단점 해결 방안 : random forest => 이런 트리를 여러개 만들어
### 장점 : decision tree 오버피팅을 해결함
### 단점 : 설명력이 부족함 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### [1. 타입 통합 / 특성 숫자컬럼 추가]
# ### YEARWEEK, YEAR, WEEK를 int 타입으로 설정해보고 홀리데이 및 프로모션 여부 컬럼에 대해서 Y→1, N→0 컬럼을
# ### HO_YN 컬럼 -> HOLIDAY(Y) -> 1, HOLIDAY(N) -> 0
# ###  PRO_YN -> PROMOTION(Y)->1, PROMOTION(N) -> 0  추가로 생성하세요

# In[10]:


feauresData = featuresData.astype( { "YEARWEEK" : int,
                                       "YEAR" : int,
                                       "WEEK" : int })


# In[ ]:





# In[ ]:




