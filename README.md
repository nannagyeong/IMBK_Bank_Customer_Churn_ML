# 고객 이탈 분류 ML 및 인사이트 분석
---
기간 : 2026년 4월 10일
--
| Category | Libraries |
|----------|----------|
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| ML Framework | scikit-learn |
| Ensemble Models | lightgbm, xgboost, catboost |
| AutoML | pycaret |
| Tuning | optuna |
| Interpretation | shap |


데이터 출처: 캐글 Bank Customer Churn Dataset (row: 10000, col:12)

## 전처리
- 고객 식별용 변수(`customer_id`) 제거 → 분석에 불필요
- 범주형 변수 인코딩
  - `country`, `gender` → Label Encoding 적용
- 데이터 구조를 모델 학습에 적합한 형태로 변환
## EDA 및 해석
<img width="746" height="495" alt="churn count" src="https://github.com/user-attachments/assets/57b0aa1b-c317-4037-a4c4-0439eb247c0c" />
- 이탈하지 않은 고객(0)이 대부분을 차지하고, 이탈 고객(1)은 상대적으로 적음  

**데이터가 불균형 구조를 가지고 모델 평가 시 F1-score 중심 평가가 필요함**

<img width="730" height="481" alt="Age " src="https://github.com/user-attachments/assets/5e31d302-4353-4e43-8d4b-4e9e4207ae0a" />
- 고객 연령이 30~40대에 집중되어 있음  

→ 데이터가 특정 연령대에 편중된 구조를 가짐  
→ **모델이 특정 연령대 패턴에 영향을 받을 가능성 존재**
<img width="507" height="327" alt="Balance" src="https://github.com/user-attachments/assets/6f610267-45b8-4627-922a-4bd03c92ae7c" />
- 잔액이 0인 고객이 다수 존재함  

→ 잔액이 없는 고객군은 서비스 이용도가 낮거나 비활성 상태일 가능성이 있음  
→ **이탈 위험군으로 이어질 수 있어 별도의 관리 전략이 필요함**
## AutoML – Hyperparameter Tuning – Stacking Pipe – Shap value
<img width="738" height="433" alt="image" src="https://github.com/user-attachments/assets/abd91e20-9f44-45e0-9427-7534bb0b9058" />

### 모델 선정 기준

각 모델의 F1-score를 비교한 결과는 다음과 같음:

- Random Forest: 0.585  
- SVM: 0.543  
- KNN: 0.483  
- Logistic Regression: 0.229  

→ Random Forest가 가장 높은 성능을 보였으며, 전반적으로 비선형 모델(RF, SVM, KNN)이 선형 모델(Logistic Regression)보다 높은 성능을 보임  

→ 이는 EDA에서 확인한 바와 같이 변수 간 상관관계가 낮아 단순 선형 관계로 설명하기 어려운 구조임을 시사함  

### 모델 선택 이유

따라서 비선형 패턴을 효과적으로 학습할 수 있는 Random Forest를 SHAP 분석 해석 모델로 선정하였으며,  SHAP 분석을 통해 주요 변수의 영향도를 해석함
<img width="932" height="681" alt="shap" src="https://github.com/user-attachments/assets/03d08d24-c19d-4f7c-ba2a-f51eba5d889f" />

## SHAP 기반 모델 해석

SHAP 분석을 통해 변수별 churn 예측에 대한 영향도를 확인한 결과는 다음과 같음:

### 주요 변수

- **Age**
  - 값이 증가할수록 SHAP value가 양의 방향으로 증가 (최대 약 0.4)
  → 연령이 높을수록 churn 확률이 증가하는 경향

- **NumOfProducts**
  - 값이 증가할수록 SHAP value가 크게 증가 (약 0.5~0.6 수준)
  → 상품 수가 많을수록 이탈 가능성이 높아지는 특징

- **IsActiveMember**
  - 비활성(0)일 때 SHAP value는 양의 방향 (약 +0.2)
  - 활성(1)일 때 SHAP value는 음의 방향 (약 -0.2)
  → 비활성 고객일수록 churn 가능성이 높고, 활성 고객은 이탈 가능성이 낮음

---

###  보조 변수

- **CreditScore**
  - 낮은 값에서 일부 양의 영향이 있으나 전체적인 영향력은 제한적

- 기타 변수
  - SHAP value가 대부분 0 근처에 분포
  → churn에 미치는 영향이 상대적으로 작음

---

##  인사이트 및 전략 제안

1. **연령 기반 고객 관리**
- 연령이 높을수록 churn 가능성이 증가하는 경향이 확인됨  
→ 고연령 고객을 대상으로 한 맞춤형 유지 전략 필요  

2. **고객 활동성 관리**
- IsActiveMember 변수에서 활동 여부가 churn에 큰 영향을 미치는 것으로 나타남  
→ 비활성 고객을 활성화하기 위한 리텐션 전략이 중요  

3. **상품 보유 수 기반 관리**
- 상품 수가 많을수록 churn 가능성이 증가하는 경향 확인  
→ 상품 수 증가가 반드시 고객 충성도로 이어지지 않음  
→ 다상품 보유 고객에 대한 집중 관리 필요  

4. **구간 기반 고객 분석 필요**
- CreditScore 등 일부 변수는 특정 구간에서만 영향력 존재  
→ 전체 평균이 아닌 구간별(segmentation) 분석 기반 전략 필요  

5. **고객 세그먼트 기반 전략**
- 성별, 국가별 churn 비율 차이 존재  
→ 단순 평균 분석이 아닌 고객 그룹별 맞춤 전략 필요

