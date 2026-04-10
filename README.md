# 고객 이탈 분류 ML 및 인사이트 분석
---
기간 : 2026년 4월 10일
--
기술스택:
| Category | Libraries |
|----------|----------|
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| ML Framework | scikit-learn |
| Ensemble Models | lightgbm, xgboost, catboost |
| AutoML | pycaret |
| Tuning | optuna |
| Interpretation | shap |
--
데이터 출처: 캐글 Bank Customer Churn Dataset (row: 10000, col:12)

## 전처리
- 고객 식별용 변수(`customer_id`) 제거 → 분석에 불필요
- 범주형 변수 인코딩
  - `country`, `gender` → Label Encoding 적용
- 데이터 구조를 모델 학습에 적합한 형태로 변환
## EDA 및 해석
<img width="746" height="495" alt="churn count" src="https://github.com/user-attachments/assets/57b0aa1b-c317-4037-a4c4-0439eb247c0c" />
이탈하지 않은 고객(0)이 대부분을 차지하고, 이탈 고객(1)은 상대적으로 적음
**데이터가 불균형 구조를 가짐**
**평가 시 Accuracy보다 F1-score 중심 평가 필요 판단**
<img width="730" height="481" alt="Age " src="https://github.com/user-attachments/assets/5e31d302-4353-4e43-8d4b-4e9e4207ae0a" />
그래프를 보면 대부분 고객이 30~40대에 집중되어 있음
**데이터가 특정 연령대에 몰려있음**
<img width="507" height="327" alt="Balance" src="https://github.com/user-attachments/assets/6f610267-45b8-4627-922a-4bd03c92ae7c" />
잔액이 0에 몰려있는 것을 확인 할 수 있음
**잔액이 없는 고객군은 이용도가 낮거나 비활성 상태일 가능성이 존재하며 관리 필요**
8. AutoML – Hyperparameter Tuning – Stacking Pipe – Shap value
9. 인사이트 제안
10. Reference
