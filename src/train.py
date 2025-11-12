import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np

a = np.array({1, 2, 3})
print(a)

# def train_model(features_csv: std):
#     df = pd.read_csv(features_csv)
#     x = df.drop(columns=['qoe_proxy'])
#     y = df['qoe_proxy']

#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#     model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
#     model.fit(x_train, y_train)

#     predictions = model.predict(x_test)
#     rmse = mean_squared_error(y_test, predictions, squared=False)
#     print(f"Test RMSE: {rmse}")

#     return model