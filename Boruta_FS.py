import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy



X, y = make_classification(n_samples=100, n_features=20, random_state=42)

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)

boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)

boruta_selector.fit(X, y)

selected_features = X[:, boruta_selector.support_]

print("Selected features:")
print(selected_features)

print("Feature ranking:")
print(boruta_selector.ranking_)

selected_feature_names = np.array([boruta_selector.support_])
print("Selected feature names:")
print(selected_feature_names)