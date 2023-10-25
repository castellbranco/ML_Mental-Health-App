import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

# Load the dataset
df = pd.read_csv(r'C:\Users\Lenovo\Desktop\ML App\final.csv')

# Prepare the data for training
X = df.drop(columns=['Would you feel comfortable discussing a mental health issue with your coworkers?',
                     'Would you feel comfortable discussing a mental health issue with your direct supervisor(s)?',
                     'Unnamed: 0'], axis=1)

Y1 = df['Would you feel comfortable discussing a mental health issue with your coworkers?']
Y2 = df['Would you feel comfortable discussing a mental health issue with your direct supervisor(s)?']

# Split the data into training and testing sets
X_train_1, X_test_1, y1_train, y1_test = train_test_split(X, Y1, test_size=0.2, random_state=25)
X_train_2, X_test_2, y2_train, y2_test = train_test_split(X, Y2, test_size=0.2, random_state=25)

# Define the models with the final parameters
xgboost_model_y1 = XGBClassifier(**{'n_estimators': 35, 'max_depth': 26})
xgboost_model_y2 = XGBClassifier(**{'n_estimators': 35, 'max_depth': 26})

random_forest_model_y1 = RandomForestClassifier(**{'n_estimators': 35, 'max_depth': 26, 'random_state': 10})
random_forest_model_y2 = RandomForestClassifier(**{'n_estimators': 35, 'max_depth': 26, 'random_state': 10})

svm_model_y1 = SVC(**{'kernel': 'rbf', 'gamma': 0.0001, 'C': 100, 'probability': True})
svm_model_y2 = SVC(**{'kernel': 'rbf', 'gamma': 0.0001, 'C': 100, 'probability': True})

ada_model_y1 = AdaBoostClassifier(**{'n_estimators': 180, 'learning_rate': 0.201, 'random_state': 0})
ada_model_y2 = AdaBoostClassifier(**{'n_estimators': 180, 'learning_rate': 0.201, 'random_state': 0})

bagging_model_y1 = BaggingClassifier(**{'estimator': RandomForestClassifier(n_estimators=100, random_state=42),
                                         'n_estimators': 50, 'random_state': 42})
bagging_model_y2 = BaggingClassifier(**{'estimator': RandomForestClassifier(n_estimators=100, random_state=42),
                                         'n_estimators': 50, 'random_state': 42})

decision_tree_model_y1 = DecisionTreeClassifier(**{'criterion': 'gini', 'max_depth': 150, 'random_state': 0})
decision_tree_model_y2 = DecisionTreeClassifier(**{'criterion': 'gini', 'max_depth': 150, 'random_state': 0})

# Train the models
xgboost_model_y1.fit(X_train_1, y1_train)
xgboost_model_y2.fit(X_train_2, y2_train)

random_forest_model_y1.fit(X_train_1, y1_train)
random_forest_model_y2.fit(X_train_2, y2_train)

svm_model_y1.fit(X_train_1, y1_train)
svm_model_y2.fit(X_train_2, y2_train)

ada_model_y1.fit(X_train_1, y1_train)
ada_model_y2.fit(X_train_2, y2_train)

bagging_model_y1.fit(X_train_1, y1_train)
bagging_model_y2.fit(X_train_2, y2_train)

decision_tree_model_y1.fit(X_train_1, y1_train)
decision_tree_model_y2.fit(X_train_2, y2_train)

# Save the models
joblib.dump(xgboost_model_y1, 'xgboost_model_y1.pkl')
joblib.dump(xgboost_model_y2, 'xgboost_model_y2.pkl')

joblib.dump(random_forest_model_y1, 'random_forest_model_y1.pkl')
joblib.dump(random_forest_model_y2, 'random_forest_model_y2.pkl')

joblib.dump(svm_model_y1, 'svm_model_y1.pkl')
joblib.dump(svm_model_y2, 'svm_model_y2.pkl')

joblib.dump(ada_model_y1, 'ada_model_y1.pkl')
joblib.dump(ada_model_y2, 'ada_model_y2.pkl')

joblib.dump(bagging_model_y1, 'bagging_model_y1.pkl')
joblib.dump(bagging_model_y2, 'bagging_model_y2.pkl')

joblib.dump(decision_tree_model_y1, 'decision_tree_model_y1.pkl')
joblib.dump(decision_tree_model_y2, 'decision_tree_model_y2.pkl')

print("Training completed and models saved!")
