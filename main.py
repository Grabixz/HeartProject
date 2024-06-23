import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

heart_info = pd.read_csv('heart.csv', delimiter=';')
connection = sqlite3.connect('heart.db')
heart_info.to_sql('heart', connection, if_exists='replace', index=False)
connection.commit()
dataframe = pd.read_sql_query("SELECT * FROM heart", connection)

dataframe.dropna(inplace=True)
categorical_variables = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
target_variable = 'target'
label_encoder = LabelEncoder()

for variable in categorical_variables:
    dataframe[variable] = label_encoder.fit_transform(dataframe[variable])
numeric_variables = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
dataframe[numeric_variables] = scaler.fit_transform(dataframe[numeric_variables])

X = dataframe.drop(columns=[target_variable])
y = dataframe[target_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 20))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

for item, variable in enumerate(categorical_variables):
    ax = axes[item // 2, item % 2]
    dataframe.groupby([target_variable, variable]).size().unstack().plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f"Distribution of {variable} for {target_variable}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Count")
    ax.legend(title=target_variable)

numeric_variables = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

plt.figure(figsize=(16, 6))
for index, column in enumerate(numeric_variables, 1):
    plt.subplot(2, 3, index)
    dataframe[dataframe['target'] == 0][column].hist(alpha=0.5, label='target=0', density=True)
    dataframe[dataframe['target'] == 1][column].hist(alpha=0.5, label='target=1', density=True)
    plt.title(column)
    plt.xlabel('')
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.show()

joblib.dump(random_forest, 'heart_disease_model.pkl')

connection.close()
