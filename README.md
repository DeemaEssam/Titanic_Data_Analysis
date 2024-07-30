
# Titanic Data Analysis

## Project Overview

- **Dataset**: Titanic survivors dataset available on Kaggle. [Link to Dataset](https://www.kaggle.com/competitions/titanic/data)
- **Objective**: Analyze and predict passenger survival using various machine learning techniques.

## Steps Covered

1. **Importing Dependencies**
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score
   ```

2. **Reading the Data**
   ```python
   data = pd.read_csv('/content/train.csv')
   data.head()
   ```

3. **Data Preprocessing**
   - Checking for and handling missing values.
   - Dropping unnecessary columns.
   - Encoding categorical variables.

4. **Handling Missing Data**
   ```python
   data['Age'].fillna(data['Age'].mean(), inplace=True)
   data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
   data = data.drop(columns=['Cabin'])
   data = data.drop(columns=['PassengerId', 'Name'])
   ```

5. **Encoding Categorical Variables**
   ```python
   data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)
   ```

6. **Handling Duplicates**
   ```python
   duplicates = data.duplicated().sum()
   print(f'Number of duplicate rows: {duplicates}')
   data.drop_duplicates(inplace=True)
   ```

7. **Model Building**
   - Separating features and target variable.
   - Splitting data into training and testing sets.
   - Training a Logistic Regression model.
   ```python
   X_train = data.drop(columns=['Survived'])
   y_train = data['Survived']
   X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

   model = LogisticRegression()
   model.fit(X_train, y_train)
   ```

8. **Model Evaluation**
   ```python
   y_pred = model.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   print(f'Accuracy: {accuracy:.2f}')
   ```

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## How to Run

1. Ensure you have all dependencies installed.
2. Download the Titanic dataset and place it in the appropriate directory.
3. Run the provided Python script or Jupyter notebook to perform the analysis and train the model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize it further based on any specific requirements or additional information you may want to include.
