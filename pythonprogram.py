
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

# 1. Healthcare
print("\n1. HEALTHCARE")
data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25,
    random_state=7
)

model = RandomForestClassifier(
    n_estimators=150, random_state=7
)
model.fit(X_train, y_train)

print("Accuracy:",
      accuracy_score(y_test, model.predict(X_test)))

patient = X_test[0].reshape(1, -1)
print("Prediction:",
      data.target_names[model.predict(patient)[0]])

# 2. Banking - Fraud Detection
print("\n2. BANKING")

rng = np.random.default_rng(7)

normal = np.column_stack([
    rng.normal(1500, 500, 500).clip(100),
    rng.normal(14, 3, 500).clip(7, 22)
])

suspicious = np.array([
    [95000, 3], [120000, 2], [78000, 4]
])

transactions = np.vstack([normal, suspicious])

detector = IsolationForest(
    contamination=0.01, random_state=7
)
detector.fit(transactions)

tests = [[1800, 13], [99000, 3]]

for amount, hour in tests:
    result = detector.predict([[amount, hour]])
    if result[0] == -1:
        print(amount, "Suspicious transaction")
    else:
        print(amount, "Normal transaction")

# 3. Recommendation System
print("\n3. RECOMMENDATION SYSTEM")

movies = [
    "Action Hero", "Space Quest", "Love in Paris",
    "Romantic Summer", "Robot Wars", "Funny Family"
]

ratings = np.array([
    [5, 4, 1, 0, 5, 2],
    [4, 5, 0, 1, 4, 1],
    [1, 0, 5, 5, 1, 3],
    [0, 1, 4, 5, 0, 4],
    [5, 5, 0, 0, 0, 0]
])

similarity = cosine_similarity(ratings.T)
user = ratings[4]
scores = similarity @ user
scores[user > 0] = -1

top = np.argsort(scores)[::-1][:2]

print("Recommended movies:")
for i in top:
    print(movies[i])

# 4. Agriculture - Crop Yield Prediction
print("\n4. AGRICULTURE")

n = 200
rainfall = rng.uniform(300, 1200, n)
soil_ph = rng.uniform(5.5, 8.0, n)
fertilizer = rng.uniform(20, 120, n)

yield_ton = (
    0.004 * rainfall
    + 0.015 * fertilizer
    - 0.3 * abs(soil_ph - 6.8)
    + rng.normal(0, 0.15, n)
)

X = np.column_stack([rainfall, soil_ph, fertilizer])

X_train, X_test, y_train, y_test = train_test_split(
    X, yield_ton, test_size=0.25, random_state=7
)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)
print("Prediction Error:",
      mean_absolute_error(y_test, prediction))

field = [[850, 6.8, 80]]
print("Predicted crop yield:",
      model.predict(field)[0], "tonnes/acre")

print("\nResult: AI applications demonstrated successfully.")
