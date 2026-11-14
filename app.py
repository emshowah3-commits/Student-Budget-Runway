import joblib

clf = joblib.load("data/categorizer.joblib")

prediction = clf.predict(["POS PURCHASE TIM HORTONS"])