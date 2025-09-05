import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

n_samples = 200


#occhi chiusi:
closed_eyes = np.column_stack([
    np.random.normal(loc=15, scale=2, size=n_samples),  # Oz alpha: alta
    np.random.normal(loc=5, scale=1, size=n_samples),   # C3 beta: bassa
    np.random.normal(loc=8, scale=1, size=n_samples)    # Fz var: media
])

#occhi aperti:
open_eyes = np.column_stack([
    np.random.normal(loc=5, scale=2, size=n_samples),   #Oz α: bassa
    np.random.normal(loc=15, scale=1, size=n_samples),  #C3 β: alta
    np.random.normal(loc=8, scale=1, size=n_samples)    #Fz var: media
])

X = np.vstack([closed_eyes, open_eyes])  
y = np.array(["chiusi"] * n_samples + ["aperti"] * n_samples)  


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nreport di classificazione")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["chiusi", "aperti"],
            yticklabels=["chiusi", "aperti"])
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.title("Matrice di Confusione - RF su EEG")
plt.tight_layout()
plt.show()

