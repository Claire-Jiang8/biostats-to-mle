# heart_analysis.py - Week 1 Clinical Project
import pandas as pd
from statistics import mean

# Load the dataset (heart.csv)
df = pd.read_csv('heart.csv')

# Extract columns (age = column 0, chol = column 4, target = column 13 in UCI format)
ages = df['age'].dropna().tolist()
cholesterol = df['chol'].dropna().tolist()
has_disease = df['target'].dropna().tolist()

# Analyze
print(f"Patients: {len(df)}")
print(f"Average Age: {mean(ages):.1f} years")
print(f"Average Cholesterol: {mean(cholesterol):.1f} mg/dl")
print(f"Heart Disease Rate: {sum(has_disease)/len(has_disease)*100:.1f}%")

# Save simple plot for GitHub
import matplotlib.pyplot as plt
plt.hist(ages, bins=10)
plt.title('Age Distribution')
plt.savefig('age_hist.png')
plt.show()