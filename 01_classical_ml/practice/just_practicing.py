import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame({
    "name": ["Pratyush", "Rahul", "Amit"],
    "level": ["Junior", "Senior", "Mid"]
})

encoder = LabelEncoder()
df["level_encoded"] = encoder.fit_transform(df["level"])

print(df)