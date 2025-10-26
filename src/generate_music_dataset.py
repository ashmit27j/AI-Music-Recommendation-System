import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

N = 2000  # Total samples (you can increase for more realism)
classes = ['happy', 'sad', 'energetic', 'calm']
data = []
for _ in range(N):
    label = np.random.choice(classes)
    # Features with realistic overlap, not perfectly separable
    danceability = np.clip(np.random.normal(loc={'happy':0.8,'sad':0.4,'energetic':0.9,'calm':0.5}[label], scale=0.1), 0, 1)
    energy = np.clip(np.random.normal(loc={'happy':0.7,'sad':0.4,'energetic':0.9,'calm':0.5}[label], scale=0.15), 0, 1)
    valence = np.clip(np.random.normal(loc={'happy':0.8,'sad':0.3,'energetic':0.7,'calm':0.5}[label], scale=0.15), 0, 1)
    acousticness = np.clip(np.random.beta(2,3), 0, 1)
    instrumentalness = np.clip(np.random.beta(2,3), 0, 1)
    liveness = np.clip(np.random.beta(2,3), 0, 1)
    speechiness = np.clip(np.random.beta(2,3), 0, 1)
    loudness = round(np.random.uniform(1,5), 2)
    tempo = round(np.random.uniform(60,200), 2)
    duration_ms = int(np.random.uniform(120000, 300000))
    key = int(np.random.randint(0,12))
    mode = int(np.random.randint(0,2))
    data.append([danceability, energy, valence, acousticness, instrumentalness,
                 liveness, speechiness, loudness, tempo, duration_ms, key, mode, label])

df = pd.DataFrame(data, columns=[
    'danceability', 'energy', 'valence', 'acousticness', 'instrumentalness',
    'liveness', 'speechiness', 'loudness', 'tempo', 'duration_ms', 'key', 'mode', 'emotion_label'
])
# Shuffle for randomness
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Train/test split (80/20)
n_train = int(0.8 * N)
df.iloc[:n_train].to_csv("train.csv", index=False)
df.iloc[n_train:].to_csv("test.csv", index=False)

print(f"Train and test datasets generated: train.csv ({n_train} rows), test.csv ({N-n_train} rows)")
print("Columns:", df.columns.tolist())
print("Example rows:")
print(df.head())
