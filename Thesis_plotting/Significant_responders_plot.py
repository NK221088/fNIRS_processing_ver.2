import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
responders_count_path = Path(os.getenv("Marwan_responders_count"))
df = pd.read_csv(rf"{responders_count_path}", index_col=0)
data = df['count'].to_dict()
total_number_of_patients = 50
threshold = 7
print("Responders:")
all_responders = [f"P{str(ID)}" for ID in range(1, total_number_of_patients + 1)]
covert_responders = [ID for ID, count in data.items() if count >= threshold]
non_covert_responders = [ID for ID, count in data.items() if count < threshold]
non_responders = [ID for ID in all_responders if ID not in covert_responders and ID not in non_covert_responders]
print("Covert Consciousness significant responders:", np.sort(covert_responders), len(covert_responders))
print("Non-Covert Consciousness significant responders:", np.sort(non_covert_responders), len(non_covert_responders))
print("Non-responders:" , np.sort(non_responders), len(non_responders))
counts = Counter(data.values())
keys = list(counts.keys())
for i in range(np.max(keys) + 1):
    if i not in keys:
        if i == 0:
            counts[i] = total_number_of_patients - len(data.keys())
        else:
            counts[i] = 0
df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
df.rename(columns={'index': 'Significant counts', 0: 'Number of Participants'}, inplace=True)

fig, ax = plt.subplots(figsize=(3.5, 3))

df = df.sort_values("Significant counts")

counts = df["Significant counts"]
weights = df["Number of Participants"]

bars = ax.bar(
    counts,
    weights,
    edgecolor="black",
    linewidth=0.6
)

for bar, x in zip(bars, counts):
    if x >= threshold:
        bar.set_color("0.25")   # responders
    else:
        bar.set_color("0.75")   # non-responders

ax.axvline(
    threshold - 0.5,
    linestyle="--",
    color="0.4",
    linewidth=0.8
)

ax.set_xlabel("Number of significant contrasts")
ax.set_ylabel("Number of subjects")

ax.set_xticks(counts)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("significant_responders_plot.png", dpi=600)