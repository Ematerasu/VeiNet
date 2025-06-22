import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../checkpoints/metrics.csv")

df_percent = df.copy()
for col in df.columns[2:]:
    df_percent[col] = df[col] * 100

df_percent.reset_index(inplace=True)

plt.figure(figsize=(14, 6))
for col in df_percent.columns[3:]:
    plt.plot(df_percent.index, df_percent[col], label=col)

plt.title("Winratio against Vei bot")
plt.xlabel("Checkpoint index")
plt.ylabel("Winratio [%]")
plt.ylim(0, 101)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
