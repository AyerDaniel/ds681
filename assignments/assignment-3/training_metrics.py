import pandas as pd

df = pd.read_csv("/home/johnsmith/Desktop/njit/workspaces/ds681/eng-ai-agents-main/assignments/assignment-3/runs/yolo-drone-20260306-185910/results.csv")
df.columns = df.columns.str.strip()
df["F1"] = 2 * (df["metrics/precision(B)"] * df["metrics/recall(B)"]) / \
                (df["metrics/precision(B)"] + df["metrics/recall(B)"])
print(df[["epoch", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "F1"]])
