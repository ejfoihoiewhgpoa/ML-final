import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # lùi lên thư mục main
DATA = os.path.join(BASE, "my_dataset")

df = pd.read_csv(os.path.join(DATA, "train.csv"))
print(df.shape)
print(df.phrase.value_counts())

seq = pd.read_parquet(os.path.join(DATA, df.loc[0, "path"]))
print(seq.shape)                                        # (số frame, 1630)
print(seq.filter(like="hand").isna().mean().mean())     # tỉ lệ NaN của tay

print("left hand:", seq.filter(like="left_hand").isna().mean().mean())
print("right hand:", seq.filter(like="right_hand").isna().mean().mean())
print("face:", seq.filter(like="face").isna().mean().mean())
print("pose:", seq.filter(like="pose").isna().mean().mean())