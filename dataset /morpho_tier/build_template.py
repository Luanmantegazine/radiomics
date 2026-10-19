"""Create the CN/AD tier table with empty fs_* columns, ready to be filled.

Rows kept: training_eligible == True and supervised_label in {CN, AD}.
MCI, unlabelled, other labels are dropped. No age/sex columns are carried.
"""
import sys
import pandas as pd
from common import ALL_FS_COLUMNS, ID_COLUMNS

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "morpho_tier_template.csv"
df = pd.read_csv(src, low_memory=False)
keep = df[(df.training_eligible == True) & df.supervised_label.isin(["CN", "AD"])]
tier = keep[ID_COLUMNS].copy()
for c in ALL_FS_COLUMNS:
    tier[c] = float("nan")
tier.to_csv(out, index=False)
print(f"{out}: {len(tier)} sessions, {tier.subject_id.nunique()} subjects, "
      f"{tier.supervised_label.value_counts().to_dict()}")
