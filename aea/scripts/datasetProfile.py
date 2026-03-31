from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport

output_dir = Path("aea/reports")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("dataset/311_Service_Requests_20260329.csv")

profile = ProfileReport(df, title="Calgary Dataset Analysis")
profile.to_file(str(output_dir / "report.html"))