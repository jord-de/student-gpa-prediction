import pandas as pd

# Read Excel file
df = pd.read_excel(
    "Dataset/Original_dataset.xlsx"
)

# Convert to CSV
df.to_csv(
    "Dataset/student_social_media.csv",
    index=False
)

print("Excel successfully converted to CSV!")