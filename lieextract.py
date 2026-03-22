import pandas as pd

# File paths
lie_file = r"C:\Users\KIIT0001\Downloads\newdataset\lie_features.xlsx"
truth_file = r"C:\Users\KIIT0001\Downloads\newdataset\truth_features.xlsx"

# Read both files
lie_df = pd.read_excel(lie_file)
truth_df = pd.read_excel(truth_file)

# Add label column
lie_df["Label"] = "Lie"
truth_df["Label"] = "Truth"

# Merge datasets
combined_df = pd.concat([lie_df, truth_df], ignore_index=True)

# Save new file
output_file = r"C:\Users\KIIT0001\Downloads\newdataset\updated.xlsx"
combined_df.to_excel(output_file, index=False)

print("Files merged successfully!")
print("Saved at:", output_file)