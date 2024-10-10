import pandas as pd
# Check unique labels in the CSV file
csv_file = './data/HAM10000_metadata.csv'
metadata = pd.read_csv(csv_file)
print(metadata['dx'].unique())  # This will print the unique labels in your dataset
