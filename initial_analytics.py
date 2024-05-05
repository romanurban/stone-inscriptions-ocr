import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Function to read JSON files into a pandas DataFrame
def read_jsons_to_dataframe(directory):
    df = pd.DataFrame()

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        # Read the entire file content
                        file_content = file.read()
                        # Remove any trailing commas before a closing bracket
                        file_content = file_content.rstrip(',\n').rstrip(',')
                        # Correctly format the content to be a list of JSON objects
                        formatted_content = "[" + file_content.replace("}\n{", "},\n{") + "]"
                        # Attempt to parse the formatted content as JSON
                        data = json.loads(formatted_content)
                        df_file = pd.json_normalize(data)
                        df = pd.concat([df, df_file], ignore_index=True)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")
                    print("Faulty file content (first 500 chars):", file_content[:500])
                    continue
                except Exception as e:
                    print(f"Unhandled error with file {file_path}: {e}")

    return df

# Function to convert column types for specific score-related columns
def convert_column_types(df):
    float_columns = [
        'composite_score',
        'scores.basic_similarity_score',
        'scores.lcs_similarity_score',
        'scores.jaro_winkler_similarity',
        'scores.difflib_similarity'
    ]
    for column in float_columns:
        df[column] = df[column].astype(float)
    return df

def extract_features(df):
    dataset_type_pattern = '(timenote|berlin-mitte)'

    # Extract the dataset type
    df['dataset_type'] = df['file_id'].str.extract(dataset_type_pattern, expand=False)

    return df

df_init = read_jsons_to_dataframe('ocr_results/revision_INITIAL_complete/')
df_init = convert_column_types(df_init)
df_init = extract_features(df_init)
print("Initial dataframe shape: ", df_init.shape)

# Optional: print column types and check for any issues
# print(df_init.dtypes)

df = df_init.copy()

# check and convert data type of 'composite_score' to float if not already
if df['composite_score'].dtype != 'float':
    df['composite_score'] = pd.to_numeric(df['composite_score'], errors='coerce')

# check for NaN values and handle them
if df['composite_score'].isna().any():
    print("NaN values found in 'composite_score'. Filling with median...")
    median_value = df['composite_score'].median()
    df['composite_score'].fillna(median_value, inplace=True)

print("DataFrame shape: ", df.shape)
print(df.describe())

# analyzing the effectiveness of each OCR method
grouped = df.groupby(['ocr_method', 'dataset_type']).agg({
    'composite_score': ['mean', 'std', 'min', 'max', 'count']
})
print("Grouped statistics by OCR Method and Dataset Type:")
print(grouped)

# plotting a Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x='ocr_method', y='composite_score', hue='dataset_type', data=df)
plt.title('Composite Score Distribution by OCR Method and Dataset Type')
plt.xlabel('OCR Method')
plt.ylabel('Composite Score')
plt.legend(title='Dataset Type')
plt.show()

# plotting a Violin Plot
plt.figure(figsize=(12, 6))
sns.violinplot(x='ocr_method', y='composite_score', hue='dataset_type', data=df, split=True)
plt.title('Composite Score Distribution by OCR Method and Dataset Type (Violin Plot)')
plt.xlabel('OCR Method')
plt.ylabel('Composite Score')
plt.legend(title='Dataset Type')
plt.show()
