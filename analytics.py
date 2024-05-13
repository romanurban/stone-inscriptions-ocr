import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Function to read JSON files into a pandas DataFrame
def read_jsons_to_dataframe(directory):
    df = pd.DataFrame()

    for root, dirs, files in os.walk(directory):
        # ignore test subdirectory
        if 'test' in root:
            continue

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

def extract_features_preproc(df):
    preprocessed_pattern = '(_processed.png|_processed_color_segmentation.png|_processed_edge_detection.png)'
    dataset_type_pattern = '(timenote|berlin-mitte)'

    # Extract the postprocessing method
    df['preprocessed_method'] = df['file_id'].str.extract(preprocessed_pattern, expand=False)
    df.loc[df['preprocessed_method'] == '_processed_edge_detection.png', 'preprocessed_method'] = 'edge_detection'
    df.loc[df['preprocessed_method'] == '_processed_color_segmentation.png', 'preprocessed_method'] = 'color_segmentation'
    df.loc[df['preprocessed_method'] == '_processed.png', 'preprocessed_method'] = 'default'

    # Extract the dataset type
    df['dataset_type'] = df['file_id'].str.extract(dataset_type_pattern, expand=False)

    return df

def extract_orig_file_id(df):
    # Define the postfixes to remove
    postfixes = [
        '_processed.png',
        '_processed_color_segmentation.png',
        '_processed_edge_detection.png'
    ]
    
    # Create a new column 'file_key' by replacing postfixes in 'file_id'
    df['file_key'] = df['file_id']
    for postfix in postfixes:
        df['file_key'] = df['file_key'].str.replace(postfix, '', regex=False).str.replace('dataset_preprocessed', 'dataset', regex=False)
    
    return df

def plot_preprocessing_effects(df, ocr_methods):
    # Set the style of seaborn for better visuals
    sns.set(style="whitegrid")

    # Loop through each OCR method and create a separate plot
    for ocr_method in ocr_methods:
        # Filter data for the current OCR method
        data = df[df['ocr_method'] == ocr_method]
        
        plt.figure(figsize=(10, 6))
        # Create a boxplot
        sns.boxplot(x='preprocessed_method', y='composite_score', hue='dataset_type', data=data, palette='Set2')
        plt.title(f'Precizitātes novērtējuma sadalījums priekš OCR metodes {ocr_method}')
        plt.xlabel('Priekšapstrādes metodes')
        plt.ylabel('Precizitātes novertejums', labelpad=20)
        plt.legend(title='Datu kopas veids', bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        plt.show()

df_init = read_jsons_to_dataframe('ocr_results/revision_INITIAL_complete/')
df_init = convert_column_types(df_init)
df_init = extract_features(df_init)
# print("Initial dataframe shape: ", df_init.shape)

# check and convert data type of 'composite_score' to float if not already
if df_init['composite_score'].dtype != 'float':
    df_init['composite_score'] = pd.to_numeric(df_init['composite_score'], errors='coerce')

# check for NaN values and handle them
if df_init['composite_score'].isna().any():
    print("NaN values found in 'composite_score'. Filling with median...")
    median_value = df_init['composite_score'].median()
    df_init['composite_score'].fillna(median_value, inplace=True)

# print("DataFrame shape: ", df_init.shape)
# print(df_init.describe())

# analyzing the effectiveness of each OCR method
grouped = df_init.groupby(['ocr_method', 'dataset_type']).agg({
    'composite_score': ['mean', 'std', 'min', 'max', 'count']
})

df_preprocessed = read_jsons_to_dataframe('ocr_results/revision_PREPROCESSED_complete/')
df_preprocessed = convert_column_types(df_preprocessed)
df_preprocessed = extract_features_preproc(df_preprocessed)
df_preprocessed = extract_orig_file_id(df_preprocessed)

# detecting best performeing preprocessing method
method_priority = {'color_segmentation': 1, 'edge_detection': 2, 'default': 3}
df_preprocessed['method_priority'] = df_preprocessed['preprocessed_method'].map(method_priority)
df_sorted = df_preprocessed.sort_values(by=['file_key', 'ocr_method', 'composite_score', 'method_priority'], ascending=[True, True, False, True])

# select the index of the first entry after sorting
idx = df_sorted.groupby(['file_key', 'ocr_method']).head(1).index

# use  indices to find the best performing preprocessed_method for each group
best_preprocessing = df_sorted.loc[idx].reset_index(drop=True)

# Drop the auxiliary column
best_preprocessing = best_preprocessing.drop(columns=['method_priority'])

print("Overal value stats on dominating -", best_preprocessing['preprocessed_method'].value_counts())

# for the init_df we need to ad preprocessed_method - none
df_init['preprocessed_method'] = 'none'

# for df_init add 'file_key' column that is the same as 'file_id' but without the file extension
df_init['file_key'] = df_init['file_id'].str.replace(r'\.[^.]+$','', regex=True) # drop the extension

# assuming the same structure of both datasets
df_combined = df_init._append(best_preprocessing, ignore_index=True)

def filter_groups(group):
    return group['composite_score'].max() > 0.05

print("Before unrecognizable cleaned: ", df_combined.shape)
df_combined = df_combined.groupby(['file_key']).filter(filter_groups)
print("After unrecognizable cleaned: ", df_combined.shape)

# creating baseline score column for further comparison
df_combined = df_combined.merge(df_init[['file_key', 'ocr_method', 'composite_score']],
                                on=['file_key', 'ocr_method'],
                                suffixes=('', '_baseline'))

def exclude_worsened_groups(group):
    # Condition 1: No score should be less than the baseline
    # Condition 2: At least one score should be greater than the baseline (indicating improvement)
    has_worsened = (group['composite_score'] < group['composite_score_baseline']).any()
    has_improved = (group['composite_score'] > group['composite_score_baseline']).any()
    return not has_worsened and has_improved

df_combined = df_combined.groupby(['file_key']).filter(exclude_worsened_groups)
print("After worsened preprocessing ignored: ", df_combined.shape)

grouped_combined = df_combined.groupby(['ocr_method', 'dataset_type', 'preprocessed_method']).agg({
    'composite_score': ['mean', 'std', 'min', 'max', 'count']
})

print(grouped_combined)

#build boxplots
ocr_methods = df_combined['ocr_method'].unique()
plot_preprocessing_effects(df_combined, ocr_methods)

df_combined['score_diff'] = df_combined['composite_score'] - df_combined['composite_score_baseline']
# group by file_key and ocr_method and get max score_diff
df_max_diff = df_combined.groupby(['file_key', 'ocr_method']).agg({
    'score_diff': 'max'
}).reset_index()

# plot the max differences average by ocr_method using a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='ocr_method', y='score_diff', data=df_max_diff, palette='Set3')
plt.title('Vidējais precizitātes uzlabojums')
plt.xlabel('OCR Metode')
plt.ylabel('Uzlabojuma vidējais novērtējums')
plt.show()

df_max_diff = df_combined.groupby(['file_key', 'preprocessed_method']).agg({
    'score_diff': 'max'
}).reset_index()

# exclude none method
df_max_diff = df_max_diff[df_max_diff['preprocessed_method'] != 'none']

# plot the max differences average by ocr_method using a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='preprocessed_method', y='score_diff', data=df_max_diff, palette='Set1')
plt.title('Vidējais precizitātes uzlabojums')
plt.xlabel('Priekšapstrādes metode')
plt.ylabel('Uzlabojuma vidējais novērtējums')
plt.show()

# Filter for rows where 'composite_score' is above 0.05
df_filtered = df_combined[df_combined['composite_score'] > 0.05]

# Count the number of images recognized initially by each OCR method
initial_counts = df_filtered[df_filtered['preprocessed_method'] == 'none'].groupby('ocr_method')['file_key'].count()

# Count the number of images recognized initially by each OCR method
preprocessed_counts = df_filtered[df_filtered['preprocessed_method'] != 'none'].groupby('ocr_method')['file_key'].count()

# Combine the counts into a single DataFrame
counts = pd.DataFrame({'Initial': initial_counts, 'Preprocessed': preprocessed_counts})
counts['Preprocessed'] = counts['Preprocessed'] - counts['Initial']

# Plot the counts
colors = ['#254d70', '#b2d942']
counts.plot(kind='bar', stacked=True, color=colors, figsize=(10, 6))
plt.title('Atpazītu attēlu skaits pēc OCR metodes')
plt.xlabel('')
plt.ylabel('Attēlu skaits')
plt.legend(['Sakotnēji', 'Ar priekšapstrādi'])
plt.xticks(rotation=0)
plt.show()