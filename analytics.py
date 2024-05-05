import pandas as pd
import os
import json

def extract_features(df):
    preprocessed_pattern = '(_processed.png|_processed_color_segmentation.png|_processed_edge_detection.png)'
    dataset_type_pattern = '(timenote|berlin-mitte)'

    # Extract the postprocessing method
    df['preprocessed_method'] = df['file_id_preprocessed'].str.extract(preprocessed_pattern, expand=False)

    # Extract the dataset type
    df['dataset_type'] = df['file_id_init'].str.extract(dataset_type_pattern, expand=False)

    return df

def get_unrecognized_file_keys(df):
    # create column that indicates that both init and preprocessed runs did not recognize anything
    df['nothing_recognized'] = (df['composite_score_init'] == 0.0) & (df['composite_score_preprocessed'] == 0.0)
   
    print(df.info())
    # those where for all filey_key entries nothing was recognized
    unrecognized = df.groupby('file_key')['nothing_recognized'].all()

    return pd.DataFrame(unrecognized[unrecognized].index, columns=['file_key'])

def convert_column_types(df):
    # List of columns to convert to float
    float_columns = [
        'composite_score',
        'scores.basic_similarity_score',
        'scores.lcs_similarity_score',
        'scores.jaro_winkler_similarity',
        'scores.difflib_similarity'
    ]
    
    # Convert each column to float
    for column in float_columns:
        df[column] = df[column].astype(float)
    
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

df_init = read_jsons_to_dataframe('ocr_results/revision_INITIAL_complete/')
print("Initial dataframe shape: ", df_init.shape)
print(df_init.head())
print(df_init.info())

df_preprocessed = read_jsons_to_dataframe('ocr_results/revision_PREPROCESSED_complete/')
print("Preprocessed dataframe shape: ", df_preprocessed.shape)
print(df_preprocessed.head())
print(df_preprocessed.info())

print("Converting column types...")
df_init = convert_column_types(df_init)
df_preprocessed = convert_column_types(df_preprocessed)
print(df_init.info())
print(df_preprocessed.info())

print("Create file key to merge data frames")
df_init['file_key'] = df_init['file_id'].str.replace(r'\.[^.]+$','', regex=True) # drop the extension
print(df_init[['file_id', 'file_key']].head(10))

print("Extract original file key from preprocessed data frame")
df_preprocessed = extract_orig_file_id(df_preprocessed)
# pd.set_option('display.max_colwidth', None)
print(df_preprocessed[['file_id', 'file_key']].head(10))

# print("Key value counts:")
# print("INIT: ", df_init['file_key'].value_counts())
# print("PREPROCESSED: ", df_preprocessed['file_key'].value_counts())

df_combined = pd.merge(df_init, df_preprocessed, on=['file_key', 'ocr_method'], how='inner', suffixes=('_init', '_preprocessed'))
print("Combined dataframe shape: ", df_combined.shape)
print(df_combined.info())

# print all columns to see what we have for file_key dataset/timenote/Tornakalna_kapi/2016_05_Numa-Palmgrens
print(df_combined[df_combined['file_key'] == 'dataset/timenote/Tornakalna_kapi/2016_05_Numa-Palmgrens'])

unrecognized = get_unrecognized_file_keys(df_combined)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
print(unrecognized)

# filter out those, that perfectly recognized from the initial run
# TODO count how many of these are there, mention in report, which method achieved that comparing to others
df_combined = df_combined[df_combined['composite_score_init'] != 1.0]

# filter out those, that were not recognized by any method
df_combined = df_combined[~df_combined['file_key'].isin(unrecognized['file_key'])]

# calculate the difference between the composite scores
df_combined['score_improvement'] = df_combined['composite_score_preprocessed'] - df_combined['composite_score_init']

# extract more features
df_combined = extract_features(df_combined)

print(df_combined.info())

df_sorted = df_combined.sort_values(by='score_improvement', ascending=False)

top_100 = df_sorted.head(4000)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
#print(top_100[['file_key', 'ocr_method', 'score_improvement','composite_score_preprocessed']])

# TODO separately analyze method comparison performance
# and main dataset recognition stats like how many recognized, how many not, how many were improved

# how many images had OCR improvement
print("Improved OCR: ", df_combined[df_combined['score_improvement'] > 0].shape[0])
print("Total images: ", df_combined.shape[0])

