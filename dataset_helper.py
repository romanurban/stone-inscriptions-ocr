import os
import json

def get_true_text(json_item, image_path):
    """
    Extracts the true text from a JSON item based on the dataset type inferred from the image path.
    
    :param json_item: The JSON item containing the details of the image.
    :param image_path: The path of the image being processed, used to determine the dataset type.
    :return: The true text extracted from the JSON item.
    """
    true_text = ""
    if 'berlin-mitte/' in image_path:
        true_text = json_item.get('description', '')
    elif 'timenote/' in image_path:
        # Concatenate the fields to form the true text for the 'timenote' dataset
        true_text = ' '.join(json_item.get(field, '') for field in ['person_name', 'extra_names', 'patronymic']).strip()
    return true_text

def get_json_details(image_path, directory):
    """
    Searches for and extracts details from a JSON file corresponding to the given image file.
    :param image_path: Path to the image file.
    :param directory: Directory to search for the JSON file.
    :return: Details from the JSON file or None if not found.
    """
    # Determine dataset type based on the presence of specific subdirectories in the image_path
    if 'timenote/' in image_path:
        # For 'timenote/' dataset, replace '_' with '/' and adjust extension to match JSON "main_image_url"
        image_file_name = os.path.basename(image_path)
        parts = image_file_name.split('_')
        if len(parts) > 2:
            # Only proceed if there are at least two underscores to replace
            formatted_image_name = '/'.join(parts[:2]) + '_' + '_'.join(parts[2:])
        else:
            # If less than two underscores, no replacement is needed
            formatted_image_name = image_file_name
        # Adjust extension to match JSON "main_image_url"
        formatted_image_name = formatted_image_name.rsplit('.', 1)[0] + '.jpg'
        image_url_key = "main_image_url"
    elif 'berlin-mitte/' in image_path:
        # For 'berlin-mitte/' dataset, use the image filename as-is to match JSON "imageURL"
        formatted_image_name = os.path.basename(image_path)
        image_url_key = "imageURL"
    else:
        # If the dataset does not match known structures, return None
        return None

    for file in os.listdir(directory):
        if file.endswith(".json"):
            json_path = os.path.join(directory, file)
            with open(json_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    # Extract filename from the URL in JSON data based on the dataset type
                    json_image_file_name = os.path.basename(item.get(image_url_key, ""))
                    if formatted_image_name.endswith(json_image_file_name):
                        return item
    return None

def extract_lang(item):
    if 'nationality' in item:
        map = {
            "latvian": "lav",
            "russian": "rus",
            "pole": "pol",
            "german": "deu",
        }
        return map.get(item['nationality'].lower(), "lav")
    else:
        return "lav"