# This script is used to preprocess the dataset in order to convert it to the JSON format that the model expects.
import os
import json

# def load_labels_vocab(json_file_path):
#     with open(json_file_path, 'r', encoding='utf-8') as file:
#         labels_vocab = json.load(file)
#     return labels_vocab

def process_file(file_path, labels2int, int2labels):
    sentences_data = {"sentences": []}
    with open(file_path, 'r', encoding='utf-8') as file:
        current_text = []
        current_labels = []
        for line in file:
            if line.strip() == "":  
                if current_text:  
                    sentences_data["sentences"].append({
                        "tokens": current_text,
                        "ner_tags": current_labels
                    })
                    current_text = []
                    current_labels = []
            else:
                parts = line.split()
                token, tag = parts[0], parts[-1]  
                current_text.append(token)
                if tag in labels2int:
                    current_labels.append(labels2int[tag])  
                else:
                    current_labels.append(labels2int["O"])

    return sentences_data

def save_processed_data(sentences_data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(sentences_data, file, ensure_ascii=False, indent=2)

def preprocessing_dataset(data_path, output_path, labels2int, int2labels):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_path, filename)
            sentences_data = process_file(file_path, labels2int, int2labels)
            output_file_path = os.path.join(output_path, os.path.splitext(filename)[0] + ".json")
            save_processed_data(sentences_data, output_file_path)

# # Paths
# dataset_dir = "Creation-of-a-synthetic-dataset-for-French-NER-in-clinical-trial-texts\data\chia_bio\chia_bio"  
# output_dir = "Creation-of-a-synthetic-dataset-for-French-NER-in-clinical-trial-texts\NER-chia-dataset"  

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# for filename in os.listdir(dataset_dir):
#     if filename.endswith(".txt"):
#         file_path = os.path.join(dataset_dir, filename)
#         sentences_data = process_file(file_path)
#         output_file_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".json")
#         save_processed_data(sentences_data, output_file_path)