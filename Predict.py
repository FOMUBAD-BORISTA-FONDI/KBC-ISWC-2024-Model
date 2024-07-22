import json
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from keras.models import load_model
import joblib

# Fonction pour charger les données JSONL
def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

# Chargement des données de test
test_data = load_jsonl('data/test.jsonl')

# Chargement du modèle et des objets nécessaires
model_nn = load_model('model/relation_model.h5')
tokenizer = joblib.load('model/tokenizer.joblib')
bert_model = joblib.load('model/bert_model.joblib')
object_entity_to_id = joblib.load('model/object_entity_to_id.joblib')
id_to_object_entity = joblib.load('model/id_to_object_entity.joblib')

# Fonction pour obtenir les embeddings BERT pour un texte donné
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Prétraitement des données pour utiliser les embeddings BERT
def encode_data_with_bert(data):
    encoded_data = []
    for item in data:
        encoded_subject = get_bert_embeddings(item['SubjectEntity'])
        encoded_relation = get_bert_embeddings(item['Relation'])
        encoded_data.append((encoded_subject, encoded_relation))
    return encoded_data

test_encoded = encode_data_with_bert(test_data)

# Préparation des données pour les prédictions
X_subjects_test = np.array([item[0] for item in test_encoded])
X_relations_test = np.array([item[1] for item in test_encoded])

# Reshape les entrées pour être de la forme attendue
X_subjects_test = X_subjects_test.reshape(X_subjects_test.shape[0], -1)
X_relations_test = X_relations_test.reshape(X_relations_test.shape[0], -1)

# Prédiction sur les données de test
y_test_pred = model_nn.predict([X_subjects_test, X_relations_test])

# Filtrage des prédictions non pertinentes
def filter_predictions(predictions, threshold=0.5):
    filtered_preds = []
    for pred in predictions:
        filtered_pred = [id_to_object_entity[idx] for idx, score in enumerate(pred) if score >= threshold]
        filtered_preds.append(filtered_pred)
    return filtered_preds

decoded_test_preds = filter_predictions(y_test_pred)

# Sauvegarde des prédictions dans un fichier JSONL
output_file_path = 'first-predictions.jsonl'
with open(output_file_path, 'w') as file:
    for i, item in enumerate(test_data):
        result = {
            "SubjectEntity": item['SubjectEntity'],
            "SubjectEntityID": item['SubjectEntityID'],
            "ObjectEntities": decoded_test_preds[i],
            "ObjectEntitiesID": [object_entity_to_id[entity] for entity in decoded_test_preds[i]],
            "Relation": item['Relation']
        }
        file.write(json.dumps(result) + '\n')

print(f"Prédictions sauvegardées dans le fichier {output_file_path}")

