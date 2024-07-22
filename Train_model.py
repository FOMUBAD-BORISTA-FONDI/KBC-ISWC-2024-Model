import json
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from keras.models import Model
from keras.layers import Input, Dense, concatenate
from keras.optimizers import Adam
import joblib

# Fonction pour charger les données JSONL
def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

# Chargement des données d'entraînement
train_data = load_jsonl('data/train.jsonl')

# Initialisation du dictionnaire
object_entity_to_id = {}
id_to_object_entity = {}
current_id = 0

# Création des dictionnaires object_entity_to_id et id_to_object_entity
for item in train_data:
    for obj in item['ObjectEntities']:
        if obj not in object_entity_to_id:
            object_entity_to_id[obj] = current_id
            id_to_object_entity[current_id] = obj
            current_id += 1

# Sauvegarde des dictionnaires
joblib.dump(object_entity_to_id, 'model/object_entity_to_id.joblib')
joblib.dump(id_to_object_entity, 'model/id_to_object_entity.joblib')

# Configuration du tokenizer BERT Large Cased
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
bert_model = BertModel.from_pretrained('bert-large-cased')

# Fonction pour obtenir les embeddings BERT pour un texte donné
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Prétraitement des données pour obtenir les embeddings BERT
def encode_data_with_bert(data):
    encoded_data = []
    for item in data:
        encoded_subject = get_bert_embeddings(item['SubjectEntity'])
        encoded_relation = get_bert_embeddings(item['Relation'])
        encoded_object_ids = [object_entity_to_id[obj] for obj in item['ObjectEntities']]
        encoded_data.append((encoded_subject, encoded_relation, encoded_object_ids))
    return encoded_data

train_encoded = encode_data_with_bert(train_data)

# Préparation des données pour l'entraînement
X_subjects = np.array([item[0] for item in train_encoded])
X_relations = np.array([item[1] for item in train_encoded])

# Création des étiquettes y
y = np.zeros((len(train_encoded), len(object_entity_to_id)))

for i, item in enumerate(train_encoded):
    for obj_id in item[2]:
        y[i][obj_id] = 1

# Reshape les entrées pour être de la forme attendue
X_subjects = X_subjects.reshape(X_subjects.shape[0], -1)
X_relations = X_relations.reshape(X_relations.shape[0], -1)

# Définition du modèle
input_subject = Input(shape=(X_subjects.shape[1],))
input_relation = Input(shape=(X_relations.shape[1],))

# Use concatenate directly from keras.layers
merged = concatenate([input_subject, input_relation])
dense1 = Dense(1024, activation='relu')(merged)
output = Dense(len(object_entity_to_id), activation='sigmoid')(dense1)

model = Model(inputs=[input_subject, input_relation], outputs=output)

# Compilation du modèle
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
model.fit([X_subjects, X_relations], y, epochs=3, batch_size=8, validation_split=0.1)

# Sauvegarde du modèle
model.save('model/relation_model.h5')

# Sauvegarde du tokenizer et du modèle BERT
joblib.dump(tokenizer, 'model/tokenizer.joblib')
joblib.dump(bert_model, 'model/bert_model.joblib')
