import json
import joblib

# Fonction pour charger les données JSONL
def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

# Charger les fichiers d'entraînement et de validation
train_data = load_jsonl('data/train.jsonl')
validation_data = load_jsonl('data/val.jsonl')

# Charger le fichier de prédictions
predictions_data = load_jsonl('first-predictions.jsonl')

# Construire les dictionnaires pour mapper les ObjectEntities aux ObjectEntitiesID
entity_to_id = {}
for item in train_data + validation_data:
    for obj, obj_id in zip(item['ObjectEntities'], item['ObjectEntitiesID']):
        if obj not in entity_to_id:
            entity_to_id[obj] = obj_id

# Fonction pour remplacer les ObjectEntitiesID dans les prédictions
def replace_object_entity_ids(predictions, entity_to_id):
    for item in predictions:
        updated_ids = []
        for obj in item['ObjectEntities']:
            if obj in entity_to_id:
                updated_ids.append(entity_to_id[obj])
            else:
                updated_ids.append(obj)  # Conserver l'objet si l'ID n'est pas trouvé
        item['ObjectEntitiesID'] = updated_ids
    return predictions

# Remplacer les ObjectEntitiesID dans les prédictions
updated_predictions = replace_object_entity_ids(predictions_data, entity_to_id)

# Sauvegarder les prédictions mises à jour dans un fichier JSONL
output_file_path = 'predictions.jsonl'
with open(output_file_path, 'w') as file:
    for item in updated_predictions:
        file.write(json.dumps(item) + '\n')

print(f"Les prédictions mises à jour ont été sauvegardées dans le fichier {output_file_path}")

