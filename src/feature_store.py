import redis
import json

class RedisFeatureStore:
    def __init__(self, host="localhost", port=6379, db =0):
        self.client = redis.StrictRedis(
            host=host,
            port=port,
            db=db,
            decode_responses=True)
   
    #Storing row by row features
    def store_features(self,entity_id, features):
        """
        Store features for a given entity.

        :param entity_id: Unique identifier for the entity.
        :param features: Dictionary of feature names and their values.
        """
        key = f"entity:{entity_id}:features"
        self.client.set(key, json.dumps(features))
    
    #Getting row one by one
    def get_features(self, entity_id):
        """
        Retrieve features for a given entity.

        :param entity_id: Unique identifier for the entity.
        :return: Dictionary of feature names and their values or None if not found.
        """
        key = f"entity:{entity_id}:features"
        features = self.client.get(key)
        return json.loads(features) if features else None
    
    
    def store_batch_features(self, batch_data):
        """
        Store features for multiple entities in a batch.

        :param batch_data: Dictionary where keys are entity IDs and values are dictionaries of features.
        """
        for entity_id, features in batch_data.items():
            self.store_features(entity_id, features)
    
    def get_batch_features(self, entity_ids):
        """
        Retrieve features for multiple entities in a batch.

        :param entity_ids: List of unique identifiers for the entities.
        :return: Dictionary where keys are entity IDs and values are dictionaries of features or None if not found.
        """
        batch_features = {}
        for entity_id in entity_ids:
            batch_features[entity_id] = self.get_features(entity_id)
        return batch_features
        
    def get_all_entity_ids(self):
        """
        Retrieve all entity IDs stored in the feature store.

        :return: List of entity IDs.
        """
        keys = self.client.keys("entity:*:features")
        entity_ids = [key.split(":")[1] for key in keys ]
        return entity_ids

