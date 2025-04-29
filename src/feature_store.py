import redis 
import json 

class RedisFeatureStore:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.StrictRedis(host=host, port=port, db=db, decode_responses=True)

    #stores features in a single entity
    def store_features(self, entity_id, features):
        key = f"entity:{entity_id}:features"
        self.redis_client.set(key, json.dumps(features))

    def get_features(self, entity_id):
        key = f"entity:{entity_id}:features"
        features = self.redis_client.get(key)
        if features:
            return json.loads(features)
        else:
            return None
        
    def store_batch_features(self, batch_data):
        for entity_id, features in batch_data.items():
            self.store_features(entity_id, features)

    def get_batch_features(self,entity_ids):
        batch_features ={}
        for entity_id in entity_ids:
            batch_features[entity_id] = self.get_features(entity_id)
        return batch_features 
    
    def get_all_entity_ids(self):
        keys = self.redis_client.keys("entity:*:features")
        entity_ids = [key.split(":")[1] for key in keys]
        return entity_ids
    



