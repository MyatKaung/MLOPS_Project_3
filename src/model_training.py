from src.logger import get_logger
from src.custom_exception import CustomException
import pandas as pd 
from src.feature_store import RedisFeatureStore
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import os  
from sklearn.metrics import accuracy_score 
logger = get_logger(__name__)   

class ModelTraining:
    def __init__(self, feature_store: RedisFeatureStore, model_save_path = "artifacts/models/"):
        self.feature_store = feature_store
        self.model_save_path = model_save_path
        self.model = None
        os.makedirs(self.model_save_path, exist_ok=True)    
        logger.info("ModelTraining class initialized")

    def load_data_from_redis(self, entity_ids):
        try:
            batch_features = self.feature_store.get_batch_features(entity_ids)
            if not batch_features:
                raise CustomException("No features found in Redis for the given entity IDs.")
            data = pd.DataFrame.from_dict(batch_features, orient='index')
            logger.info("Data loaded from Redis successfully")
            return data
        except Exception as e:
            logger.error(f"Error loading data from Redis: {e}")
            raise CustomException(f"Error loading data from Redis: {e}")
        
    def prepare_data(self):
        try:
            entity_ids = self.feature_store.get_all_entity_ids()
            data = self.load_data_from_redis(entity_ids)
            X = data.drop(columns=['Survived'])
            y = data['Survived']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logger.info("Data prepared successfully")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise CustomException(f"Error preparing data: {e}")
        
    def hyperparameter_tuning(self, X_train, y_train):
        try:
            hyperparameters ={
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],

            }

            rf = RandomForestClassifier(random_state=42)
            random_search = RandomizedSearchCV(
                rf, hyperparameters, n_iter=10, cv=3, scoring='accuracy',random_state=42
            )
            random_search.fit(X_train, y_train)
            logger.info(f"Best hyperparameters: {random_search.best_params_}")
            return random_search.best_estimator_
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {e}")
            raise CustomException(f"Error in hyperparameter tuning: {e}")
        
    def train_and_evaluate(self , X_train , y_train , X_test , y_test):
        try:
            best_rf = self.hyperparameter_tuning(X_train,y_train)

            y_pred = best_rf.predict(X_test)

            accuracy = accuracy_score(y_test,y_pred)

            logger.info(f"Accuracy is {accuracy}")


            return best_rf, accuracy
        
        except Exception as e:
            logger.error(f"Error while model training {e}")
            raise CustomException(str(e))
    
    def save_model(self , model):
        try:
            model_filename = f"{self.model_save_path}random_forest_model.pkl"

            with open(model_filename,'wb') as model_file:
                pickle.dump(model , model_file)

            logger.info(f"Model saved at {model_filename}")
            return model_filename 
        except Exception as e:
            logger.error(f"Error while model saving {e}")
            raise CustomException(str(e))
        
    def run(self):
        try:
            logger.info("Starting Model Training Pipleine....")
            X_train , X_test , y_train, y_test = self.prepare_data()
            model, accuracy = self.train_and_evaluate(X_train , y_train, X_test , y_test)
            model_path = self.save_model(model)
            logger.info(f"Model saved at: {model_path}")

            logger.info("End of Model Training pipeline...")
            logger.info(f"Model accuracy: {accuracy}")  

        except Exception as e:
            logger.error(f"Error while model training pipeline {e}")
            raise CustomException(str(e))
        
if __name__ == "__main__":
    feature_store = RedisFeatureStore()
    model_trainer = ModelTraining(feature_store)
    model_trainer.run()



        
    

