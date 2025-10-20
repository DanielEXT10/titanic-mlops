from src.logger import get_logger
from src.custom_exception import CustomException
import pandas as pd
from src.feature_store import RedisFeatureStore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import os
import pickle

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self, feature_store: RedisFeatureStore, model_save_path = "artifacts/models/"):
        self.feature_store = feature_store
        self.model_save_path = model_save_path
        self.model =None

        os.makedirs(self.model_save_path, exist_ok=True)
        logger.info("ModelTrainer initialized.")

    def load_data_from_redis(self,entity_ids):
        try:
            logger.info("Loading data from Redis Feature Store.")
            data =[]
            for entity_id in entity_ids:
                features = self.feature_store.get_features(entity_id)
                if features:
                    data.append(features)
                else:
                    logger.warning(f"No features found for entity ID: {entity_id}")
            return data
        except Exception as e:
            logger.error("Error loading data from Redis.")
            raise CustomException(e)
    
    def prepare_data(self):
        try:
            entity_ids = self.feature_store.get_all_entity_ids()
            train_entity_ids, test_entity_ids = train_test_split(entity_ids, test_size=0.2, random_state=42)
            
            train_data = self.load_data_from_redis(train_entity_ids)
            test_data = self.load_data_from_redis(test_entity_ids)

            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)

            X_train = train_df.drop("Survived", axis=1)
            y_train = train_df["Survived"]
            X_test = test_df.drop("Survived", axis=1)
            y_test = test_df["Survived"]
            
            features_used = X_train.columns.tolist()
            logger.info(f"Features used for training ({len(features_used)}): {features_used}")
            
            logger.info("Data preparation completed.")

            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logger.error("Error preparing data.")
            raise CustomException(e)
        
    def hyperparameter_tuning(self, x_train, y_train):

        try:
            param_distributions = {
                'n_estimators': [50, 100, 200, 300], #Number of trees in the forest
                'max_depth': [None, 10, 20, 30], #Maximum depth of the tree
                'min_samples_split': [2, 5, 10], #Minimum number of samples required to split an internal node
                'min_samples_leaf': [1, 2, 4] #Minimum number of samples required to be at a leaf node
                }

            rf = RandomForestClassifier(random_state=42)
            random_search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_distributions,
                n_iter=10,
                cv=3,
                scoring='accuracy',
                verbose=2,
                random_state=42,
                n_jobs=-1
            )
            random_search.fit(x_train, y_train)
            logger.info("Hyperparameter tuning completed.")
            return random_search.best_estimator_
        except Exception as e:
            logger.error("Error during hyperparameter tuning.")
            raise CustomException(e)
    
    def train_and_evaluate(self, x_train, x_test, y_train, y_test):
        try:
            best_rf = self.hyperparameter_tuning(x_train, y_train)
            y_pred = best_rf.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model training and evaluation completed with accuracy: {accuracy}")

            self.save_model(best_rf)

            return accuracy
        except Exception as e:
            logger.error("Error during model training and evaluation.")
            raise CustomException(e)
    
    def save_model(self, model):
        try:
            model_filename = f"{self.model_save_path}/random_forest_model.pkl"

            with open(model_filename, 'wb') as model_file:
                pickle.dump(model, model_file)
            
            logger.info(f"Model saved at {model_filename}.")
        except Exception as e:
            logger.error("Error saving the model.")
            raise CustomException(e)
        
    def run(self):
        try:
            logger.info("Model training process started.")
            X_train, X_test, y_train, y_test = self.prepare_data()
            accuracy = self.train_and_evaluate(X_train, X_test, y_train, y_test)

            logger.info("Model training process completed.")
        
        except Exception as e:
            logger.error("Error in the model training process.")
            raise CustomException(e)
        

if __name__ == "__main__":
    feature_store = RedisFeatureStore()
    model_trainer = ModelTrainer(feature_store=feature_store)
    model_trainer.run()





