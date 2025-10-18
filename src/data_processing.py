import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.feature_store import RedisFeatureStore
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import TRAIN_PATH, TEST_PATH
logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, train_data_path, test_data_path,feature_store: RedisFeatureStore):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.X_resampled = None
        self.y_resampled = None

        self.feature_store = feature_store
        logger.info("DataProcessor initialized.")

    def load_data(self):
        try:
            self.data = pd.read_csv(self.train_data_path)
            self.test_data = pd.read_csv(self.test_data_path)
            logger.info("Data loaded successfully.")
        except Exception as e:
            logger.error("Error loading data.")
            raise CustomException(e)
    
    def process_data(self):
        try:
            self.data['Age'] = self.data['Age'].fillna(self.data['Age'].median())
            self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode()[0])
            self.data['Fare'] = self.data['Fare'].fillna(self.data['Fare'].median())
            self.data['Sex'] = self.data['Sex'].map({'male': 0, 'female': 1}).astype(int)
            self.data['Embarked'] = self.data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
            
            self.data['FamilySize'] = self.data['SibSp'] + self.data['Parch'] + 1
            self.data['HasCabin'] = self.data['Cabin'].notnull().astype(int)
            self.data['Isalone'] = (self.data['FamilySize'] == 1).astype(int)
            self.data['Title'] = self.data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False).map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare':4}).fillna(4).astype(int)
            self.data['Pclass_Fare'] = self.data['Pclass'] * self.data['Fare']
            self.data['Age_Fare'] = self.data['Age'] * self.data['Fare']

            logger.info("Data Preprocessing completed")
        
        except Exception as e:
            logger.error(f"Error while processing data {e}")
            raise CustomException(e)
        
    def handle_imbalaced_data(self):
        try:
            x = self.data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',  'FamilySize', 'HasCabin', 'Isalone', 'Title', 'Pclass_Fare', 'Age_Fare'] ]
            y = self.data['Survived']

            smote = SMOTE(random_state=42)
            self.X_resampled, self.y_resampled = smote.fit_resample(x, y)
            logger.info("Imbalanced data handled using SMOTE.") 

        except Exception as e:
            logger.error(f"Error while handling imbalanced data: {e}")
            raise CustomException(e)
        
    def store_features_in_redis(self):
        try:
            batch_data = {}
            for idx, row in self.data.iterrows():
                entity_id = row['PassengerId']
                features = {
                    "Age": row['Age'],
                    "Pclass": row['Pclass'],
                    "Fare": row['Fare'],
                    "Sex": row['Sex'],
                    "Embarked": row['Embarked'],
                    "FamilySize": row['FamilySize'],
                    "HasCabin": row['HasCabin'],
                    "Isalone": row['Isalone'],
                    "Title": row['Title'],
                    "Pclass_Fare": row['Pclass_Fare'],
                    "Age_Fare": row['Age_Fare'],
                    "Survived": row['Survived']
                }
                batch_data[entity_id] = features
            
            self.feature_store.store_batch_features(batch_data)
            logger.info("Features stored in Redis successfully.")
        
        except Exception as e:
            logger.error(f"Error storing features in Redis: {e}")
            raise CustomException(e)
        
    def retrieve_features_from_redis(self, entity_ids):
        features = self.feature_store.get_features(entity_ids)
        if features:
            return features
        return None

    
    def run(self):
        try:
            self.load_data()
            self.process_data()
            self.handle_imbalaced_data()
            self.store_features_in_redis()
            logger.info("Data processing pipeline completed.")
        except Exception as e:
            logger.error(f"Error in data processing pipeline: {e}")
            raise CustomException(e)
        
if __name__ == "__main__":
    feature_store = RedisFeatureStore(host='localhost', port=6379, db=0)
    data_processor = DataProcessor(TRAIN_PATH, TEST_PATH, feature_store=feature_store)
    data_processor.run()

    print(data_processor.retrieve_features_from_redis(entity_ids="332"))