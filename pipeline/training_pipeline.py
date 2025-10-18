from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessor   
from src.model_training import ModelTrainer
from src.feature_store import RedisFeatureStore
from config.database_config import DB_CONFIG
from config.paths_config import RAW_DIRECTORY, TRAIN_PATH, TEST_PATH

if __name__ == "__main__":
    data_ingestion = DataIngestion(DB_CONFIG, RAW_DIRECTORY)
    data_ingestion.run()

    feature_store = RedisFeatureStore()
    data_processor = DataProcessor(TRAIN_PATH, TEST_PATH, feature_store=feature_store)
    data_processor.run()

    feature_store = RedisFeatureStore()
    model_trainer = ModelTrainer(feature_store=feature_store)
    model_trainer.run()
