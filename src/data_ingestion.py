import psycopg2
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import os
from sklearn.model_selection import train_test_split
import sys
from config.database_config import DB_CONFIG
from config.paths_config import RAW_DIRECTORY, TEST_PATH, TRAIN_PATH

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, db_params, output_dir):
        self.db_params = db_params
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
    
    def connect_db(self):
        try:
            conn = psycopg2.connect(**self.db_params)
            logger.info("Database connection established.")
            return conn
        except Exception as e:
            logger.error("Error connecting to database.")
            raise CustomException(e, sys) from e
    
    def extract_data(self):
        try:
            conn = self.connect_db()
            query = "SELECT * from titanic_data"
            df = pd.read_sql_query(query, conn)
            conn.close()
            logger.info("Data extraction successful.")
            return df
        except Exception as e:
            logger.error("Error extracting data from database.")
            raise CustomException(e, sys) from e
    
    def split_and_save_data(self, df):
        try:
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_df.to_csv(TRAIN_PATH, index=False)
            test_df.to_csv(TEST_PATH, index=False)
            logger.info(f"Data split into train and test sets and saved to {self.output_dir}.")
        except Exception as e:
            logger.error("Error splitting and saving data.")
            raise CustomException(e, sys) from e 
        
    def run(self):
        try:
            logger.info("Starting data ingestion process.")
            df = self.extract_data()
            self.split_and_save_data(df)
            logger.info("Data ingestion process completed successfully.")
        except Exception as e:
            logger.error("Data ingestion process failed.")
            raise CustomException(e, sys) from e
        

if __name__ == "__main__":
    data_ingestion = DataIngestion(DB_CONFIG, RAW_DIRECTORY)
    data_ingestion.run()

