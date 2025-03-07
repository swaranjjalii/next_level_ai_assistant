import pandas as pd
import json
import logging
from typing import Union, Dict, List, Any

class DataProcessor:
    """
    A class for processing and loading data from various file formats.
    Supports CSV, JSON, and plain text files.
    """

    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: DataFrame containing the CSV data.

        Raises:
            Exception: If the file cannot be loaded.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logging.error(f"Error loading CSV file '{file_path}': {e}")
            raise

    @staticmethod
    def load_json(file_path: str) -> Union[Dict, List]:
        """
        Load data from a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            Union[Dict, List]: Parsed JSON data as a dictionary or list.

        Raises:
            Exception: If the file cannot be loaded.
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading JSON file '{file_path}': {e}")
            raise

    @staticmethod
    def load_text(file_path: str) -> str:
        """
        Load data from a plain text file.

        Args:
            file_path (str): Path to the text file.

        Returns:
            str: Content of the text file as a string.

        Raises:
            Exception: If the file cannot be loaded.
        """
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error loading text file '{file_path}': {e}")
            raise

    @staticmethod
    def save_csv(data: pd.DataFrame, file_path: str) -> None:
        """
        Save data to a CSV file.

        Args:
            data (pd.DataFrame): DataFrame to save.
            file_path (str): Path to save the CSV file.

        Raises:
            Exception: If the file cannot be saved.
        """
        try:
            data.to_csv(file_path, index=False)
        except Exception as e:
            logging.error(f"Error saving CSV file '{file_path}': {e}")
            raise

    @staticmethod
    def save_json(data: Union[Dict, List], file_path: str) -> None:
        """
        Save data to a JSON file.

        Args:
            data (Union[Dict, List]): Data to save.
            file_path (str): Path to save the JSON file.

        Raises:
            Exception: If the file cannot be saved.
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving JSON file '{file_path}': {e}")
            raise

    @staticmethod
    def save_text(data: str, file_path: str) -> None:
        """
        Save data to a plain text file.

        Args:
            data (str): Text data to save.
            file_path (str): Path to save the text file.

        Raises:
            Exception: If the file cannot be saved.
        """
        try:
            with open(file_path, 'w') as f:
                f.write(data)
        except Exception as e:
            logging.error(f"Error saving text file '{file_path}': {e}")
            raise