from dataclasses import field
import pandas as pd
from typing import List,Dict,Any
import os
import logging
from pathlib import Path
import diskcache as dc
from config.settings import Config

logger=logging.getLogger(__name__)

def get_cache() -> dc.Cache:
    """Initializes and returns a diskcache.Cache object."""
    cache_dir = Path(Config.CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return dc.Cache(str(cache_dir))


class Utils:
    @staticmethod
    def load_employee_data(csv_path:str)->pd.DataFrame:
        # csv data 
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Error loading employee data: {e}")
            return pd.DataFrame()
        
    @staticmethod
    def find_employee_by_name(df: pd.DataFrame,name:str)->Dict[str,Any]:
        # finding employees by their name in the dataframe

        matches=df[df['Employee Name'].str.contains(name,case=False,na=False)]
        if not matches.empty:
            return matches.iloc[0].to_dict()
        return {}

    @staticmethod
    def format_currency(amount:float)->str:
        return f"₹{amount:,.0f}"

    @staticmethod
    def get_file_extention(filename:str)->str:
        return os.path.splitext(filename)[1].lower()
    
    @staticmethod
    def validate_employee_data(employee_data:Dict[str,Any])->bool:
        required_fields=['Employee Name', 'Department', 'Band', 'Base Salary (INR)', 'Total CTC (INR)']
        return all(field in employee_data for field in required_fields)