"""
Utility functions for UIDAI pipeline
All reusable components extracted here
"""

import pandas as pd
import numpy as np
import logging
from difflib import get_close_matches
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """Validates data quality with detailed logging"""
    
    @staticmethod
    def validate_pincodes(df, col='pincode'):
        """
        Validate Indian pincodes
        
        Args:
            df: DataFrame
            col: Pincode column name
        
        Returns:
            DataFrame with pincode_valid column
        """
        try:
            df['pincode_valid'] = (
                (df[col] >= 100000) & 
                (df[col] <= 855999) &
                (df[col] != 999999)
            ).astype('int8')
            
            invalid_count = (~df['pincode_valid'].astype(bool)).sum()
            logger.info(f"Pincode validation: {invalid_count:,} invalid ({invalid_count/len(df)*100:.2f}%)")
            
            return df
        
        except Exception as e:
            logger.error(f"Pincode validation failed: {e}")
            raise
    
    @staticmethod
    def detect_outliers_iqr(df, numeric_cols, multiplier=1.5):
        """
        IQR-based outlier detection
        
        Args:
            df: DataFrame
            numeric_cols: List of columns to check
            multiplier: IQR multiplier (default 1.5 = standard)
        
        Returns:
            DataFrame, outlier flags, outlier count
        """
        try:
            outlier_flags = pd.DataFrame(index=df.index)
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                outlier_flags[f'{col}_outlier'] = (
                    (df[col] < lower_bound) | (df[col] > upper_bound)
                ).astype('int8')
            
            outlier_count = outlier_flags.sum().sum()
            logger.info(f"Outliers detected (IQR): {outlier_count:,} values")
            
            return df, outlier_flags, outlier_count
        
        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
            raise


class NameStandardizer:
    """Standardize geographic names using fuzzy matching"""
    
    @staticmethod
    def clean_text(text):
        """Normalize text for matching"""
        if pd.isna(text):
            return ''
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    @classmethod
    def standardize_state(cls, state_name, official_list, fuzzy_cutoff=0.75):
        """
        Match state name to official list
        
        Args:
            state_name: Input state name
            official_list: Canonical state names
            fuzzy_cutoff: Similarity threshold
        
        Returns:
            Standardized state name
        """
        if pd.isna(state_name) or state_name == '':
            return 'Unknown'
        
        # Clean
        cleaned = str(state_name).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'[&]', 'and', cleaned)
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
        
        # Exact match
        for official in official_list:
            official_clean = re.sub(r'[^\w\s]', '', official)
            if cleaned.lower() == official_clean.lower():
                return official
        
        # Historical variants
        HISTORICAL_MAPPING = {
            'orissa': 'Odisha',
            'uttaranchal': 'Uttarakhand',
            'pondicherry': 'Puducherry'
        }
        
        if cleaned.lower() in HISTORICAL_MAPPING:
            return HISTORICAL_MAPPING[cleaned.lower()]
        
        # Fuzzy match
        matches = get_close_matches(
            cleaned, 
            [re.sub(r'[^\w\s]', '', s) for s in official_list],
            n=1, 
            cutoff=fuzzy_cutoff
        )
        
        if matches:
            clean_official_list = [re.sub(r'[^\w\s]', '', s) for s in official_list]
            idx = clean_official_list.index(matches[0])
            return official_list[idx]
        
        return cleaned.title()
    
    @staticmethod
    def clean_district(name):
        """Clean district names"""
        if pd.isna(name) or name == '':
            return 'Unknown'
        
        name = str(name).strip()
        name = re.sub(r'^(Dist|District)\s*:?\s*', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s*\*+\s*$', '', name)
        name = re.sub(r'\s+', ' ', name)
        
        if name.isupper() or name.islower():
            name = name.title()
        
        return name
