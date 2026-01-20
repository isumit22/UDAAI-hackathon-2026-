"""
Configuration management for UIDAI Data Pipeline
All magic numbers, paths, and parameters centralized here
"""

import os

class PipelineConfig:
    """Central configuration class"""
    
    # Data Sources
    DATA_DIR = 'data/raw/'
    PROCESSED_DIR = 'data/processed/'
    OUTPUT_DIR = 'outputs/'
    
    ENROL_FILES = [
        'api_data_aadhar_enrolment_0_500000.csv',
        'api_data_aadhar_enrolment_500000_1000000.csv',
        'api_data_aadhar_enrolment_1000000_1006029.csv'
    ]
    
    DEMO_FILES = [
        'api_data_aadhar_demographic_0_500000.csv',
        'api_data_aadhar_demographic_500000_1000000.csv',
        'api_data_aadhar_demographic_1000000_1500000.csv',
        'api_data_aadhar_demographic_1500000_2000000.csv',
        'api_data_aadhar_demographic_2000000_2071700.csv'
    ]
    
    BIO_FILES = [
        'api_data_aadhar_biometric_0_500000.csv',
        'api_data_aadhar_biometric_500000_1000000.csv',
        'api_data_aadhar_biometric_1000000_1500000.csv',
        'api_data_aadhar_biometric_1500000_1861108.csv'
    ]
    
    # Processing Parameters
    OUTLIER_METHOD = 'IQR'
    OUTLIER_MULTIPLIER = 1.5  # Standard IQR threshold
    MISSING_STRATEGY = 'district_median'
    INVALID_PINCODE_THRESHOLD = 100000
    
    # Feature Engineering
    KMEANS_CLUSTERS = 5
    ISOLATION_FOREST_CONTAMINATION = 0.05
    RANDOM_SEED = 42
    
    # Fuzzy Matching
    FUZZY_MATCH_THRESHOLD = 0.75
    
    # Official Indian States (Census 2011 + 2020 reorganization)
    OFFICIAL_STATES = [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
        'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
        'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
        'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
        'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
        'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
        'Andaman and Nicobar Islands', 'Chandigarh', 
        'Dadra and Nagar Haveli and Daman and Diu', 'Delhi',
        'Jammu and Kashmir', 'Ladakh', 'Lakshadweep', 'Puducherry'
    ]
    
    # City-to-State mapping (for data quality fixes)
    CITY_STATE_MAPPING = {
        'Balanagar': 'Telangana',
        'Madanapalle': 'Andhra Pradesh',
        'Puttenahalli': 'Karnataka',
        'Raja Annamalai Puram': 'Tamil Nadu'
    }
    
    @classmethod
    def create_directories(cls):
        """Create output directories if they don't exist"""
        os.makedirs(cls.PROCESSED_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
    
    @classmethod
    def get_full_path(cls, filename, dir_type='output'):
        """Get full file path"""
        if dir_type == 'output':
            return os.path.join(cls.OUTPUT_DIR, filename)
        elif dir_type == 'processed':
            return os.path.join(cls.PROCESSED_DIR, filename)
        else:
            return filename
