import pandas as pd
print("=== YOUR ACTUAL COLUMNS ===")
for name, file in [("ENROL", "api_data_aadhar_enrolment_0_500000.csv"), 
                   ("DEMO", "api_data_aadhar_demographic_0_500000.csv"),
                   ("BIO", "api_data_aadhar_biometric_0_500000.csv")]:
    cols = pd.read_csv(file).columns.tolist()
    print(f"{name}: {cols}")
