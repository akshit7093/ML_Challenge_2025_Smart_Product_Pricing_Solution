import pandas as pd

# Define file paths
file_path_in = r'c:\Users\Akshit\OneDrive\Documents\code\New folder (5)\68e8d1d70b66d_student_resource\student_resource\dataset\sample_test.csv'
file_path_out = r'c:\Users\Akshit\OneDrive\Documents\code\New folder (5)\68e8d1d70b66d_student_resource\student_resource\dataset\sample_test_out.csv'

# Read the datasets
df_in = pd.read_csv(file_path_in)
df_out = pd.read_csv(file_path_out)

print("--- Structure and details for sample_test.csv ---")
print("\nHead of the DataFrame:")
print(df_in.head())
print("\nInfo about the DataFrame:")
print(df_in.info())
print("\nDescription of numerical columns:")
print(df_in.describe())
print("\n")

print("--- Structure and details for sample_test_out.csv ---")
print("\nHead of the DataFrame:")
print(df_out.head())
print("\nInfo about the DataFrame:")
print(df_out.info())
print("\nDescription of numerical columns:")
print(df_out.describe())