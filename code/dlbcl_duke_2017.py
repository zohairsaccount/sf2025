import pandas as pd

# Load the patient-level clinical data
file_path = "../data/dlbcl_duke_2017/data_clinical_patient.txt"
df_patient = pd.read_csv(file_path, sep="\t", comment="#")  # Ignore header comments

# Print first few rows
print(df_patient.head())

# Print column names to understand the structure
print(df_patient.columns)

# Load the sample-level clinical data
file_path_sample = "../data/dlbcl_duke_2017/data_clinical_sample.txt"  # Corrected path
df_sample = pd.read_csv(file_path_sample, sep="\t", comment="#")  # Ignore header comments

# Print first few rows
print(df_sample.head())

# Print column names
print(df_sample.columns)

# Merge patient and sample data (assuming 'Patient_ID' is a common column)
df_merged = pd.merge(df_patient, df_sample, on="PATIENT_ID", how="inner")
print(df_merged.head())

# Load the mutation data
mutation_file_path = "../data/dlbcl_duke_2017/data_mutations.txt"  # Adjust file path as necessary
df_mutation = pd.read_csv(mutation_file_path, sep="\t", comment="#")

# Check the first few rows and columns to understand the structure
print(df_mutation.head())
print(df_mutation.columns)

# Merge mutation data with patient-level clinical data based on 'PATIENT_ID' and 'Tumor_Sample_Barcode'
df_merged = pd.merge(df_patient, df_mutation, left_on='PATIENT_ID', right_on='Tumor_Sample_Barcode', how='inner')

# Check the first few rows of the merged data
print(df_merged.head())

# Check for any differences in the patient/sample IDs
patient_ids = df_patient['PATIENT_ID'].unique()
tumor_barcodes = df_mutation['Tumor_Sample_Barcode'].unique()

# Check if there are mismatches or formatting issues
print(f"Patient IDs in clinical data: {len(patient_ids)}")
print(f"Tumor barcodes in mutation data: {len(tumor_barcodes)}")
print(f"Matching Tumor_Sample_Barcode and PATIENT_ID: {sum(df_mutation['Tumor_Sample_Barcode'].isin(df_patient['PATIENT_ID']))}")


# Clean the IDs in both datasets
df_patient['PATIENT_ID'] = df_patient['PATIENT_ID'].astype(str).str.strip().str.upper()
df_mutation['Tumor_Sample_Barcode'] = df_mutation['Tumor_Sample_Barcode'].astype(str).str.strip().str.upper()

# Select the columns we need from the mutation file: Patient identifier and gene name
df_mutation_filtered = df_mutation[['Tumor_Sample_Barcode', 'Hugo_Symbol']].copy()

# Since the presence of a row means the gene is mutated, create a column that marks this with a 1
df_mutation_filtered['Mutation_Status'] = 1

# Let's check a sample of the mutation data
print(df_mutation_filtered.head())

# Pivot the data to create the mutation matrix
mutation_matrix = df_mutation_filtered.pivot_table(index='Tumor_Sample_Barcode',
                                                   columns='Hugo_Symbol',
                                                   values='Mutation_Status',
                                                   aggfunc='max',
                                                   fill_value=0)

# Rename the index for clarity
mutation_matrix.index.name = 'PATIENT_ID'

# Preview the matrix
print("Mutation matrix from mutation file:")
print(mutation_matrix.head())


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Assume mutation_matrix is your DataFrame (without the index name "PATIENT_ID")
# If the mutation_matrix has a 'Cluster' column from a previous run, remove it:
if 'Cluster' in mutation_matrix.columns:
    X = mutation_matrix.drop('Cluster', axis=1)
else:
    X = mutation_matrix.copy()

# Define the number of clusters you want to try (adjust as needed)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

# Add the cluster assignments to your mutation matrix for later reference
mutation_matrix['Cluster'] = clusters

# Optionally, use PCA to visualize the clusters in 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='viridis', alpha=0.7)
plt.title("Clustering of Patients Based on Mutation Profiles")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster")
plt.show()


import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# 'X' is your mutation matrix without the 'Cluster' column
X = mutation_matrix.drop('Cluster', axis=1) if 'Cluster' in mutation_matrix.columns else mutation_matrix.copy()

# Agglomerative Clustering using a distance threshold.
# Adjust the 'distance_threshold' parameter to control the granularity of clusters.
agg_cluster = AgglomerativeClustering(distance_threshold=1.5, n_clusters=None)
clusters = agg_cluster.fit_predict(X)

# Add the cluster assignments back to the mutation matrix
mutation_matrix['Cluster'] = clusters

# Visualize using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.title("Agglomerative Clustering (distance_threshold=1.5)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster")
plt.show()

