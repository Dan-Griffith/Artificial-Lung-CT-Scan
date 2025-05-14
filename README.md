This project details various ways on how to segment lungs from a thoractic cavity CT scan.
The data sets used in this project are: 
- https://www.kaggle.com/code/muhakabartay/osic-pulmonary-fibrosis-eda-dicom-full
- https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6ACUZJ

The recration is done using an autoencoder mapping the biplanar X-rays to the 3D point cloud
that has either been segmented using the file dcm_prep.py or with the software "3D Slicer".
