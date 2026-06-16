# SLECS Explorer


![描述文字](images/demo.png)



Description
-----------
SLECS Explorer is a framework for constructing and exploring Skeleton-Level Chemical Space (SLECS) from tandem mass spectrometry (MS/MS) data. SLECS is a feature-structured representation of MS/MS data derived from non-negative matrix factorization (NMF), in which MS/MS spectra are reorganized according to fragment feature patterns extracted through NMF decomposition to reveal underlying skeleton-level connections among metabolites. This skeleton-level chemical space facilitates the prioritization of structurally distinct regions and supports structure-guided discovery of novel molecular scaffolds.

This repository provides the source code, demonstration datasets, and web interface associated with the SLECS workflow. Representative input datasets and source code implementing the key steps of the workflow are included in this repository.

The complete datasets supporting this study, including raw MS/MS files, processed results, and figure source data required for reproducing the published analyses, are publicly available through Zenodo:

**https://doi.org/10.5281/zenodo.20700190**
<img width="6624" height="233" alt="image" src="https://github.com/user-attachments/assets/51363fad-0760-4aea-aca9-d9fd78030b6e" />



## Directory Structure

```plaintext
SLECS Explorer/
├── README.md       
├── requirements.txt
├── docs/         
└── demo/            
``` 

## DATA
The `input/` folder contains representative datasets used to demonstrate the SLECS workflow, including:

* Raw MS/MS spectra (`*.mgf`);
* Pre-filtered MS/MS spectra (`*.mgf`);
* Molecular networking files generated through the GNPS-FBMN workflow, containing similarity information between connected nodes (`*.graphml`).

These datasets can be used together with the scripts provided in this repository to reproduce the construction of fragment-feature matrices and subsequent NMF analyses.

The `demo/` folder provides a pre-generated optimized fragment-feature matrix for rapid testing of the SLECS workflow.
<img width="1865" height="293" alt="image" src="https://github.com/user-attachments/assets/305cc20b-59a8-4d65-9db7-4e2607f671e4" />





## Contact

For questions regarding the dataset or workflow,please contact shuchenlan@simm.ac.cn(Chenlan Shu) or yuzhuohao@simm.ac.cn(Zhuohao Yu)
