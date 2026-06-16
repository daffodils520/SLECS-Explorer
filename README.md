# SLECS Explorer


![描述文字](images/demo.png)



Description
-----------
This repository provides a submission-stage version of SLECS Explorer, an interactive web-based interface for exploring skeleton-level chemical space derived from MS/MS data.

SLECS Explorer is built upon a feature-structured analytical framework in which MS/MS data are reorganized through fragment–feature representation and matrix factorization to reveal a skeleton-level chemical space (SLECS). Within this space, structurally distinct regions can be explored and prioritized to support structure-guided discovery of novel molecular scaffolds.

This submission version includes:
- A minimal reproducible demo of the NMF decomposition step  
- A pre-generated fragment–intensity matrix for direct testing  
- A static web interface for conceptual illustration of the SLECS framework  

The complete preprocessing workflow, parameter settings, and full datasets are described in the manuscript and will be released upon publication.


## Directory Structure

```plaintext
SLECS Explorer/
├── README.md       
├── requirements.txt
├── docs/         
└── demo/            
``` 

## DATA
 The `demo/` folder provides a pre-generated optimized fragment–intensity matrix (`demo_optimized_fragment_matrix.csv`) that can be directly used as input for NMF decomposition within the SLECS-based framework.

This dataset allows users to reproduce the core step of fragment–feature factorization and explore the resulting skeleton-level chemical space.

The complete preprocessing workflow is described in the manuscript, and all related code and datasets will be released upon publication.




## Contact

For questions regarding the dataset or workflow,please contact shuchenlan@simm.ac.cn(Chenlan Shu) or yuzhuohao@simm.ac.cn(Zhuohao Yu)
