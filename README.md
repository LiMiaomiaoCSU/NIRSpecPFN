# NIRSpecPFN
***
This is the code repo for the paper **Precise Modeling of Scarce Near-Infrared Spectral Data Based on TabPFN**.
We developed a method for Near-Infrared Spectral Analysis.This is a supervised machine learning approach based on the TabPFN, which is a Transformer-Based model. NIRSpecPFN strictly adheres to structured machine learning methodologies, with the framework integrating key steps such as data preprocessing, feature selection, and modeling prediction.

<img width="822" height="462" alt="image" src="https://github.com/user-attachments/assets/a386cf14-b63f-4499-ba50-ff115df63ebe" />

# Installation & Setup
***
* Official installation of TabPFN (pip)

    ```python
    pip install tabpfn
    ```

* Compatible versions: 3.9, 3.10, 3.11, 3.12, 3.13.
The entire process of this experiment was implemented in Python 3.12.9. TabPFN requires Python 3.9+ due to newer language features. For further details regarding the installation and configuration of TabPFN, please refer to [TabPFN](https://github.com/PriorLabs/TabPFN).

# Workflow
***
## 1. Dataset
Our three experimental datasets are as follows:
| Dataset | Resource|
| --- | --- |
| Corn dataset | [Eigenvector Research](https://eigenvector.com/resources/data-sets/) |
| CGL dataset | [Eigenvector Research](https://eigenvector.com/resources/data-sets/) |
| Wheat dataset | [IDRC 2016](https://www.cnirs.org/content.aspx?page_id=86&club_id=409746) |

## 2. Dataprocessing
We consolidates several commonly used spectral preprocessing and feature selection methods into a Python package.
* Spectral Preprocessing:airPLS, MSC, SNV, Detrend, Derivative
* Feature Selection:SPA, Univariate, UVE, RFE
  An example of data processing：

    ```python
    from process import derivative
    from feature import rfe
    
    #Load the data
    ...Loading your sprectral data
    
    #Data partitioning
    X_train, X_test, y_train, y_test = train_test_split(spectra, y, test_size=0.25, random_state=42, shuffle=True)
    
    #Spectral preprocessing
    X_train_de = derivative(X_train)
    X_test_de = derivative(X_test)
    
    #Feature Selection
    X_train_rfe, X_test_rfe = rfe(X_train_de, y_train, X_test_de)
    ```

## 3. Modelling and prediction
NIRSpecPFN enables prediction of target values (chemical composition) on test sets without requiring hyperparameter tuning, utilising the train set of real spectral datasets as contextual information.
An example of modelling and prediction：

```python
from tabpfn import TabPFNRegressor

model = TabPFNRegressor(device=device, random_state=42, ignore_pretraining_limits=True)
model.fit(X_train_rfe, y_train)
```

# Usage
The example codes for usage is included in the example.ipynb

# Information of maintainers
* zmzhang@csu.edu.cn
* 242311023@csu.edu.cn




