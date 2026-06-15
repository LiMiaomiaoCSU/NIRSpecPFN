# NIRSpecPFN
***
This is the code repo for the paper **In-Context Learning with Prior-Data Fitted Networks for Efficient Nonlinear Modeling in Near-Infrared Spectroscopy**. We developed a method for Near-Infrared Spectral quantitative analysis.This is a efficient approach based on the TabPFN-2.5, which is a Transformer-Based foundation model. NIRSpecPFN comprises three key steps: spectral processing, context preparation, and in-context learning.

<img width="7372" height="7205" alt="Figure1" src="https://github.com/user-attachments/assets/49562061-6fcf-4c80-a602-2529cb16c967" />






# Installation & Setup
***
* We recommend to use pip. By using the [requirements.txt](requirements.txt) file, it will install all the required packages.

    ```python
    git clone https://github.com/lmmcsu/NIRSpecPFN.git
    cd NIRSpecPFN
    ```
* The NIRSpecPFN requires Python 3.9+. For further details regarding the installation and configuration of TabPFN, please refer to [TabPFN](https://github.com/PriorLabs/TabPFN).
* Download TabPFN-2.5 model weights

    * Visit https://huggingface.co/Prior-Labs/tabpfn_2_5 and accept the license terms.
    * File location
      
    ```python
    # set your local cache directory here
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", r"D:\workspace\TabPFN\tabpfn")
    ```

# Workflow
***
## 1. Dataset
The experimental datasets are included in the [Datasets](Datasets) and organized from public datasets
<img width="978" height="320" alt="屏幕截图 2026-06-15 085408" src="https://github.com/user-attachments/assets/b3f6e1f3-b967-4af7-bfdf-abbacdb4bf74" />




## 2. Dataprocessing
We consolidates several commonly used spectral preprocessing methods into a Python package.
* [Spectral Preprocessing](preprocessing/process.py): airPLS, MSC, SNV, SG, SG-2D 

An example of data processing：

```python
from process import derivative

#Load the data
...Loading your sprectral data

#Data partitioning
X_support, X_query, y_support, y_query = train_test_split(spectra, y, test_size=0.2, random_state=42)

#Spectral processing
X_support_deriv = derivative(X_support)
X_query_deriv = derivative(X_query)
```

## 3. Modelling and prediction
NIRSpecPFN enables prediction of target values (chemical composition) on the test set without requiring hyperparameter tuning, utilising the support set of real spectral datasets as contextual information.
An example of modelling and prediction：

```python
from tabpfn import TabPFNRegressor

# ICL and inference
model = load_local_tabpfn(kind="regressor", version="2.5", variant="real") 
model.fit(X_support_deriv, y_support)
preds = model.predict(X_query_deriv)
```

# Usage
The example codes for usage is included in the [example.ipynb](example.ipynb).

* Regression performance: [Model](Model)
* Relative analysis: [Analysis](analysis)

# Information of maintainers
* zmzhang@csu.edu.cn
* 242311023@csu.edu.cn




