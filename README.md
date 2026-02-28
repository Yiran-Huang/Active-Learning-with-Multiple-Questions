# Eﬃcient Human-in-the-Loop Active Learning: A Novel Framework for Data Labeling in AI Systems

We provide the code for the paper. A Jupyter Notebook (in the `Outcomes` folder) is also included to directly generate the figures from the uploaded outputs, without rerunning the code.

This `requirements.txt` file specifies the required Python dependencies. Install them using `pip install -r requirements.txt`.

It is recommended to use **Python 3.6 or later**.



## Dependencies

Our program uses five packages: "numpy", "scipy", "torch", "torchvision", and "matplotlib".
If these packages are not installed, please open the Command Prompt (Windows) or Terminal (Mac), and run `pip install -r requirements.txt`



## Folder Structure

- Handwriting: code for the Handwriting dataset.
- HAR: code for the HAR dataset.
- MNIST: code for the MNIST dataset.
- Animal: code for the Animal dataset.
- Tumor: code for the Tumor dataset.
- Dataset: local datasets for Handwriting, HAR, and MNIST. The Animal and Tumor datasets are hosted externally (see the **Datasets** section below for download instructions).
- Outcomes: results used to generate **Figures 3--5, S4--S5** of the paper. To reproduce the figures directly without running the previous folders, execute `Reproducible_report_figures.ipynb` in this directory.



## Datasets

Five datasets are used:

- Handwriting: available at `https://doi.org/10.24432/C50P49`.  
  For convenience, it is included in `Dataset/Handwriting_dataset`.

- HAR: available at `https://doi.org/10.24432/C54S4K`.  
  For convenience, it is included in `Dataset/HAR_dataset`.
- MNIST: publicly available from multiple sources, e.g., `https://www.openml.org/d/554`.  
  For convenience, it is included in `Dataset/MNIST_dataset`.
- Animal: the original dataset is available at `https://www.kaggle.com/datasets/alessiocorrado99/animals10`.  
  We provide a preprocessed version at  
  `https://huggingface.co/datasets/YRHuang/Animal_Image_Dataset`.
- Tumor: the original dataset is available at `https://doi.org/10.34740/kaggle/dsv/2645886`.  
  We provide a preprocessed version at  
  `https://huggingface.co/datasets/YRHuang/Tumor_Image_Dataset`.

For additional dataset details, please refer to `data_dictionary.md`.



## Run Simulations

### Run Handwriting, HAR, and MNIST:

Execute `main.py` in the corresponding folder.

**Estimated running time:**

\- Handwriting: approximately 4 minutes per iteration (CPU).  
  The default number of iterations is set to 5 (30 in the paper).

\- HAR: approximately 4 minutes per iteration (CPU).  
  The default number of iterations is set to 5 (30 in the paper).

For MNIST, running on a GPU is strongly recommended. Each iteration may take several hours.

### Run Animal and Tumor datasets:

First download the folder `Animal_dataset` (`Tumor_dataset`) from the released repository on Hugging Face and place it under the `Dataset` directory. Then run `main.py` in the corresponding folder.

Running on a GPU is strongly recommended. Each iteration may take several hours.



## Core Files

Key files, functions, and selected function arguments are summarized below.

### `core_function.py`

- `build_model_A`: Fits the model based on the answers to the questions.
- `exploration_and_exploitation`: The proposed exploration and exploitation framework for data screening.
- `AL_multi_question`: The proposed active learning procedure. Selected input arguments include:

  - `get_model_f`: Model initialization function.
  - `L`: Number of classes.
  - `trueanswer_fun`: Function that returns the true answer to a question.
  - `locX`: Indices of `Xall` queried by the “Class” question.
  - `maxm`: Maximum value of $m$ across all questions.
  - `question_set`: List of available questions.
  - `question_and_answer`: List of realized answers for each question.
  - `AL_max_iteration`: Maximum number of iterations.
  - `hyper_regular`: $L_1$ penalty coefficient for model parameters (set to 0 in all simulations in the paper).
  - `layers_to_regularize`: Layers subject to the $L_1$ penalty (set to `None` in all simulations in the paper).
  - `e_and_e`: Whether to use the exploration and exploitation framework.
  - `trans_x`: Transformation function applied to $x$ before feeding it into the model. For example, in the Animal dataset, model fitting typically requires normalization of $X$, which converts the data type to `float32` and substantially increases memory usage. A dataset of size $20000 \times 3 \times 224 \times 224$ occupies approximately 11.2 GB in `float32`, compared to about 2.8 GB in `uint8`. Therefore, for the Animal and Tumor datasets, $X$ is stored as `uint8`. During model fitting or probability prediction, each batch is normalized and converted to `float32` via `trans_x`. The design trades additional computation time for reduced memory consumption.

### Question_set.py`

- `Exchanging_algorithm_includecluster`: Exchanging algorithm for questions involving class labels (e.g., “All” and “Any” types).

- `Exchanging_algorithm_notincludecluster`: Exchanging algorithm for questions not involving class labels (e.g., “Class” type).

- `Create_question`: Constructs a question object, including cost, acquisition function, and exchanging algorithm.

  - `loss_function`: Loss function associated with the question.
  - `change_fast_function`: Computes the acquisition value when only one sample in the realization is changed.
  - `deltaen2function`: Acquisition function.
  - `screen_function`: Quickly filters out samples unlikely to yield high acquisition values.
  - `exchang_function`: Exchanging algorithm used for this question.
  - `parameter`: A list containing the $m$ value, cost, and threshold for `screen_function`.

  For convenience, several question types are pre-defined. Users only need to specify the $m$ value, cost, threshold for `screen_function`, and (if required) the MCdrop rate.

- `Q1_create_GainTVq`: Creates an “All” question under the proposed criterion.

- `Q2_create_GainTVq`: Creates an “Any” question under the proposed criterion.

- `Q4_create_GainTVq`: Creates a “Class” question under the proposed criterion.

- `Q4_create_entropy`: Creates a “Class” question under the entropy-based criterion.



