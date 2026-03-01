## Handwriting: 

- Available at `https://doi.org/10.24432/C50P49`.  
- For convenience, it is included in `Dataset/Handwriting_dataset`.
- File format: csv.
- $X$: 64-dimensional feature vector (8×8), uint8, pixel counts for each block, value range ${0,\dots,16}$.
- $Y$: class label, categorical (10 classes).
- Variable file: `data.csv`. Inside the file, the features have column name `V1`–`V64`, while the class label has column name `Yall`.

Each $32\times32$ bitmap image is partitioned into non-overlapping $4\times4$ blocks, counting active pixels per block to form an $8\times8$ matrix (values 0--16).



## HAR: 

- Available at `https://doi.org/10.24432/C54S4K`. 
- For convenience, it is included in `Dataset/HAR_dataset`.
- File format: txt.
- $X$: 561-dimensional feature vector, float32,  time- and frequency-domain features, real-valued.
- $Y$: class label for six daily activities, categorical (6 classes).
- Variable files: X_train.txt and X_test.txt are feature files. Y_train.txt and Y_test.txt are class label files.

30 volunteers (age 19–48) performed six daily activities while wearing a waist-mounted smartphone that recorded 3-axis acceleration and gyroscope data at 50Hz. Signals were filtered, segmented into overlapping 2.56-second windows, and transformed into 561 time- and frequency-domain features.



## MNIST：

- Publicly available from multiple sources, e.g., `https://www.openml.org/d/554`.  
- For convenience, it is included in `Dataset/MNIST_dataset`.
- File format: pkl.
- $X$: $1\times 28\times 28$ grayscale images, uint8, pixel value range: $\{0,\dots,255\}$.
- $Y$: class label for ten handwritten digits, categorical (10 classes).
- Variable files: `alldata.pkl` and `testdata.pkl` are Python lists, where the first element stores the image tensor (features) and the second element stores the class labels.

MNIST consists of 70,000 grayscale handwritten digit images ($28\times 28$), split into 60,000 training and 10,000 test samples.



## Animal:

- The original dataset is available at `https://www.kaggle.com/datasets/alessiocorrado99/animals10`.  
- We provide a preprocessed version at `https://huggingface.co/datasets/YRHuang/Animal_Image_Dataset`.
- File format: npy.
- $X$: $3\times 224\times 224$ RGB images, uint8, pixel value range: $\{0,\dots,255\}$.
- $Y$: class label for ten animals, categorical (10 classes).
- Variable files: For each class, all images are stored in a separate NPY file named `<label\_name>x.npy` (e.g., `canex.npy`). The class label is determined by the file name.

The dataset contains approximately 28,000 animal images from 10 categories: dog, cat, horse, spider, butterfly, chicken, sheep, cow, squirrel, and elephant.  
Images with a single color channel were removed, and the remaining images were reshaped to $3\times224\times224$. The preprocessed data are available at `https://huggingface.co/datasets/YRHuang/Animal_Image_Dataset`.



## Tumor:

- The original dataset is available at `https://doi.org/10.34740/kaggle/dsv/2645886`.  
- We provide a preprocessed version at `https://huggingface.co/datasets/YRHuang/Tumor_Image_Dataset`.
- File format: npy.
- $X$: $3\times 224\times 224$ brain MRI images, uint8, pixel value range: $\{0,\dots,255\}$.
- $Y$: class label for four diagnoses, categorical (4 classes).
- Variable files: For each class, all images are stored in a separate NPY file named `<label\_name>x.npy` (e.g., `gliomax.npy`). The class label is determined by the file name.

This dataset combines Figshare, SARTAJ, and Br35H datasets. It contains 7,023 brain MRI images classified into four categories: glioma, meningioma, no tumor, and pituitary.  
All images were reshaped to $3\times224\times224$. The preprocessed data are available at `https://huggingface.co/datasets/YRHuang/Tumor_Image_Dataset`.

