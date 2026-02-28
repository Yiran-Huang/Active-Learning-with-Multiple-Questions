## Handwriting: 

- Available at `https://doi.org/10.24432/C50P49`.  
- For convenience, it is included in `Dataset/Handwriting_dataset`.
- File format: csv.
- Original pixel value range: $\{0,\dots,16\}$.

Each $32\times32$ bitmap image is partitioned into non-overlapping $4\times4$ blocks, counting active pixels per block to form an $8\times8$ matrix (values 0--16).



## HAR: 

- Available at `https://doi.org/10.24432/C54S4K`. 
- For convenience, it is included in `Dataset/HAR_dataset`.
- File format: txt.
- Original variable value range: $(-\infty,+\infty)$.

30 volunteers (age 19–48) performed six daily activities while wearing a waist-mounted smartphone that recorded 3-axis acceleration and gyroscope data at 50Hz. Signals were filtered, segmented into overlapping 2.56-second windows, and transformed into 561 time- and frequency-domain features



## MNIST：

- Publicly available from multiple sources, e.g., `https://www.openml.org/d/554`.  
- For convenience, it is included in `Dataset/MNIST_dataset`.
- File format: pkl.
- Original pixel value range: $\{0,\dots,255\}$.

MNIST consists of 70,000 grayscale handwritten digit images ($28\times 28$), split into 60,000 training and 10,000 test samples.



## Animal:

- The original dataset is available at `https://www.kaggle.com/datasets/alessiocorrado99/animals10`.  
- We provide a preprocessed version at `https://huggingface.co/datasets/YRHuang/Animal_Image_Dataset`.
- File format: npy.
- Original pixel value range: $\{0,\dots,255\}$.

The dataset contains approximately 28,000 animal images from 10 categories: dog, cat, horse, spider, butterfly, chicken, sheep, cow, squirrel, and elephant.  
Images with a single color channel were removed, and the remaining images were reshaped to $3\times224\times224$. The preprocessed data are available at `https://huggingface.co/datasets/YRHuang/Animal_Image_Dataset`.



## Tumor:

- The original dataset is available at `https://doi.org/10.34740/kaggle/dsv/2645886`.  
- We provide a preprocessed version at `https://huggingface.co/datasets/YRHuang/Tumor_Image_Dataset`.
- File format: npy.
- Original pixel value range: $\{0,\dots,255\}$.

This dataset combines Figshare, SARTAJ, and Br35H datasets. It contains 7,023 brain MRI images classified into four categories: glioma, meningioma, no tumor, and pituitary.  
All images were reshaped to $3\times224\times224$. The preprocessed data are available at `https://huggingface.co/datasets/YRHuang/Tumor_Image_Dataset`.

