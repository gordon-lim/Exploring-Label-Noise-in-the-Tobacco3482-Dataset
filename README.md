# Exploring Label Noise in the Tobacco3482 Dataset: Null Results and Other Issues

<p align="center">
  <embed src="Figures/example-errors.pdf" />
</p>

This repository contains code and dataset annotations for the paper *Exploring Label Noise in the Tobacco3482 Dataset: Null Results and Other Issues*.
## Setup
To reproduce benchmark results from the paper, please first download the Tobacco3482 dataset from Kaggle. You may follow instructions from [this tutorial](https://www.endtoend.ai/tutorial/how-to-download-kaggle-datasets-on-ubuntu/) and download the dataset from Kaggle [here](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg). You should have something like the following:
```
kaggle datasets download patrickaudriaz/tobacco3482jpg
```
Next, install the required Python packages.
```
pip install -r requirements.txt
```
## Benchmarks
To run benchmark, use `evaluate.py`. You may provide three arguments, namely `model`,  `dataset` and `reproduce`. To reproduce our VGG16 results on the original Tobacco3482 you can run
```
python evaluate.py -model vgg16 -dataset tobacco3482 --reproduce
```
Or if you want to train ResNet50 yourself and benchmark on CleanTobacco you can omit the `--reproduce` flag 
```
python evaluate.py model resnet50 -dataset cleantobacco  
```
## CleanTobacco
Please find our Tobacco3482 annotations in `tobacco3482-errors.csv`. The table has the following columns:

 - **filename:** Tobacco3482 image filename
 - **Mislabeled:** 1 if identified as label error else 0
 - **Low Quality:** "Yes" if identified as low quality image else NaN
 - **Contains also:** Multi-label issue e.g. an image *has* both Letter and Note
 - **Can be:** Ontological class overlap i.e a document can semantically both be Scientific and Report
 - **label:** Tobacco3482 given label

Here is a snippet of the table:
|  **filename**  | **Mislabeled**  | **Low Quality** | **Contains also** | **Can be** | **label** |
| --- | --- | --- | --- | --- | --- |
| 500323180+-3180.jpg | 0 | Yes | Form | NaN | ADVE |
| 501131762+-1762.jpg | 1 | NaN | NaN | NaN | ADVE |
| 501131762+-1762.jpg | 1 | NaN | NaN | NaN | ADVE |

Note: 'Low Quality' and 'Contains also' are not discussed in the paper but provided anyways. All identified issues including 'Mislabeled' and 'Can be' are non-exhaustive and there are potentially more errors.
## CleanLab
Use `cleanlab-cross-validation.py` to obtain out-of-sample predicted probabilities using  5-fold cross-validation for CleanLab. See `CleanLab Report.ipynb` for CleanLab results.
