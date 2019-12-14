# CS221-Project

### Data

The dataset is Chexpert dataset from Stanford Machine Learning Group. We downloaded the smaller version and hosted it in Google Cloud Storage.

Before launching any job, downloading and processing the csv file of training and validation with:

```
bash ./data/download_data.sh
python3 ./data/data_processing.py
```

### Training

```
# global model
python3 main.py ResNet50

# local model (need to run Attention Cropping Experiment.ipynb prior)
python3 main.py AG-CNN-FINAL
```
Fusion branch training is in `Fusion Branching Training.ipynb`
### Attention Cropping Experiment.ipynb

This notebook is for the attention area cropping experiment. 
Inside the notebook, we visualize all contour area for the input image so that 
we are able to pick the best one to represent the "enlarged heart". We also perform extraction operation against original image in this notebook.

### Fusion Preparation.ipynb
This notebook will run both global and local branch to get the global-average-pooling-layer output so that we can concatenate 
them and finally feed into fusion branch

### Fusion Branching Training.ipynb
This notebook simply build a fully connected layer with sigmoid to take concatenated input 

### Evaluation
After all of the training on global, local and fusion branch, this notebook will finally evaluate the prediction performance on the test set.
Inside of it we also visualize the cases where either branch made mistakes so that we can analyze why and how attention helps.

