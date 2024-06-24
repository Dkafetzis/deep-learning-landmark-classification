# deep-learning-landmark-classification

## Environment setup
1. Install conda.
2. Install requirements.txt `conda create --name dl-env --file ./requirements.txt`.
3. Depending on environment cuda might or might not be needed consult [this page](https://pytorch.org/get-started/locally/) and install torch and friends accordingly.
4. Create an empty `checkpoints` directory if it does not exist already
5. Download the dataset
   - [International dataset](https://drive.google.com/drive/folders/1V1aAhMyjmPgT_dzNn5mONWzt-qds8gru?usp=sharing)
   - [Greek dataset](https://drive.google.com/drive/folders/1ayCzhYn3lODGh8tEtYzmSmnSHwwNlyHZ?usp=sharing)
6. Rename the dataset file to `landmark_images` if necessary.

## Training the models
The models can be trained from the 2 notebooks provided:
- `custom_cnn.ipynb` for out custom cnn model
- `transfer_learning.ipynb` for the 3 transfer learning models

Running the notebook trains the corresponding model saving the best performing checkpoint each time
and in the end of the training it wraps the model with a predicting wrapper to change the output of the model from
percentages to a single prediction, runs a confusion matrix and saves the model with the wrapper.

## Android application
The trained models are used in the [SpotTheSpot](https://github.com/Dkafetzis/spotthespot) application