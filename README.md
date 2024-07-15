# BERT-Based-Model-Training
#######################################################################################
## BERT-based Text Classification Model Training
### Kagan Yavuz
#

This code is written to train a classification model using the BERT model. Let's explain step by step what is being done:

### Importing Required Libraries
* Libraries like pandas and scikit-learn are used for data processing and splitting.
* The transformers library is used to provide the BERT model and tokenizer.
* The torch library is used to handle the dataset and model with PyTorch.

### Loading the Dataset
* The dataset is loaded from a CSV file using pandas.
* The BERT tokenizer is loaded using BertTokenizer.

### Splitting the Dataset for Training and Testing
* The dataset is split into training and testing sets using the train_test_split function.
* Training and testing datasets are created using the CustomDataset class.

### Setting Training Parameters
* Training parameters are defined using the TrainingArguments class.
* Training parameters include the output directory for saving results, the number of training epochs, batch size values, and other hyperparameters.

### Loading and Training the BERT Model
* A BERT-based classification model is loaded using the BertForSequenceClassification class.
* The model is set to classify into two classes (num_labels=2).
* The compute_metrics function is used to calculate the accuracy metric for the model.
* The Trainer class is used to train the model.
* The model is trained using the trainer.train() command.

#

Thank you,

Kagan Yavuz
