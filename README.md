# attr-pred-model-py
Attrition prediction model using Python script.

Here, we will see an attrition prediction model using a python notebook script.

We will use a generally available sample data from the internet. So, any appropriate input dataset can be used for this activity.

First, we will load the required modules and dataset in python notebook.

Then, we will do some basic descriptive analysis.

We will also remove unwanted features and retain only the required features.

We will do some pre-processing, like encoding descriptive fields and then scaling the numeric fields.

Mainly, we will create train and test splits of input dataset using 25% split ratio.

Finally, we will create two models to predict the attrition using Logistic Regression model as base model and will also compare it with an advanced Random Forest Classifier model.

To compare the performance of the models, we will use the metrics like - accuracy, confusion matrix, precison and recall, and select our best performing model to move to production. However, we should remember that a best trained model might not be a best performing model in real situations after deployment.

Generally, we can also withhold some 15% of the input dataset as validation dataset which should not be used for training and testing the models. It should only be used to evaluate the actual performance once the model is in production.

We can improve the model by performing feature selection, feature engineering and paramater tuning. It is also important to focus on model explainability to show how the model works and also to get domain specific inputs and suggestions to improve the model,

Note that, its a building a machinle learning model is not a one time project. It's a continuous process cycle which requires tracking the model's real world performance and to continuously enhance the model.

The Python script (.py format after converting from .pynb notebook format - to hide the results) is provided in a separate Model.py file.
