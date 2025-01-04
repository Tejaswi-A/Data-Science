# Disease Prediction from Symptoms
This project leverages machine learning algorithms to predict possible diseases based on symptoms inputted by a user. It's designed to demonstrate the capabilities of different machine learning techniques in a healthcare context.

### Algorithms Used
We explore several machine learning models to understand their effectiveness in disease prediction:

Naive Bayes: A simple probabilistic classifier based on applying Bayes' theorem.
Decision Tree: A model that uses a tree-like graph of decisions and their possible consequences.
Random Forest: An ensemble of decision trees, typically trained via the bagging method.
Gradient Boosting: A method that builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions.

### Dataset Overview
Kaggle Dataset
The primary dataset includes 133 columns: 132 representing symptoms and one for the diagnosis (prognosis). It can be downloaded using the following link:

Kaggle Disease Prediction Dataset

### Directory Structure
css
Copy code
|_ dataset/
         |_ training_data.csv
         |_ test_data.csv
|_ saved_model/
         |_ [pre-trained models]
|_ main.py
|_ infer.py
|_ demo.ipynb

## Setup and Usage
Before using this project, ensure you have Python installed. This project is designed to run in Visual Studio Code, which can be downloaded and installed from Visual Studio Code. After installing Visual Studio Code, you will need to install the Python extension and connect to the Python kernel to run Jupyter notebooks.

#### Installing Dependencies
First, install all necessary Python libraries by running the following command in your terminal or command prompt:

Copy code
    `pip install -r requirements.txt`

 Interactive Demo in Jupyter

First run mai.py file to save the trained model.

To explore an User Interface version of the model predictions:

Open Visual Studio Code.
Open the terminal in Visual Studio Code (Terminal > New Terminal).
Navigate to the project directory.
Run the following command to open the Jupyter Notebook:

    `jupyter notebook demo.ipynb` (or) Use the Run all option given in the notebook at top left to run the entire code.
    One the code is run, Then you can use the interface from output log or just go to the local host URL provided in outputlog to enter the symptoms in browser through URL.

Follow the instructions in the notebook for an interactive demonstration.

Standalone Demo
To run the model on a set dataset or custom inputs without using a notebook:

    `python infer.py`

*Once the user enters the symptoms and submit it, List of predicted diseases appear in resuls section. user can click on flag button to save the log. the saved logs can be saved in log.csv file in Flagged folder.*

**NOTE:** ***This project is for demonstration purposes only. For any medical symptoms or conditions, please consult a healthcare professional.***
