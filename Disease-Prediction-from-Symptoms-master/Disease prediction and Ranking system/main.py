# Import Dependencies
import yaml
from joblib import dump, load
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score,recall_score, confusion_matrix
# Naive Bayes Approach
from sklearn.naive_bayes import MultinomialNB
# Trees Approach
from sklearn.tree import DecisionTreeClassifier
# Ensemble Approach
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
# from skmultilearn.problem_transform import BinaryRelevance  

class DiseasePrediction:
    # Initialize and Load the Config File
    def __init__(self, model_name=None):
        # Load Config File
        try:
            with open('./config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print("Error reading Config file...")

        # Verbose
        self.verbose = self.config['verbose']
        # Load Training Data
        self.train_features, self.train_labels, self.train_df = self._load_train_dataset()
        # Load Test Data
        self.test_features, self.test_labels, self.test_df = self._load_test_dataset()
        # Feature Correlation in Training Data
        self._feature_correlation(data_frame=self.train_df, show_fig=False)
        # Model Definition
        self.model_name = model_name
        # Model Save Path
        self.model_save_path = self.config['model_save_path']
        self.severity_data = self._add_severity_index()

    def _add_severity_index(self):
        """
        Adds a severity index to the DataFrame based on the frequency of each disease.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the patient records.
        disease_column (str): The name of the column in df that contains the disease names.

        Returns:
        pandas.DataFrame: A copy of df with a new column 'Severity Index' added.
        """
        # load the dataset
        df = pd.read_csv(self.config['dataset']['patient_data_path'], header=0)
        # Calculate the count of each disease
        disease_counts = df['prognosis'].value_counts()
        # print("disease_counts",disease_counts)
        # Normalize counts to create a severity index between 0 and 1
        severity_index = disease_counts / disease_counts.max()
        
        # Map the severity index back to the original DataFrame
        df['Severity Index'] = df['prognosis'].map(severity_index)

        severity_df = pd.DataFrame({
            'Disease': severity_index.index,
            'Severity Index': severity_index.values
        })
        #  save the severity index associated with each disease(41 diseases)
        severity_df.to_csv(self.config['severity_save_path'], index=False)
        return severity_df

    # Function to Load Train Dataset
    def _load_train_dataset(self):
        df_train = pd.read_csv(self.config['dataset']['training_data_path'], header=0)
        cols = df_train.columns
        cols = cols[:-2]
        train_features = df_train[cols]
        train_labels = df_train['prognosis']

        # Check for data sanity
        assert (len(train_features.iloc[0]) == 132)
        assert (len(train_labels) == train_features.shape[0])
        
        df_train = df_train[cols]
        if self.verbose:
            print("Length of Training Data: ", df_train.shape)
            print("Training Features: ", train_features.shape)
            print("Training Labels: ", train_labels.shape)
        return train_features, train_labels, df_train

    # Function to Load Test Dataset
    def _load_test_dataset(self):
        df_test = pd.read_csv(self.config['dataset']['test_data_path'], header=0)
        cols = df_test.columns
        cols = cols[:-2]
        test_features = df_test[cols]
        test_labels = df_test['prognosis']

        # Check for data sanity
        assert (len(test_features.iloc[0]) == 132)
        assert (len(test_labels) == test_features.shape[0])

        df_test = df_test[cols]
        if self.verbose:
            print("Length of Test Data: ", df_test.shape)
            print("Test Features: ", test_features.shape)
            print("Test Labels: ", test_labels.shape)
        return test_features, test_labels, df_test

    # Features Correlation
    def _feature_correlation(self, data_frame=None, show_fig=False):
        # Get Feature Correlation
        corr = data_frame.corr()
        sn.heatmap(corr, square=True, annot=False, cmap="YlGnBu")
        plt.title("Feature Correlation")
        plt.tight_layout()
        if show_fig:
            plt.show()
        plt.savefig('feature_correlation.png')

    # Dataset Train Validation Split
    def _train_val_split(self):
        X_train, X_val, y_train, y_val = train_test_split(self.train_features, self.train_labels,
                                                          test_size=self.config['dataset']['validation_size'],
                                                          random_state=self.config['random_state'])

        if self.verbose:
            print("Number of Training Features: {0}\tNumber of Training Labels: {1}".format(len(X_train), len(y_train)))
            print("Number of Validation Features: {0}\tNumber of Validation Labels: {1}".format(len(X_val), len(y_val)))
        return X_train, y_train, X_val, y_val

    # Model Selection
    def select_model(self):
        if self.model_name == 'mnb':
            self.clf = MultinomialNB()
        elif self.model_name == 'decision_tree':
            self.clf = DecisionTreeClassifier(criterion=self.config['model']['decision_tree']['criterion'])
        elif self.model_name == 'random_forest':
            self.clf = RandomForestClassifier(n_estimators=self.config['model']['random_forest']['n_estimators'])
        elif self.model_name == 'gradient_boost':
            self.clf = GradientBoostingClassifier(n_estimators=self.config['model']['gradient_boost']['n_estimators'],
                                                  criterion=self.config['model']['gradient_boost']['criterion'])
        return self.clf

    # ML Model
    def train_model(self):
        # Get the Data
        X_train, y_train, X_val, y_val = self._train_val_split()
        classifier = self.select_model()
        # Training the Model
        classifier = classifier.fit(X_train, y_train)
        # Trained Model Evaluation on Validation Dataset
        confidence = classifier.score(X_val, y_val)
        # Validation Data Prediction
        y_pred = classifier.predict(X_val)
        # Model Validation Accuracy
        accuracy = accuracy_score(y_val, y_pred)
        # Model Confusion Matrix
        conf_mat = confusion_matrix(y_val, y_pred)
        # Model Classification Report
        clf_report = classification_report(y_val, y_pred)
        # Model Cross Validation Score
        score = cross_val_score(classifier, X_val, y_val, cv=3)

        if self.verbose:
            print('\nTraining Accuracy: ', confidence)
            print('\nValidation Prediction: ', y_pred)
            print('\nValidation Accuracy: ', accuracy)
            print('\nValidation Confusion Matrix: \n', conf_mat)
            print('\nCross Validation Score: \n', score)
            print('\nClassification Report: \n', clf_report)

        # Save Trained Model
        # with open('./saved_model/random_forest.pkl', 'wb') as file:
        # pickle.dump(model, file)
        with open(str(self.model_save_path + self.model_name + ".pkl"), 'wb') as file:
            pickle.dump(classifier, file)
        # pickle.dump(classifier, str(self.model_save_path + self.model_name + ".pkl"))

    # Assume 'model' is your trained model and 'features' is your input features
    def predict_and_rank_diseases(self, model, features):
        # Get probabilities of all classes (diseases)
        print("features", np.array(model.predict_proba(features).flat))
        class_probabilities = np.array(model.predict_proba(features).flat)  # Assuming a single input for prediction
        # Get class labels (diseases names) if you have them in the correct order
        disease_names = model.classes_  # This depends on your model

        # Create a sorted list of (disease, probability) tuples
        ranked_diseases = sorted(zip(disease_names, class_probabilities), key=lambda x: x[1], reverse=True)

        # Optionally, format the output as a list of strings for better readability
        formatted_output = [f"{disease}: {prob:.2%}" for disease, prob in ranked_diseases]

        return formatted_output

    # Function to Make Predictions on Test Data
    def make_prediction(self, saved_model_name=None, test_data=None):
        try:
            # Load Trained Model
            # clf = pickle.load(str(self.model_save_path + saved_model_name + ".pkl"))
            with open(str(self.model_save_path + saved_model_name + ".pkl"), 'rb') as file:
                clf = pickle.load(file)
        except Exception as e:
            print("Model not found...")

        if test_data is not None:
            predictions = self.predict_and_rank_diseases(clf, test_data)
            result = clf.predict(test_data)
            return result
        else:
            result = clf.predict(self.test_features)
            predictions = self.predict_and_rank_diseases(clf, self.test_features)
        accuracy = accuracy_score(self.test_labels, result)
        f1 = f1_score(self.test_labels, result, average='weighted')  # Use 'weighted' for imbalanced classes
        recall = recall_score(self.test_labels, result,average='weighted')
        precision = precision_score(self.test_labels, result, average = "weighted")
        conf_mat = confusion_matrix(self.test_labels, result)
        clf_report = classification_report(self.test_labels, result)
        sn.heatmap(conf_mat, annot=True, fmt="d",
            xticklabels=clf.classes_, yticklabels=clf.classes_)

        plt.title('Confusion Matrix')
        plt.ylabel('Actual Labels')
        plt.xlabel('Predicted Labels')
        plt.show()
        return accuracy,f1, precision, recall, clf_report, predictions, conf_mat


if __name__ == "__main__":
    # Load your dataset
    # df = pd.read_csv("Disease-Prediction-from-Symptoms-master/dataset/patient_dataset.csv")

    # # Assuming the column with disease names is labeled 'Disease'
    # df_with_severity = add_severity_index(df, 'Disease')

    # Display the first few rows to verify
    # print(df_with_severity.head())

    # Model Currently Training
    current_model_name = 'random_forest'
    # other models random_forest, decision_tree,mnb, gradient_boost
    # Instantiate the Class
    dp = DiseasePrediction(model_name=current_model_name)
    data =dp._add_severity_index()
    # Train the Model
    dp.train_model()
    # Get Model Performance on Test Data
    test_accuracy,f1, precision, recall, classification_report, predictions, conf_mat = dp.make_prediction(saved_model_name=current_model_name)
    print("model name:", current_model_name,"Model Test Accuracy: ", test_accuracy)
    print("f1",f1, "precison",precision, "recall",recall)
    print("Test Data Classification Report: \n", classification_report)
    print("predictions:", predictions) 
   