import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.layers import Dense, Input  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
import data

ACCURACY_SCORE = {}
PREDICTIONS = {}
MODELS = {
    'dt': 'Decision Tree Classifier',
    'rf': 'Random Forest Classifier',
    'svml': 'Linear Support Vector Machine',
    'svmr': 'RBF Support Vector Machine',
    'svms': 'Sigmoid Support Vector Machine',
    'svmp': 'Poly Support Vector Machine',
    'knn': 'K-Nearest Neighbor',
    'nbm': 'Naive Bayes MultinomialNB',
    'lr': 'Logistic Regression',
    'dnn': 'Deep Neural Network'
}

# Get the all the data
all_features_scaled, all_classes, all_features_minmax_scaled= data.preprocess_data('mammography_masses.data.txt')
all_predict_features_scaled, all_predicted_features_minmax_scaled = data.preprocess_prediction_data('predict_masses.data.txt')

'''
Create a single train/test split of our data.
Set aside 75% for training and 25% for testing
'''
np.random.seed(42)
training_inputs, testing_inputs, training_classes, testing_classes = train_test_split(all_features_scaled, all_classes, train_size=0.75, random_state=1)

# Use K-Fold cross validation to get a better measure of the model's accuracy (K=10)
def crossValScore(model, features):
    cvs = cross_val_score(model, features, all_classes, cv=10)
    return round(cvs.mean() * 100, 2)

'''
Decision Tree Classifier
Create a decision tree classifier and fit it to the training data
'''
def decisionTreeClassifierModel():
    model_key = 'dt'
    dt_clf = DecisionTreeClassifier(criterion='entropy')
    dt_accuracy_score = crossValScore(dt_clf, all_features_scaled)
    mammogram_predictor(dt_clf, model_key, dt_accuracy_score)

'''
Random Forest Classifier
Create a random forest classifier and fit it to the training data
'''
def RandomForestClassifierModel():
    model_key = 'rf'
    rf_clf = RandomForestClassifier(n_estimators=10, random_state=1)
    rf_accuracy_score = crossValScore(rf_clf, all_features_scaled)
    mammogram_predictor(rf_clf, model_key, rf_accuracy_score)
    
'''
Support Vector Machine (SVM)
Use an SVM with a linear kernel
'''
def supportVectorMachineLinearModel():
    model_key = 'svml'
    svm_linear = svm.SVC(kernel='linear', C=0.1)
    svm_linear_accuracy_score = crossValScore(svm_linear, all_features_scaled)
    mammogram_predictor(svm_linear, model_key, svm_linear_accuracy_score)

'''
Support Vector Machine (SVM)
Use an SVM with a rbf kernel
'''
def supportVectorMachineRbfModel():
    model_key = 'svmr'
    svm_rbf = svm.SVC(kernel='rbf', C=0.1)
    svm_rbf_accuracy_score = crossValScore(svm_rbf, all_features_scaled)
    mammogram_predictor(svm_rbf, model_key, svm_rbf_accuracy_score)

'''
Support Vector Machine (SVM)
Use an SVM with a sigmoid kernel
'''
def supportVectorMachineSigmoidModel():
    model_key = 'svms'
    svm_sigmoid = svm.SVC(kernel='sigmoid', C=0.1)
    svm_sigmoid_accuracy_score = crossValScore(svm_sigmoid, all_features_scaled)
    mammogram_predictor(svm_sigmoid, model_key, svm_sigmoid_accuracy_score)

'''
Support Vector Machine (SVM)
Use an SVM with a poly kernel
'''
def supportVectorMachinePolyModel():
    model_key = 'svmp'
    svm_poly = svm.SVC(kernel='poly', C=0.1)
    svm_poly_accuracy_score = crossValScore(svm_poly, all_features_scaled)
    mammogram_predictor(svm_poly, model_key, svm_poly_accuracy_score)

'''
K-Nearest Neighbours
User K-Nearest Neighbor with a K hyperparameter
Since K is tricky, we'll try different values ranging from 1 to 50
to see if K makes a substantial difference
'''
def kNearestNeighborsModel():
    model_key = 'knn'
    k_scores = {}
    for k in range(1, 50):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn_accuracy_score = crossValScore(knn, all_features_scaled)
        k_scores[k] = knn_accuracy_score

    k_highest = max(k_scores, key=k_scores.get)
    highest_accuracy = k_scores[k_highest]
    knn = KNeighborsClassifier(n_neighbors=k_highest)
    mammogram_predictor(knn, model_key, highest_accuracy)

'''
Naive Bayes Multinomial NB
Use Naive Bayes MultinomialNB
'''
def naiveBayesMultinomialModel():
    model_key = 'nbm'
    naive_clf = MultinomialNB()
    naive_bayes_accuracy_score = crossValScore(naive_clf, all_features_minmax_scaled)
    mammogram_predictor(naive_clf, model_key, naive_bayes_accuracy_score, nb=True)

'''
Logistic Regression
Use logistic regression
'''
def logisticRegressionModel():
    model_key = 'lr'
    lr_clf = LogisticRegression()
    lr_accuracy_score = crossValScore(lr_clf, all_features_scaled)
    mammogram_predictor(lr_clf, model_key, lr_accuracy_score)

'''
Neural Networks
'''
# Define the neural network with 4 feature inputs
# going into a 6-unit layer using relu activation.
# Then going into a 4-unit layer using a relu activation
# Then goes into an output layer with a sigmoid activation
def neuralNetworkModel():
    model_key = 'dnn'
    dnn = Sequential([
        Input(shape=(4,)),
        Dense(6, activation='relu'), 
        Dense(4, activation='relu'),                       
        Dense(1, activation='sigmoid')
    ])
    # Compile the model using the adam optimizer and binary cross-entropy loss
    dnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model
    dnn.fit(training_inputs, training_classes, epochs=100, batch_size=10, verbose=0)
    evaluation = dnn.evaluate(testing_inputs, testing_classes, verbose=0)
    dnn_accuracy_score = round(evaluation[1] * 100, 2)
    ACCURACY_SCORE['dnn'] = dnn_accuracy_score
    prediction = dnn.predict(all_predict_features_scaled)
    model_result = 'Benign' if prediction[0] == 0 else 'Malignant'
    PREDICTIONS[model_key] = model_result
    print(f"\n{MODELS[model_key]} model")
    print(f"Prediction: {model_result}")
    print(f"Accuracy: {dnn_accuracy_score}% ")
    
# Return mammography prediction
def mammogram_predictor(model, key, accuracy, nb=False):
    dataset = (all_features_minmax_scaled, all_classes, all_predicted_features_minmax_scaled) if nb else (training_inputs, training_classes, all_predict_features_scaled)
    x_train, y_train, predict_data = dataset
    model.fit(x_train, y_train)
    prediction = model.predict(predict_data)
    ACCURACY_SCORE[key] = accuracy
    model_result = 'Benign' if prediction[0] == 0 else 'Malignant'
    PREDICTIONS[key] = model_result
    print(f"\n{MODELS[key]} model")
    print(f"Prediction: {model_result}")
    print(f"Accuracy: {accuracy}% ")


def main():
    print('\nMammogram mass prediction results:')
    decisionTreeClassifierModel()
    RandomForestClassifierModel()
    supportVectorMachineLinearModel()
    supportVectorMachineRbfModel()
    supportVectorMachineSigmoidModel()
    supportVectorMachinePolyModel()
    kNearestNeighborsModel()
    naiveBayesMultinomialModel()
    logisticRegressionModel()
    neuralNetworkModel()
    print('\n--------------------------------------------\n')
    best_model = max(ACCURACY_SCORE, key=ACCURACY_SCORE.get)
    best_prediction = PREDICTIONS[best_model]
    print(f"Best Model: {MODELS[best_model]}")
    print(f"Best Prediction: The mammographic mass is {best_prediction}")
    print(f"Highest Accuracy: {ACCURACY_SCORE[best_model]}%\n")

if __name__ == "__main__":
    main()


