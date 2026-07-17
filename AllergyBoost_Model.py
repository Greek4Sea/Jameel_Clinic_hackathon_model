from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score # for getting results
import joblib

#import the libraries we need: pandas for loading the data from the csv
#and numpy for clean calculations without the need for "for loops"
#and sklearn for the HistGradientBoostingClassifier which is the model we will train on

TARGET = "Seen_Allergist"
STATE = 42 #seed to lock the random number generator
#to load the data

root = Path(__file__).resolve().parent
trainingdata = pd.read_csv(root / "question4" / "train.csv")
testingdata  = pd.read_csv(root / "question4" / "test_X.csv")

#dropping column ID because it is useless for training
features = trainingdata.drop(columns=["id", TARGET])
test_features = testingdata[features.columns]#load test data to convert to numbers

#convert everything to numbers in test and train for them to be the same
combined = pd.concat([features, test_features], keys=["train", "test"])
for i in combined.columns:
    if not pd.api.types.is_numeric_dtype(combined[i]):
        nums, _ = pd.factorize(combined[i])
        nums = nums.astype(float) #convert all to floats
        for t in range(len(nums)): #if there is a missing value then it replaces it with NaN, since the factorizing will turn it into -1
            if nums[t] == -1:
                nums[t] = np.nan
        combined[i] = nums

combined = combined.fillna(-1000)#this fills in NaN with a number so stupid that the model learns to not account for it
X = combined.loc["train"].to_numpy() #these two seperate the data we had factorized into its training and testing sets since they had gotten links
X_test = combined.loc["test"].to_numpy()
Y = trainingdata[TARGET].astype(int).to_numpy() #answers
# we will split the training data into a randomized 80 - 20 split
random_data = np.random.RandomState(STATE)
shuffle_patients_index = random_data.permutation(len(X))
N_validation = len(X) // 5 #splitting into 20% for validation
#break up the index numbers into validation and training
validation_index = shuffle_patients_index[:N_validation]
training_index = shuffle_patients_index[N_validation:]


#split up the training data into its x and y
x_training = X[training_index]
y_training = Y[training_index]
#split up the validation data into its x and y
x_validation = X[validation_index]
y_validation = Y[validation_index]

model = HistGradientBoostingClassifier(
    learning_rate=0.1, #how small the steps are
    max_iter=500,# how many itirations
    early_stopping=True,# stops overfitting from noise
    random_state=STATE,# keeps the same randomized state
)

model.fit(x_training, y_training) #trains the model
# real prediations on the test data
test_probs = model.predict_proba(X_test)[:, 1]
#saving model stuff
joblib.dump(
    {
        "model": model,
        "x_validation": x_validation,
        "y_validation": y_validation,
        "X_test": X_test,
        "feature_names": list(combined.columns),
        "base_rate": y_training.mean(),
        "test_ids": testingdata["id"].to_numpy(),
        "test_probs": test_probs,
    },
    root / "allergyboost.joblib",
)
#saving the real test results
submission = pd.DataFrame({"id": testingdata["id"], "prediction": test_probs})
submission.to_csv(root / "question4_predictions.csv", index=False)