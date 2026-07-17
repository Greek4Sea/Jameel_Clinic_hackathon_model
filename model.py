from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

#import the libraries we need: pandas for loading the data from the csv
#and numpy for clean calculations without the need for "for loops"
#and sklearn for the DecisionTreeRegressor which is the model we will train on

TARGET = "Seen_Allergist"

#to load the data

root = Path(__file__).resolve().parents[1]
trainingdata = pd.read_csv(root / "question4" / "train.csv")
testingdata  = pd.read_csv(root / "question4" / "test_X.csv")

#dropping column ID because it is useless for training
features = trainingdata.drop(columns=["id", TARGET])
test_features = testingdata[features.columns]#load test data to convert to numbers

#convert everything to numbers in test and train for them to be the same
combined = pd.concat([features, test_features], keys=["train", "test"])
for i in combined.columns:
    if not pd.api.types.is_numeric_dtype(combined[i]):
        nums, _ = pd.factorize(combined[col])
        nums = nums.astype(float)
        for i in range(len(nums)): #if there is a missing vaue then it replaces it with NaN, since the factorizing will turn it into -1
            if nums[i] == -1:
                nums[i] = np.nan






