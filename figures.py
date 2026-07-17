from pathlib import Path
import numpy as np
import joblib
import matplotlib.pyplot as plt # plots
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, roc_curve #results
from sklearn.inspection import permutation_importance # feature weights

STATE = 42
MODEL_NAME = "AllergyBoost"
#to load the saved model stuff

root = Path(__file__).resolve().parent
bundle = joblib.load(root / "allergyboost.joblib") #load model

#unpack bundle
model = bundle["model"]
x_val = bundle["x_validation"]
y_val = bundle["y_validation"]
feature_names = bundle["feature_names"]
base_rate = bundle["base_rate"]

model_probs = model.predict_proba(x_val)[:, 1] #real models guesses
baseline_probs = np.full(len(y_val), base_rate) #baseline


def metrics(y_true, probs):
    return {
        "log_loss": log_loss(y_true, probs),
        "accuracy": accuracy_score(y_true, probs > 0.5),
        "auc": roc_auc_score(y_true, probs),
    }

real_model = metrics(y_val, model_probs) #real model score
base_model = metrics(y_val, baseline_probs) #baseline score


print(f"{MODEL_NAME:12s} log_loss {real_model['log_loss']:.4f}  acc {real_model['accuracy']:.4f}  auc {real_model['auc']:.4f}")
print(f"{'Baseline':12s} log_loss {base_model['log_loss']:.4f}  acc {base_model['accuracy']:.4f}  auc {base_model['auc']:.4f}")

#figure one
fig, ax = plt.subplots(figsize=(8, 5))

labels = ["Log loss\n(lower better)", "Accuracy\n(higher better)", "AUC\n(higher better)"]
base_validation = [base_model["log_loss"], base_model["accuracy"], base_model["auc"]]
model_validation = [real_model["log_loss"], real_model["accuracy"], real_model["auc"]]

x = np.arange(len(labels))
w = 0.35

ax.bar(x - w/2, base_validation, w, label="Baseline (base rate)", color="#b0b0b0")
ax.bar(x + w/2, model_validation, w, label=MODEL_NAME, color="#2c7fb8")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title(f"{MODEL_NAME} vs Baseline")
ax.legend()


for xi, (b, mv) in enumerate(zip(base_validation, model_validation)):
    ax.text(xi - w/2, b + 0.01, f"{b:.3f}", ha="center", fontsize=9)
    ax.text(xi + w/2, mv + 0.01, f"{mv:.3f}", ha="center", fontsize=9)

fig.tight_layout()
fig.savefig(root / "figure_1.png", dpi=150)
print("saved figure_1.png")


#FIGURE 2 feature weights
imp = permutation_importance(model, x_val, y_val, scoring="accuracy", n_repeats=10, random_state=STATE)
order = imp.importances_mean.argsort()[::-1][:15]
names = [feature_names[i] for i in order]
validation = imp.importances_mean[order]
error = imp.importances_std[order]

fig, ax = plt.subplots(figsize=(8, 6))
ypos = np.arange(len(names))[::-1]
ax.barh(ypos, validation, xerr=error, color="#2c7fb8")
ax.set_yticks(ypos)
ax.set_yticklabels(names)
ax.set_xlabel("Drop in accuracy when feature is shuffled")
ax.set_title(f"{MODEL_NAME}: top 15 feature weights")

fig.tight_layout()
fig.savefig(root / "Figure_2.png", dpi=150)
print("saved Figure_2.png")