## import libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

## import files

df = pd.read_excel("./FinancialFraud.xlsx")
df.head()
df.drop(columns = ["t", "Mck"], inplace = True)
df

### Data Exloratory Data Analysis

df["fraud"].value_counts(normalize = True).to_frame()
sns.countplot(data = df, x = "RST",hue = "fraud", palette = 'Blues')

features = ['RST', 'CEO', 'BIG 4', "AUDITOR REPORT"]

n_rows = 2
n_cols = 2

fit, ax = plt.subplots(n_rows, n_cols, figsize = (n_cols*3.5, n_rows*3.5))

for r in range(0, n_rows):
    for c in range(0, n_cols):
        i = r*n_cols + c
        #index through loop through list "cols":
        if i < len(features):
            ax_i = ax[r,c]
            sns.countplot(data = df, x = features[i], hue = "fraud", palette = "Blues", ax = ax_i)
            ax_i.set_title(f"Figure{i+1}: fraud rate vs {features[i]}")
            ax_i.legend(title = "", loc = "upper right", labels = ["Not fraud", "fraud"])
        

#ax.flat[-1].set_visible(False)
plt.tight_layout()
plt.show()


def CorrelationMatrix(data):
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df.corr()
    sb.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right')


fig, ax = plt.subplots(figsize=(11, 9))
corr = df.corr()
sb.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')


