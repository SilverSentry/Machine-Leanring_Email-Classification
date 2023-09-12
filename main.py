import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("emails.csv")
df.head(20)

print(df.isnull().sum())
print()
df.shape

df.describe()

# df.corr()

X = df.iloc[:, 1:3001]
# X

Y = df.iloc[:, -1].values
# Y

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.25)

mnb = MultinomialNB(alpha=1.9)
mnb.fit(train_x, train_y)
y_pred1 = mnb.predict(test_x)
print("Accuracy Score for Naive Bayes : ", accuracy_score(y_pred1, test_y))

svc = SVC(C=1.0, kernel='rbf', gamma='auto')
svc.fit(train_x, train_y)
y_pred2 = svc.predict(test_x)
print("Accuracy Score for SVC : ", accuracy_score(y_pred2, test_y))

rfc = RandomForestClassifier(n_estimators=100, criterion='gini')
rfc.fit(train_x, train_y)
y_pred3 = rfc.predict(test_x)
print("Accuracy Score of Random Forest Classifier : ", accuracy_score(y_pred3, test_y))

preset1 = test_x.iloc[[1]]
preset2 = test_x.iloc[[376]]
preset3 = test_x.iloc[[928]]

# gui
window = tk.Tk()
window.geometry("400x300+600+200")
window.configure()
window.title("Prediction Menu")

cip = tk.Label(window, text="Custom Input: ")
cip.place(x=20, y=20)
cip_var = tk.StringVar()
cip_entry = ttk.Entry(window, width=30, textvariable=cip_var)
cip_entry.place(x=150, y=20)

preset = tk.Label(window, text="Preset Input: ")
preset.place(x=20, y=130)
preset_var = tk.StringVar()
presets = ttk.Combobox(window, width=14, textvariable=preset_var, state='readonly')
presets['values'] = ("preset1", "preset2", "preset3")
presets.current(0)
presets.place(x=150, y=130)


def assign():
    inputval = preset_var.get()
    if inputval == "preset1":
        a = preset1
        outputvalue = svc.predict(a)
        submit(inputval, outputvalue)
    if inputval == "preset2":
        a = preset2
        outputvalue = svc.predict(a)
        submit(inputval, outputvalue)
    if inputval == "preset3":
        a = preset3
        outputvalue = svc.predict(a)
        submit(inputval, outputvalue)


def submit(a, b):
    messagebox.showinfo('Resulting Prediction', "Prediction of %s : %s" % (a, b))


asg = tk.Button(window, text="Submit", bg="teal", foreground="white", command=assign)
asg.place(x=170, y=250)
window.mainloop()
