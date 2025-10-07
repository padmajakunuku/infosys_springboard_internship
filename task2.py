#read a student.csv file and find avg marks per subject
import pandas as pd
data=pd.read_csv(student.csv)
avg_per_subject = data.drop("Name", axis=1).mean()
print("Average marks per subject: ")
print(avg_per_subject)