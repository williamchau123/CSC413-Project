import pandas as pd

df = pd.read_csv('data.csv')

answer = df['Answer'].values
label = df['Label (AI0/ Human1)'].values

train_loader = []
for i in range(len(answer)):
    one_data = [answer[i],label[i]]
    train_loader.append(one_data)
print(train_loader)


