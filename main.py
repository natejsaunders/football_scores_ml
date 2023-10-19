from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import csv

def csv_to_data(csv_file):
    data = open(csv_file)
    csvreader = csv.reader(data)
    
    data_columns_heading = ["MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA"]#, "MaxCH", "MaxCD", "MaxCA", "AvgCH", "AvgCD", "AvgCA"]
    bi_columns_heading = ["AvgH", "AvgD", "AvgA"]
    label_columns_heading = ["FTHG", "FTAG"]


    headings = next(csvreader)

    data_columns = [headings.index(heading) for heading in data_columns_heading]
    label_columns = [headings.index(heading) for heading in label_columns_heading]
    bi_columns = [headings.index(heading) for heading in bi_columns_heading]

    data = []
    labels = []
    betting_info = []

    for row in csvreader:
        data.append([row[data_index] for data_index in data_columns])
        labels.append([row[label_index] for label_index in label_columns])
        betting_info.append([float(row[bi_index]) for bi_index in bi_columns])

    return data, labels, betting_info

scaler = StandardScaler()

training_data, training_labels, train_bi = csv_to_data('samples/2022_2023_season_pl.csv')
testing_data, testing_labels, test_bi = csv_to_data('samples/2021_2022_season_pl.csv')

scaler.fit(training_data)

training_data = scaler.transform(training_data)
testing_data = scaler.transform(testing_data)

training_hg = [score[0] for score in training_labels]
training_ag = [score[1] for score in training_labels]

testing_hg = [score[0] for score in testing_labels]
testing_ag = [score[1] for score in testing_labels]

hg_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 2), random_state=1, max_iter=10000)
ag_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 2), random_state=1, max_iter=10000)

hg_clf.fit(training_data, training_hg)
ag_clf.fit(training_data, training_ag)
print('Fitting done')

hg_result = hg_clf.predict(testing_data)
ag_result = ag_clf.predict(testing_data)

# BETTING SIMULATOR

money = 0
money_all = []
BET = 10

for i, result in enumerate(testing_labels):
    real_score = int(result[0]) - int(result[1])
    predict_score = int(hg_result[i]) - int(ag_result[i])

    if real_score == 0 and predict_score == 0:
        money += BET + round(BET * test_bi[i][1], 2)
    elif real_score < 0 and predict_score < 0:
        money += BET + round(BET * test_bi[i][2], 2)
    elif real_score > 0 and predict_score > 0:
        money += BET + round(BET * test_bi[i][0], 2)
    else:
        money -= BET

    money_all.append(money)

print(round(money, 2))

graph = plt.plot(range(len(testing_labels)), money_all)
plt.show()
