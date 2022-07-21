import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

path = r'/home/qqq/PycharmProjects/qi/experiment_results/1h_dag_best/results/'

GOOD_IDS = [3, 6, 11, 12, 14, 16, 18, 22, 23, 28, 29, 31, 32, 37, 38, 46, 50, 54, 151, 182, 188, 300, 307, 458, 1049,
            1050, 1053, 1063, 1462, 1464, 1468, 1478, 1489, 1494, 1497, 1501, 1510, 1590, 4534, 4538, 23381, 40499,
            40668, 40670, 40701, 40966, 40975, 40979, 40982, 40983, 40984, 40994, 41027][:3]

for dir in os.listdir(path):
    if int(dir) in GOOD_IDS:
        with open(osp.join(path, dir, '1', 'train_history.csv')) as f:
            lines = []
            print(osp.join(path, dir, '1', 'train_history.csv'))
            for line in f:
                temp = line.split(',')
                temp[-1] = temp[-1][:-1]  # remove \n
                temp[0] = float(temp[0])  # convert score to float

                # convert the input ids into integers
                for i in range(1, len(temp), 2):
                    temp[i] = int(temp[i])

                lines.append(temp)
            for i in range(len(lines)):
                lines[i][0] = float(lines[i][0])

            # prepare for plotting
            points = list(range(len(lines)))

            # normalize scores to color the points when plotting
            sorted_lines = sorted(lines, key=lambda x: x[0], reverse=True)
            max_score = float(sorted_lines[0][0])
            min_score = float(sorted_lines[-1][0])

            scores = [line[0] for line in lines]
            scores = np.array(scores)
            normalized_scores = (scores - min_score) / (max_score - min_score)

            plt.scatter(points, scores, c=normalized_scores, cmap='coolwarm')
            plt.xlabel('Model Index')
            plt.ylabel('Score')
            plt.title('Score across model')
            plt.show()
