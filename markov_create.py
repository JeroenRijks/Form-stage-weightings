import numpy as np
import csv
stage_options = [[1, 15], [2, 15], [3, 13, 15], [4, 5, 6, 13, 15], [7, 9, 15], [9, 15], [8, 9, 15], [9, 15], [9, 15],
                 [10, 13, 15], [11, 15], [12, 15], [13, 15], [14, 15], [15], [15]]
stage_totals = [[0,0], [0, 0], [0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0], [0, 0, 0], [0, 0], [0, 0],
                [0, 0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0], [0]]
normalized_array = [[0,0], [0, 0], [0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0], [0, 0, 0], [0, 0], [0, 0],
                [0, 0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0], [0]]

markov_matrix = np.zeros((16, 16))


def tally_index(stage_num, option_index):
    stage_totals[stage_num][option_index] += 1
    return True


def analyze_col(stage_num, row):
    for option_index, stage_index in enumerate(stage_options[stage_num]):
        if row[stage_index] == 1:
            tally_index(stage_num, option_index)
            break
    return True


def analyze_row(row):
    for stage_num, stage_val in enumerate(row):
        if stage_val == 1:
            analyze_col(stage_num, row)
    return True


with open('raw_casetracker_data.csv', 'r') as csvfile:
    data_reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in data_reader:
        analyze_row(row)


def normalize_array(array):
    array_sum = sum(array)
    output_array = np.zeros(len(array))
    for k, v in enumerate(array):
        output_array[k] = float(v)/array_sum
    return output_array


def markov_maker():
    for stage_num, array in enumerate(stage_totals):
        try:
            norm_array=normalize_array(array)
            for opt_key, opt_val in enumerate(norm_array):
                markov_matrix[stage_num][stage_options[stage_num][opt_key]] = opt_val
        except ZeroDivisionError:
            pass


markov_maker()
print markov_matrix
# csv.writer(markov_matrix, )
np.savetxt("markov_matrix.csv", markov_matrix, delimiter=",")
