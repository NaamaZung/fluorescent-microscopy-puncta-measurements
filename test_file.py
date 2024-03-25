import pandas as pd
import numpy as np

import IO
from image_processing import calculate_puncta_per_cell_in_image


def df_to_list(output_filename):
    df = pd.read_csv(output_filename)
    number_of_puncta = IO.number_of_puncta
    column = df[number_of_puncta].iloc[1:]
    return column.tolist()


def mean_sd_variance_of_the_puncta(number_of_puncta):
    total_sum = sum(number_of_puncta)
    mean = total_sum / len(number_of_puncta)
    variance = np.var(number_of_puncta)
    std_dev = np.std(number_of_puncta)
    return mean, variance, std_dev


def test_mean_variance_and_sd(generate_data=True):
    if generate_data:
        calculate_puncta_per_cell_in_image("images/LD_Control.tif", "output.csv")
    list_of_puncta = df_to_list("output.csv")
    mean, variance, std_dev = mean_sd_variance_of_the_puncta(list_of_puncta)
    if mean < 1:
        print(f"the mean is: '{mean}' the mean is reasonable")
    else:
        print(f"the mean is: '{mean}' Something is wrong with the mean")
    if variance < 5:
        print(f"the variance is: '{variance}' the variance is ok")
    else:
        print(f"the variance is: '{variance}' Something is wrong with the variance")
    if std_dev < 5:
        print(f"the standard_deviation is: '{std_dev}' the standard deviation is ok")
    else:
        print(f"the standard deviation is: '{std_dev}' Something is wrong with the standard deviation")


def test_negative_numbers(generate_data=True):
    if generate_data:
        calculate_puncta_per_cell_in_image("images/LD_Control.tif", "output.csv")
    list_of_puncta_number = df_to_list("output.csv")
    for number in list_of_puncta_number:
        if number < 0:
            print("Something is wrong, there is a negative number")
    else:
        print("All numbers are positive")


def main():
    calculate_puncta_per_cell_in_image("images/LD_Control.tif", "output.csv")

    test_mean_variance_and_sd(False)
    test_negative_numbers(False)


if __name__ == "__main__":
    main()
