import pandas as pd

cell_id = "Cell ID"
number_of_puncta = "Number of puncta"


def create_dataframe():
    return pd.DataFrame(index=[cell_id], columns=['x', 'y', number_of_puncta])


def dataframe_to_csv(df, filename):
    df.to_csv(filename)


def dataframe_to_excel(df, filename):
    df.to_excel(filename)
