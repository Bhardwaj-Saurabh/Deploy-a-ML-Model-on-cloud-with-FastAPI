import pandas as pd
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


def data_cleaning(df, output_file):

    df_copy = df.copy()
    logging.info(f"Replacing  ' ?' with None value...")
    df_copy = df_copy.replace(' ?', None)

    # Remove white spaces from column names
    df_copy.columns = df_copy.columns.map(lambda x: x.strip())

    logging.info(f"Dropping NaN values from the dataset...")
    df_copy = df_copy.dropna()

    logging.info(f"Saving cleaned data to {output_file}...")
    df_copy.to_csv(output_file, index=False)
    logging.info("Basic Cleaning Completed...")

    return df_copy
