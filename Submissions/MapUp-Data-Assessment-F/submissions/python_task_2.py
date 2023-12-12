import pandas as pd
import numpy as np
from itertools import combinations

   
def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    df = pd.read_csv(df)
    
    #Identify unique id's from Start & End Column
    unique_ids = sorted(set(df['id_start']).union(df['id_end']))

    #No of unique id's
    matrix_size = len(unique_ids)

    # Create a matrix of length of id's
    distance_matrix = pd.DataFrame(index=range(1, matrix_size + 1), columns=range(1, matrix_size + 1))

    #Filling NA Values with 0
    distance_matrix = distance_matrix.fillna(0)

    #Assign name to index and column
    distance_matrix.index = unique_ids
    distance_matrix.columns = unique_ids

    #Filling the distance for given pair of id's

    for _,row in df.iterrows():
        start_id = row['id_start']
        end_id   = row['id_end']
        distance = row['distance']
        
        #print(start_id, end_id, distance)
        distance_matrix.loc[start_id, end_id] = distance
        distance_matrix.loc[end_id, start_id] = distance
        
    # Iterate through rows and columns
    for i in df['id_start']:
        for j in df['id_start']:
            if i == j:
                # If row index is the same as column name, set the distance to 0
                distance_matrix.at[i, j] = 0
            else:
                # Compute the distance as the sum of distances between rows i to j
                in_between_distances = df[(df['id_start'] >= min(i, j)) & (df['id_end'] <= max(i, j))]['distance']
                distance_matrix.at[i, j] = in_between_distances.sum()

    return distance_matrix


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    
    # Initialize an empty list to store the unrolled data
    unrolled_data = []

    # Iterate through the rows and columns of the distance matrix
    for row in df.index:
        for col in df.columns:
            # Skip entries where the row and column names are the same
            if row != col:
                # Append a dictionary with id_start, id_end, and distance
                unrolled_data.append({
                    'id_start': row,
                    'id_end': col,
                    'distance': df.loc[row, col]
                })

    # Create a new DataFrame from the unrolled data
    df = pd.DataFrame(unrolled_data)

    return df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
        # Filter rows for the specified reference value
    reference_rows = df[df['id_start'] == reference_id]

    # Calculate the average distance for the reference value
    average_distance = reference_rows['distance'].mean()

    # Calculate the threshold range (10% of the average distance)
    threshold_range = 0.1 * average_distance

    # Filter rows within the threshold range
    within_threshold = df[
        (df['distance'] >= average_distance - threshold_range) &
        (df['distance'] <= average_distance + threshold_range)
    ]

    # Get unique values from the 'id_start' column within the threshold
    df = within_threshold[['id_start']].drop_duplicates()

    # Sort the result_ids list
    df.sort_values(by='id_start', inplace=True)


    return df

def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    # rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Adding columns for each vehicle type with their respective rate coefficients
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    

    return df
