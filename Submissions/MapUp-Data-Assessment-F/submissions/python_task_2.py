import pandas as pd
import numpy as np

    
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
        
    cols = distance_matrix.columns
    currentSum = 0

    for col_Idx,active_Col in enumerate(cols):
        if col_Idx!=len(cols)-1:
            currentSum=distance_matrix.iloc[col_Idx,col_Idx+1]
        else:
            break
           
        for next_Col_Idx in range(col_Idx+1,len(cols)):
            distance_matrix.iloc[col_Idx,next_Col_Idx]=currentSum
            currentSum+=distance_matrix.iloc[next_Col_Idx-1,next_Col_Idx]
            
    n = distance_matrix.shape[0] #Blank matrix with 0's
    
    upper_triangle = np.triu(distance_matrix.values, k=1)
    
    lower_triangle = upper_triangle.T
    
    distance_matrix.values[np.tril_indices(n, k=-1)] = lower_triangle[np.tril_indices(n, k=-1)]
    
    df = distance_matrix

    return df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here

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
