import pandas as pd
import numpy as np

def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here
    dff = pd.read_csv(df)
    
    df_1 = dff.pivot(index='id_1', columns='id_2', values='car')
    df_2 = df_1.fillna(0)
    return df_2


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here
    dff = pd.read_csv(df)

    #Listing given conditions
    conditions = [
    (dff['car']<=15),
    (dff['car']>15) & (dff['car']<=25),
    (dff['car']>25)]
    
    #Listing categories
    categories = ['low','medium','high']

    #Assigning value to the new_column['car_type']
    dff['car_type'] = np.select(conditions,categories)

    return dict(sorted(dff['car_type'].value_counts().to_dict().items()))


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
    dff = pd.read_csv(df)

    #index_of_bus_value
    return sorted(dff.index[dff['bus'] > (2*(dff['bus'].mean()))])
     
    return list(get_bus_indexes(dff))


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here
    dff = pd.read_csv(df)
        
    #Average value(mean) of truck for each route
    avg_truck_value_each_route =dff.groupby('route')['truck'].mean()

    #Filter route whr avg truck value > 7
    route = avg_truck_value_each_route[avg_truck_value_each_route>7].index.to_list()

    return list(sorted(route))


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here
    
    #Using verctorization
    matrix = dataset.map(lambda x:x*0.75 if x>20 else x*1.25 if x<=20 else x).round(1)
    

    return matrix



def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    df = pd.read_csv(df)
    
    # Converting startDay and endDay to categorical with R8 order
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['startDay'] = pd.Categorical(df['startDay'], categories=days_order, ordered=True)
    df['endDay'] = pd.Categorical(df['endDay'], categories=days_order, ordered=True)

    # Create a new column with the combined datetime for start and end
    df['startDatetime'] = pd.to_datetime(df['startDay'].astype(str) + ' ' + df['startTime'],format='%Y-%m-%d %H:%M:%S',errors='ignore')
    df['endDatetime'] = pd.to_datetime(df['endDay'].astype(str)+ ' ' + df['endTime'],format='%Y-%m-%d %H:%M:%S',errors='ignore')

    # Indicating if the time span is incorrect
    time_completeness = df.apply(lambda row: (row['startDatetime'] > row['endDatetime']), axis=1)

    # Group by (id, id_2) and check if any row has incorrect time span
    result = df.groupby(['id', 'id_2']).apply(lambda group: any(time_completeness.loc[group.index]))

    return pd.Series(result)
