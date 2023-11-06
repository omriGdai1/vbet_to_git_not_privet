import pandas as pd
import numpy as np
import pickle
import boto3
from io import BytesIO

print("start")
# # Created for aggregating model 2 ( i.e. next week and next month)

# # This code can run on a container and get it through s3 or locally and then needs to be downloaded
 

# with open('/Users/omrilapidot/Vbet_adjusted_data/dataframes.pkl', 'rb') as f:
#     dataframes = pickle.load(f)
# print("Dataframes loaded.")


# # Download and read from s3
s3_client = boto3.client('s3')
# Download the pickle file from S3
bucket_name, key_dataframes, key_dataframes_relevant_clients = "vbet-adjusted-data", "dataframes.pkl", "relevant_clients.pkl"
response = s3_client.get_object(Bucket=bucket_name, Key=key_dataframes)
pickle_bytes = response['Body'].read()

# Load the pickle file into a DataFrame
dataframes = pickle.loads(pickle_bytes)

response = s3_client.get_object(Bucket=bucket_name, Key=key_dataframes_relevant_clients)
pickle_bytes = response['Body'].read()

# Load the pickle file
relevant_clients = pickle.loads(pickle_bytes)


# one_year_ago = pd.Timestamp.now() - pd.Timedelta(days=365)
def create_agg_dict_for_ClientSession(df, columns_to_exclude=[]):
    agg_dict = create_agg_dict(df, columns_to_exclude)
    mode_and_nunique_cols = ['DeviceId', 'Product', 'LogoutSource', 'LoginType']
    for col in mode_and_nunique_cols:
        if col not in columns_to_exclude:
            agg_dict[col] = ['nunique', lambda x: x.mode()[0] if not x.mode().empty else np.nan]
    agg_dict["Token"] = ['count']
    return agg_dict


def create_agg_dict(df, columns_to_exclude):
    #  Create aggregation mapping for each table
    agg_dict = {}
    for col, dtype in df.dtypes.items():
        if dtype in ['float64', 'int64']:
            agg_dict[col] = ['sum', 'count']
        elif dtype == 'bool':
            agg_dict[col] = ['sum', 'mean']

    correlated_columns = remove_highly_correlated_columns(df)
    for col in list(set(columns_to_exclude + correlated_columns)):
        agg_dict.pop(col, None)
    return agg_dict


def aggregate_data(df,
                   agg_dict,
                   time_window_timedeltas,
                   current_time=None,
                   additional_suffix="",
                   time_col='TimestampHour'):
    # Create aggregation for each table per mapping and per timestamp_dict
    
    if current_time is None:
        current_time = pd.Timestamp.now()

    aggregated_data_dict = {}
    # Create aggregation per time window
    for window_name in time_window_timedeltas.keys():
        window_size = time_window_timedeltas[window_name]
        window_start = current_time - window_size
        window_start = window_start.tz_localize(df[time_col].iloc[0].tz)

        # Filter out all data outside of time window
        aggregated_data = df[
            (df[time_col] >= window_start) & (df[time_col] < current_time.tz_localize(df[time_col].iloc[0].tz))]
        aggregated_data = aggregated_data.groupby('ClientId').agg(agg_dict).reset_index()
        aggregated_data.columns = [
            col[0] if col[0] == 'ClientId' else '_'.join(
                filter(None, col)).strip() + f"_{window_name}{additional_suffix}"
            for col in aggregated_data.columns.values
        ]

        aggregated_data_dict[window_name] = aggregated_data

    # merge columns from different timestamps
    final_aggregated_df = pd.DataFrame()
    for window_name, window_data in aggregated_data_dict.items():
        if final_aggregated_df.empty:
            final_aggregated_df = window_data.copy()
        else:
            final_aggregated_df = pd.merge(final_aggregated_df, window_data, on='ClientId', how='outer')
    return final_aggregated_df


def remove_highly_correlated_columns(df, threshold=0.8):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_df.corr()

    columns_to_drop = set()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                columns_to_drop.add(colname)

    return list(columns_to_drop)


# Define time windows once
# time_windows = {
#     'last_week': '7D',
#     'last_3_weeks': '21D',
#     'last_2_months': '60D',
#     'last_half_year': '182D',
#     'last_year': '365D',
#     'last_2_years': '730D'
# }

time_windows = {
    '3_hours': '3H',
    '12_hours': '12H',
    '1_day': '1D',
    '2_days': '2D',
    '1_week': '7D',
    '1_month': '30D',
    '2_month': '60D',
}

time_window_timedeltas = {
    window_name: pd.to_timedelta(window_value) for window_name, window_value in time_windows.items()
}

start_date = pd.Timestamp("2020-01-01")
end_date = pd.Timestamp("2023-06-01")
current_date = start_date

timestamps = {}

while current_date <= end_date:
    key = current_date.strftime('%Y-%m-%d')
    timestamps[key] = current_date
    current_date += pd.Timedelta(weeks=1)


# f.e. table create relevant aggs until current time based on the table.
# On the end of agg extraction for each table - merge them all
def process_data(current_time_value):
    
    # 1. cube_sportsbook_bet processing
    
    df = dataframes["cube_sportsbook_bet"]
    df = df[df["ClientId"].isin(relevant_clients)]
    # columns to exclude from aggregations
    columns_to_exclude_sportsbook = ['ReportDTS', 'TimestampHour', 'ClientId', 'PartnerId', 'SourceName', 'BetTypeName',
                                     'CurrencyId']
    # Create the aggregation per type ( sum, count and mean for bool)
    agg_dict = create_agg_dict(df, columns_to_exclude_sportsbook)
    # Aggregate per time windows
    final_aggregated_df = aggregate_data(df, agg_dict, time_window_timedeltas, current_time=current_time_value)

    print("Processed cube_sportsbook_bet.")
   
    # 2.  cube_finance processing
    df = dataframes['cube_finance']
    df = df[df["ClientId"].isin(relevant_clients)]
    columns_to_exclude_finance = ['TimestampHour', 'ClientId', 'PartnerId', 'CurrencyId', 'Month']
    agg_dict = create_agg_dict(df, columns_to_exclude_finance)
    final_aggregated_df_cube_finance = aggregate_data(df, agg_dict, time_window_timedeltas,
                                                      current_time=current_time_value)

    # 3. cube_CasinoSpins processing
    df = dataframes['cube_CasinoSpins']
    df = df[df["ClientId"].isin(relevant_clients)]
    columns_to_exclude_casino = [
        'TimestampHour', 'CasinoPlayerId', 'ClientId', 'PartnerId', 'CurrencyId',
        'IsRakeTransaction', 'IsCalculated', 'IsRollbacked', 'IsTournamentTransaction',
        'IsBonus', 'GameId', 'ProviderId', 'ProductId', 'BonusId', 'TournamentId', 'TournamentPartnerId'
    ]
    agg_dict = create_agg_dict(df, columns_to_exclude_casino)
    final_aggregated_df_cube_CasinoSpins = pd.DataFrame()
    # Create spins and filter, one time if bonus is true, one time if bonus is false, one time isRakeTransaction true or false.
    # Omeri thinks Rake is if transaction was cancelled
    for column, value in [('IsBonus', True), ('IsBonus', False), ('IsRakeTransaction', True),
                          ('IsRakeTransaction', False)]:
        filtered_df = df[df[column] == value]
        aggregated_df = aggregate_data(filtered_df, agg_dict, time_window_timedeltas,
                                       additional_suffix=f"_{column}_{value}", current_time=current_time_value)
        if final_aggregated_df_cube_CasinoSpins.empty:
            final_aggregated_df_cube_CasinoSpins = aggregated_df.copy()
        else:
            final_aggregated_df_cube_CasinoSpins = pd.merge(final_aggregated_df_cube_CasinoSpins, aggregated_df,
                                                            on='ClientId', how='outer')

    # 4. client session
    df_client_session = dataframes['ClientSession']
    df_client_session = df_client_session[df_client_session["ClientId"].isin(relevant_clients)]
    df_client_session['StartTime'] = pd.to_datetime(df_client_session['StartTime'], format='mixed',
                                                    utc=True)  # Step 1: Specify columns to exclude for ClientSession dataframe
    columns_to_exclude_client_session = ['Id', 'ClientId', 'StartTime', 'Source', 'Token', 'LoginType',
                                         'EndTime', 'LogoutSource', 'DeviceId', 'Product']
    print("Processed cube_sportsbook_bet.")

    # Step 2: Create aggregation dictionary

    agg_dict_client_session = create_agg_dict_for_ClientSession(df_client_session, columns_to_exclude_client_session)

    print("Processed cube_finance.")

    # Step 3: Aggregate data
    final_aggregated_df_client_session = aggregate_data(df_client_session, agg_dict_client_session,
                                                        time_window_timedeltas,
                                                        time_col='StartTime', current_time=current_time_value)
 
     # 5. cube_sportsbook_bet_selection

    df = dataframes["cube_sportsbook_bet_selection"]
    columns_to_exclude_selection = ['TimestampHour', 'ClientId', 'PartnerId', 'SourceName', 'BetTypeName', 'CurrencyId']
    agg_dict = create_agg_dict(df, columns_to_exclude_selection)
    final_aggregated_df_cube_sportsbook_bet_selection = aggregate_data(df, agg_dict, time_window_timedeltas,
                                                                       current_time=current_time_value)
    # 6. viewmat_ClientDetails

    final_combined_df = dataframes['viewmat_ClientDetails'].merge(final_aggregated_df, right_on='ClientId',
                                                                  left_on="Id", how='left')
    final_combined_df = final_combined_df.merge(final_aggregated_df_cube_finance, on='ClientId', how='left')
    final_combined_df = final_combined_df.merge(final_aggregated_df_cube_CasinoSpins, on='ClientId', how='left')
    final_combined_df = final_combined_df.merge(final_aggregated_df_client_session, on='ClientId', how='left')
    final_combined_df = final_combined_df.merge(final_aggregated_df_cube_sportsbook_bet_selection, on='ClientId',
                                                how='left')
    final_combined_df = final_combined_df[final_combined_df["ClientId"].isin(relevant_clients)]
    print("All dataframes merged successfully.")

#  # Save final agg - Write back to s3 or locally

    # with open(f'/Users/omrilapidot/Vbet_adjusted_data/final_combined_df_{current_time_value}.pkl', 'wb') as f:
    #     pickle.dump(final_combined_df, f)


    buffer = BytesIO()
    final_combined_df.to_pickle(buffer)
    s3_client.put_object(Body=buffer.getvalue(), Bucket=bucket_name, Key=f'final_combined_df_{current_time_value}.pkl')

    print("data saved.")

# Main loop for aggregations - iterate every timestamp e.g. 1 week atm
for time_name, time_value in timestamps.items(): #
    process_data(time_value) # Create aggregations until current date
    print(f"Processed and saved for {time_name}.")
