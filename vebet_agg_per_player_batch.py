import pandas as pd
import numpy as np
import pickle
import boto3
from io import BytesIO

print("start")

with open('/Users/omrilapidot/Vbet_adjusted_data/dataframes.pkl', 'rb') as f:
    dataframes = pickle.load(f)
print("Dataframes loaded.")

dataframes['viewmat_ClientDetails']["ClientId"] = dataframes['viewmat_ClientDetails']["Id"]

with open('/Users/omrilapidot/Vbet_adjusted_data/dates_df.pkl', 'rb') as f:
    dates_df = pickle.load(f)

relevant_clients = dates_df.ClientId.to_list()


# s3_client = boto3.client('s3')
# # Download the pickle file from S3
# bucket_name, key_dataframes, key_dataframes_relevant_clients = "vbet-adjusted-data", "dataframes.pkl", "relevant_clients.pkl"
# response = s3_client.get_object(Bucket=bucket_name, Key=key_dataframes)
# pickle_bytes = response['Body'].read()
#
# # Load the pickle file into a DataFrame
# dataframes = pickle.loads(pickle_bytes)

# response = s3_client.get_object(Bucket=bucket_name, Key=key_dataframes_relevant_clients)
# pickle_bytes = response['Body'].read()

# # Load the pickle file
# relevant_clients = pickle.loads(pickle_bytes)


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
                   additional_suffix="",
                   time_col='TimestampHour'):
    aggregated_data_dict = {}
    for window_name in time_window_timedeltas.keys():
        window_size = time_window_timedeltas[window_name]
        # window_start = current_time - window_size
        # window_start = window_start.tz_localize(df[time_col].iloc[0].tz)

        # Assuming the dataframes are named 'dates_df' and 'cube_CasinoSpins'

        # Merge the two dataframes on 'ClientId'
        merged_df = pd.merge(df, dates_df, on="ClientId", how="inner")
        time_after_FTD = time_window_timedeltas["1_month"]
        # time_after_FTD = pd.to_timedelta("42D")
        # Filter the rows based on the condition
        filtered_df = merged_df[merged_df[time_col] <= merged_df['STD_date'] + time_after_FTD]
        filtered_df =filtered_df[filtered_df[time_col] >= filtered_df['STD_date'] + time_after_FTD - window_size]

        # If you want to keep only columns from 'cube_CasinoSpins' after filtering:
        filtered_df = filtered_df[df.columns]

        # aggregated_data = df[
        #     (df[time_col] >= window_start) & (df[time_col] < current_time.tz_localize(df[time_col].iloc[0].tz))]

        aggregated_data = filtered_df.groupby('ClientId').agg(agg_dict).reset_index()
        aggregated_data.columns = [
            col[0] if col[0] == 'ClientId' else '_'.join(
                filter(None, col)).strip() + f"_{window_name}{additional_suffix}"
            for col in aggregated_data.columns.values
        ]

        aggregated_data_dict[window_name] = aggregated_data

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
    '1_day': '1D',
    '2_days': '2D',
    '1_week': '7D',
    '2_weeks': '14D',
    '1_month': '30D',
    "6_weeks":"42D",
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


def process_data():
    # cube_sportsbook_bet processing
    df = dataframes["cube_sportsbook_bet"]
    df = df[df["ClientId"].isin(relevant_clients)]
    columns_to_exclude_sportsbook = ['ReportDTS', 'TimestampHour', 'ClientId', 'PartnerId', 'SourceName', 'BetTypeName',
                                     'CurrencyId']
    agg_dict = create_agg_dict(df, columns_to_exclude_sportsbook)
    final_aggregated_df = aggregate_data(df, agg_dict, time_window_timedeltas)

    print("Processed cube_sportsbook_bet.")
    # cube_finance processing
    df = dataframes['cube_finance']
    df = df[df["ClientId"].isin(relevant_clients)]
    columns_to_exclude_finance = ['TimestampHour', 'ClientId', 'PartnerId', 'CurrencyId', 'Month']
    agg_dict = create_agg_dict(df, columns_to_exclude_finance)
    final_aggregated_df_cube_finance = aggregate_data(df, agg_dict, time_window_timedeltas)
    print(len(final_aggregated_df_cube_finance))
    # cube_CasinoSpins processing
    df = dataframes['cube_CasinoSpins']
    df = df[df["ClientId"].isin(relevant_clients)]
    columns_to_exclude_casino = [
        'TimestampHour', 'CasinoPlayerId', 'ClientId', 'PartnerId', 'CurrencyId',
        'IsRakeTransaction', 'IsCalculated', 'IsRollbacked', 'IsTournamentTransaction',
        'IsBonus', 'GameId', 'ProviderId', 'ProductId', 'BonusId', 'TournamentId', 'TournamentPartnerId'
    ]
    agg_dict = create_agg_dict(df, columns_to_exclude_casino)
    final_aggregated_df_cube_CasinoSpins = pd.DataFrame()
    for column, value in [('IsBonus', True), ('IsBonus', False), ('IsRakeTransaction', True),
                          ('IsRakeTransaction', False)]:
        filtered_df = df[df[column] == value]
        aggregated_df = aggregate_data(filtered_df, agg_dict, time_window_timedeltas,
                                       additional_suffix=f"_{column}_{value}")
        if final_aggregated_df_cube_CasinoSpins.empty:
            final_aggregated_df_cube_CasinoSpins = aggregated_df.copy()
        else:
            final_aggregated_df_cube_CasinoSpins = pd.merge(final_aggregated_df_cube_CasinoSpins, aggregated_df,
                                                            on='ClientId', how='outer')
    print(len(final_aggregated_df_cube_CasinoSpins))
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
                                                        time_col='StartTime')

    df = dataframes["cube_sportsbook_bet"]
    columns_to_exclude_selection = ['TimestampHour', 'ClientId', 'PartnerId', 'SourceName', 'BetTypeName', 'CurrencyId']
    agg_dict = create_agg_dict(df, columns_to_exclude_selection)
    final_aggregated_df_cube_sportsbook_bet_selection = aggregate_data(df, agg_dict, time_window_timedeltas)

    final_combined_df = dataframes['viewmat_ClientDetails'].merge(final_aggregated_df, on='ClientId', how='left')
    final_combined_df = final_combined_df.merge(final_aggregated_df_cube_finance, on='ClientId', how='left')
    final_combined_df = final_combined_df.merge(final_aggregated_df_cube_CasinoSpins, on='ClientId', how='left')
    final_combined_df = final_combined_df.merge(final_aggregated_df_client_session, on='ClientId', how='left')
    final_combined_df = final_combined_df.merge(final_aggregated_df_cube_sportsbook_bet_selection, on='ClientId',
                                                how='left')
    print(len(final_combined_df))
    final_combined_df = final_combined_df[final_combined_df["ClientId"].isin(relevant_clients)]
    print(len(final_combined_df))
    print("All dataframes merged successfully.")

    with open('/Users/omrilapidot/Vbet_adjusted_data/test/final_combined_df_1_month_after_STD.pkl', 'wb') as f:
        pickle.dump(final_combined_df, f)

    # buffer = BytesIO()
    # final_combined_df.to_pickle(buffer)
    # s3_client.put_object(Body=buffer.getvalue(), Bucket=bucket_name, Key=f'final_combined_df_{current_time_value}.pkl')

    print("data saved.")


# for time_name, time_value in timestamps.items():
process_data()
print(f"Processed and saved for.")
