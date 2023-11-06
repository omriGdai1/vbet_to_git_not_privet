import pandas as pd
import numpy as np
import pickle
import multiprocessing
import concurrent.futures
import boto3
from io import BytesIO

print("start")

with open('/Users/omrilapidot/Vbet_adjusted_data/dataframes.pkl', 'rb') as f:
    dataframes = pickle.load(f)
print("Dataframes loaded.")

dataframes['viewmat_ClientDetails']["ClientId"] = dataframes['viewmat_ClientDetails']["Id"]

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

print("corelative_strt")

# corelative_rows_per_df_dict = {df_name: remove_highly_correlated_columns(df, threshold=0.8) for df_name, df in
#                                dataframes.items()}

print("corelative_end")
with open('/Users/omrilapidot/Vbet_adjusted_data/dates_df.pkl', 'rb') as f:
    dates_df = pickle.load(f)

def create_agg_dict_for_ClientSession(df, columns_to_exclude=[], df_name="ClientSession"):
    agg_dict = create_agg_dict(df, columns_to_exclude, df_name)
    mode_and_nunique_cols = ['DeviceId', 'Product', 'LogoutSource', 'LoginType']
    for col in mode_and_nunique_cols:
        if col not in columns_to_exclude:
            agg_dict[col] = ['nunique', lambda x: x.mode()[0] if not x.mode().empty else np.nan]
    agg_dict["Token"] = ['count']
    return agg_dict


def create_agg_dict(df, columns_to_exclude, df_name):
    agg_dict = {}
    for col, dtype in df.dtypes.items():
        if dtype in ['float64', 'int64']:
            agg_dict[col] = ['sum', 'count']
        elif dtype == 'bool':
            agg_dict[col] = ['sum', 'mean']

    # correlated_columns = remove_highly_correlated_columns(df)
    correlated_columns = corelative_rows_per_df_dict[df_name]
    for col in list(set(columns_to_exclude + correlated_columns)):
        agg_dict.pop(col, None)
    return agg_dict


def aggregate_data(df,
                   agg_dict,
                   time_window_timedeltas,
                   current_time=None,
                   additional_suffix="",
                   time_col='TimestampHour'):
    if df.empty:
        print("*")
        return pd.DataFrame()

    if current_time is None:
        current_time = pd.Timestamp.now()

    if current_time.tz is None:
        current_time_tz = current_time.tz_localize(df[time_col].iloc[0].tz)
    else:
        current_time_tz = current_time.tz_convert(df[time_col].iloc[0].tz)

    aggregated_data_dict = {}
    for window_name in time_window_timedeltas.keys():
        window_size = time_window_timedeltas[window_name]
        window_start = current_time - window_size
        # window_start = window_start.tz_localize(df[time_col].iloc[0].tz)
        if window_start.tz is None:
            window_start = window_start.tz_localize(df[time_col].iloc[0].tz)
        else:
            window_start = window_start.tz_convert(df[time_col].iloc[0].tz)

        aggregated_data = df[
            (df[time_col] >= window_start) & (df[time_col] < current_time_tz)]
        aggregated_data = aggregated_data.groupby('ClientId').agg(agg_dict).reset_index()
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


def process_data(current_time_value, relevant_clients):
    # cube_sportsbook_bet processing
    df = dataframes["cube_sportsbook_bet"]
    # df = df[df["ClientId"].isin(relevant_clients)]
    df = df[df["ClientId"] == relevant_clients]
    columns_to_exclude_sportsbook = ['ReportDTS', 'TimestampHour', 'ClientId', 'PartnerId', 'SourceName', 'BetTypeName',
                                     'CurrencyId']
    agg_dict = create_agg_dict(df, columns_to_exclude_sportsbook, "cube_sportsbook_bet")
    final_aggregated_df = aggregate_data(df, agg_dict, time_window_timedeltas, current_time=current_time_value)

    print("Processed cube_sportsbook_bet.")
    # cube_finance processing
    df = dataframes['cube_finance']
    # df = df[df["ClientId"].isin(relevant_clients)]
    df = df[df["ClientId"] == relevant_clients]
    columns_to_exclude_finance = ['TimestampHour', 'ClientId', 'PartnerId', 'CurrencyId', 'Month']
    agg_dict = create_agg_dict(df, columns_to_exclude_finance, "cube_finance")
    final_aggregated_df_cube_finance = aggregate_data(df, agg_dict, time_window_timedeltas,
                                                      current_time=current_time_value)

    # cube_CasinoSpins processing
    df = dataframes['cube_CasinoSpins']
    # df = df[df["ClientId"].isin(relevant_clients)]
    df = df[df["ClientId"] == relevant_clients]
    columns_to_exclude_casino = [
        'TimestampHour', 'CasinoPlayerId', 'ClientId', 'PartnerId', 'CurrencyId',
        'IsRakeTransaction', 'IsCalculated', 'IsRollbacked', 'IsTournamentTransaction',
        'IsBonus', 'GameId', 'ProviderId', 'ProductId', 'BonusId', 'TournamentId', 'TournamentPartnerId'
    ]
    agg_dict = create_agg_dict(df, columns_to_exclude_casino, "cube_CasinoSpins")
    final_aggregated_df_cube_CasinoSpins = pd.DataFrame()
    for column, value in [('IsBonus', True), ('IsBonus', False), ('IsRakeTransaction', True),
                          ('IsRakeTransaction', False)]:
        filtered_df = df[df[column] == value]
        aggregated_df = aggregate_data(filtered_df, agg_dict, time_window_timedeltas,
                                       additional_suffix=f"_{column}_{value}", current_time=current_time_value)
        if final_aggregated_df_cube_CasinoSpins.empty:
            final_aggregated_df_cube_CasinoSpins = aggregated_df.copy()
        elif not aggregated_df.empty:
            final_aggregated_df_cube_CasinoSpins = pd.merge(final_aggregated_df_cube_CasinoSpins, aggregated_df,
                                                            on='ClientId', how='outer')

    df_client_session = dataframes['ClientSession']
    # df_client_session = df_client_session[df_client_session["ClientId"].isin(relevant_clients)]
    df_client_session = df_client_session[df_client_session["ClientId"] == relevant_clients]
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

    # df = dataframes["cube_sportsbook_bet_selection"]
    df = dataframes["cube_sportsbook_bet"]
    columns_to_exclude_selection = ['TimestampHour', 'ClientId', 'PartnerId', 'SourceName', 'BetTypeName', 'CurrencyId']
    agg_dict = create_agg_dict(df, columns_to_exclude_selection, "cube_sportsbook_bet")
    final_aggregated_df_cube_sportsbook_bet_selection = aggregate_data(df, agg_dict, time_window_timedeltas,
                                                                       current_time=current_time_value)

    if not final_aggregated_df_cube_finance.empty:
        final_combined_df = dataframes['viewmat_ClientDetails'].merge(final_aggregated_df, on='ClientId', how='left')
    else:
        final_combined_df = dataframes['viewmat_ClientDetails']
    if not final_aggregated_df_cube_finance.empty:
        final_combined_df = final_combined_df.merge(final_aggregated_df_cube_finance, on='ClientId', how='left')
    if not final_aggregated_df_cube_CasinoSpins.empty:
        final_combined_df = final_combined_df.merge(final_aggregated_df_cube_CasinoSpins, on='ClientId', how='left')
    if not final_aggregated_df_client_session.empty:
        final_combined_df = final_combined_df.merge(final_aggregated_df_client_session, on='ClientId', how='left')
    if not final_aggregated_df_cube_sportsbook_bet_selection.empty:
        final_combined_df = final_combined_df.merge(final_aggregated_df_cube_sportsbook_bet_selection, on='ClientId',
                                                    how='left')
    # final_combined_df = final_combined_df[final_combined_df["ClientId"].isin(relevant_clients)]
    final_combined_df = final_combined_df[final_combined_df["ClientId"] == relevant_clients]
    print("All dataframes merged successfully.")

    with open(
            f'/Users/omrilapidot/Vbet_adjusted_data/test/final_combined_df_{current_time_value}{relevant_clients}.pkl',
            'wb') as f:
        pickle.dump(final_combined_df, f)

def process_row(data):
    index, row = data  # Unpack the tuple here
    client_id = row['ClientId']
    ftd_date = row['FTD_date']
    print(f"Client ID: {client_id}, FTD Date: {ftd_date}")
    process_data(current_time_value=ftd_date, relevant_clients=client_id)


def main():
    num_processes = 4  # or any other desired number
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_row, dates_df.iterrows())

if __name__ == '__main__':
    main()