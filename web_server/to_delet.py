import pandas as pd
import numpy as np

def create_agg_dict(df, columns_to_exclude=[]):
    agg_dict = {}
    for col, dtype in df.dtypes.items():
        if col not in columns_to_exclude:
            if dtype in ['float64', 'int64']:
                agg_dict[col] = ['sum']
            elif dtype == 'bool':
                agg_dict[col] = ['sum', 'mean']
    return agg_dict

def create_agg_dict_for_ClientSession(df, columns_to_exclude=[]):
    agg_dict = create_agg_dict(df, columns_to_exclude)
    mode_and_nunique_cols = ['DeviceId', 'Product', 'LogoutSource', 'LoginType']
    for col in mode_and_nunique_cols:
        if col not in columns_to_exclude:
            agg_dict[col] = ['nunique', lambda x: x.mode()[0] if not x.mode().empty else np.nan]
    agg_dict["Token"] = ['count']
    return agg_dict

def aggregate_data(df, agg_dict, time_window_timedeltas, time_col='TimestampHour', additional_suffix=""):
    aggregated_data_dict = {}
    for window_name, window_value in time_window_timedeltas.items():
        window_start = pd.Timestamp.now() - window_value
        window_start = window_start.tz_localize(df[time_col].iloc[0].tz)

        aggregated_data = df[df[time_col] >= window_start].groupby('ClientId').agg(agg_dict).reset_index()
        aggregated_data.columns = [
            col[0] if col[0] == 'ClientId' else '_'.join(filter(None, col)).strip() + f"_{window_name}{additional_suffix}"
            for col in aggregated_data.columns.values
        ]

        aggregated_data_dict[window_name] = aggregated_data

    final_aggregated_df = pd.concat(aggregated_data_dict.values(), axis=1)
    final_aggregated_df = final_aggregated_df.loc[:, ~final_aggregated_df.columns.duplicated()]
    return final_aggregated_df

# Define time windows
time_windows = {
    'last_week': '7D',
    'last_3_weeks': '21D',
    'last_2_months': '60D',
    'last_half_year': '182D',
    'last_year': '365D',
    'last_2_years': '730D'
}
time_window_timedeltas = {name: pd.to_timedelta(value) for name, value in time_windows.items()}

# Process and aggregate data for each cube/dataframe
cubes_info = {
    "cube_sportsbook_bet": ['ReportDTS', 'TimestampHour', 'ClientId', 'PartnerId', 'SourceName', 'BetTypeName', 'CurrencyId'],
    "cube_finance": ['TimestampHour', 'ClientId', 'PartnerId', 'CurrencyId', 'Month'],
    "cube_CasinoSpins": [
        'TimestampHour', 'CasinoPlayerId', 'ClientId', 'PartnerId', 'CurrencyId',
        'IsRakeTransaction', 'IsCalculated', 'IsRollbacked', 'IsTournamentTransaction',
        'IsBonus', 'GameId', 'ProviderId', 'ProductId', 'BonusId', 'TournamentId', 'TournamentPartnerId'
    ],
    "ClientSession": ['Id', 'ClientId', 'StartTime', 'Source', 'Token', 'LoginType', 'EndTime', 'LogoutSource', 'DeviceId', 'Product']
}

final_aggregated_data = {}
for cube, exclusions in cubes_info.items():
    df = dataframes[cube]
    if cube == "ClientSession":
        df['StartTime'] = pd.to_datetime(df['StartTime'], format='mixed', utc=True)
        agg_dict = create_agg_dict_for_ClientSession(df, exclusions)
        final_aggregated_data[cube] = aggregate_data(df, agg_dict, time_window_timedeltas, time_col='StartTime')
    else:
        agg_dict = create_agg_dict(df, exclusions)
        final_aggregated_data[cube] = aggregate_data(df, agg_dict, time_window_timedeltas)

# Merging all the final aggregated dataframes on ClientId
final_combined_df = pd.DataFrame()
for _, df in final_aggregated_data.items():
    if final_combined_df.empty:
        final_combined_df = df.copy()
    else:
        final_combined_df = pd.merge(final_combined_df, df, on='ClientId', how='outer')

final_combined_df = final_combined_df.merge(dataframes['viewmat_ClientDetails'], left_on='ClientId', right_on="Id", how='outer')