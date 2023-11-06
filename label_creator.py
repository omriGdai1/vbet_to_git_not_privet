import pandas as pd


def calculate_deposit(df, dates, time_deltas):
    """
    For each date in the list of dates, calculate the total DepositAmountInEUR
    per user from the given date to the date + time delta. Do this for each time delta.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - dates (list): A list of dates for which the deposit total should be calculated.
    - time_deltas (list): A list of time deltas for the calculation.

    Returns:
    - list: A list of DataFrames. Each DataFrame corresponds to one date and contains ClientId and the total deposits for each time delta.
    """

    results = []

    for date in dates:
        # Initialize a DataFrame with ClientIds (this ensures we capture all clients even if they don't have data for all time deltas)
        df_result = df[['ClientId']].drop_duplicates()

        for td in time_deltas:
            # Filter the DataFrame for records between the date and date + time delta
            mask = (df['TimestampHour'] >= date) & (df['TimestampHour'] < (date + td))
            filtered_data = df[mask]

            # Group by ClientId and sum the DepositAmountInEUR
            grouped = filtered_data.groupby('ClientId')['DepositAmountInEUR'].sum().reset_index()

            # Define the column name based on the start and end date of the period
            col_name = f'Total_DepositAmountInEUR_{date.date()}_{(date + td).date()}'
            grouped = grouped.rename(columns={'DepositAmountInEUR': col_name})

            # Merge the result into the df_result DataFrame
            df_result = pd.merge(df_result, grouped, on='ClientId', how='left').fillna(0)

        results.append(df_result)

    return results


# Example:
# dates = [pd.Timestamp('2021-01-01', tz='UTC'), pd.Timestamp('2021-02-01', tz='UTC')]
# time_deltas = [pd.Timedelta(days=7), pd.Timedelta(days=14)]
#
# dfs = calculate_deposit(df, dates, time_deltas)
