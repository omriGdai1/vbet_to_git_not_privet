with open('/Users/omrilapidot/Vbet_adjusted_data/cube_finance_df.pkl', 'rb') as f:
    cube_finance_df = pickle.load(f)
cube_finance_df = cube_finance_df[["ClientId","TimestampHour","DepositAmountInEUR"]]
cube_finance_df['FirstDepositDate'] = cube_finance_df.groupby('ClientId')['TimestampHour'].transform('min')
sorted_df = cube_finance_df.sort_values(by=['ClientId', 'TimestampHour'])
sorted_df= sorted_df.groupby('ClientId').nth(1)[["ClientId","TimestampHour"]].rename(columns = {"TimestampHour":"SecondTransactionDate"})
dates_df = cube_finance_df.merge(sorted_df, on="ClientId", how="left")