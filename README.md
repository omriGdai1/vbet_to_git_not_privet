# Data instructions in pandaIO
# additional noted from vbet in drive

# vbet_to_git_not_privet

## fetch data
Convert vbet jsons from google cloud into organized dict of dataframes.
Output:
dataframes.pkl - all tables - see vbet_from_json_files_to_pandas_df

relevant_clients.df - list of relevant clienyt based on a business logic. - Omri thinks its unique over client table provided from vbet - He will validate

dates.df - df that includes FTD and STD date


### Model 2 - predict high roller per week,month

## vbet_agg_data_creator.py
Created for aggregating model 2 ( next week and next month)
process_data(time_value) is the main processing  # Create aggregations until current date
TODO: understand how relevant_clients.pkl is being created

## In the notebook in aws sagemaker called vbet - is label creation and training and evaluation
Created for Model 2 ( per week, month)


### Model 1 - predict high roller globally

Create features and aggregartions
## vebet_agg_per_player_batch.py 
Same process as in vbet_agg_data_creator.py, but relative to ftd/std which are user based.
This file needs to be evaluaterd for each model timestamp e.g. ftd+ 2w etc. For each timestamps
Currently, the script aggregated data for std+4w model

Labels and training in model_early_stage.ipynb locally in notebook