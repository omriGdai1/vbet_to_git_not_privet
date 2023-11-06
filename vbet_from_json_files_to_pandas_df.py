subdir_path = "/Users/omrilapidot/Vbet_data/cube_sportsbook_bet"
json_files = [f for f in os.listdir(subdir_path) if f.endswith('.json')]

# Create a list to hold the data from each JSON file
data = []
print(len(json_files))
i = 0
for json_file in json_files:
    if i%100==0:
        print(i)
    i+=1
    file_path = os.path.join(subdir_path, json_file)
    try:
        json_data = pd.read_json(file_path, lines=True)
        data.append(json_data)
    except ValueError:
        print(f"Error parsing JSON in file: {json_file}. Skipping this file.")

# Create a DataFrame for the current subdirectory
df = pd.concat(data)