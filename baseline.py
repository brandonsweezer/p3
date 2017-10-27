def load_file(file):
	with open(file) as data_file:
		data = json.loads(data_file)
	return data

load_file(development)
json_data=open(file_directory).read()

data = json.loads(json_data)
pprint(data)