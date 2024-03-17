import json

def read_data(data_file):
    output = dict()
    with open(data_file, "r") as f:
        for line in f.readlines():
            instance = json.loads(line)
            output[instance["id"]] = instance
    return output


