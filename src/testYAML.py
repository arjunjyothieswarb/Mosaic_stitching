import yaml

with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

val = config["SIFT"]["nOctaveLayers"]

print(val)