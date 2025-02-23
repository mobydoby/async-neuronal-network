import yaml
import os


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    config = load_config("../config.yaml")
    print(config)
