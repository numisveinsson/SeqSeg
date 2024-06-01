import yaml


def load_yaml(fn):
    """loads a yaml file into a dict"""
    with open(fn, 'r') as file_:
        try:
            return yaml.load(file_, Loader=yaml.loader.SafeLoader)
        except RuntimeError as e:
            print("failed to load yaml fille {}, {}\n".format(fn, e))
