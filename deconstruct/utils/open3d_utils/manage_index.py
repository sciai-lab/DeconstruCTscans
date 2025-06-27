import os
import yaml
import time


def create_index_entry(file_name):
    """expects absolute path to file"""
    return {"ctime": time.ctime(os.path.getctime(file_name)), "visualized": 0}


def create_fresh_index(folder_path, overwrite_existing=False):
    """
    Create a new index file for the folder at folder_path.
    :param folder_path: path to the folder for which to create the index file
    :param overwrite_existing: if True, overwrite existing index file
    :return: None
    """
    index_file_path = os.path.join(folder_path, "index.yml")
    if os.path.exists(index_file_path):
        if overwrite_existing:
            print(f"Overwriting existing index file at {index_file_path}.")
        else:
            print(f"Index file already exists at {index_file_path}.")
            return
    else:
        print(f"Creating index file at {index_file_path}.")

    # create index file:
    list_files = os.listdir(folder_path)
    list_files = [f for f in list_files if f.endswith(".pkl")]

    # create a dict and write to yml file:
    dict_index = {file: create_index_entry(os.path.join(folder_path, file)) for file in list_files}

    # write to yml file:
    with open(index_file_path, "w") as f:
        yaml.dump(dict_index, f)


def update_index(folder_path):
    """
    Update the index file for the folder at folder_path.
    :param folder_path:
    :return:
    """
    # index file exists?
    index_file_path = os.path.join(folder_path, "index.yml")
    assert os.path.exists(index_file_path), "Index file does not exist."

    # load index yml-file:
    with open(index_file_path, "r") as f:
        dict_index = yaml.load(f, yaml.SafeLoader)

    # get list of files in folder:
    list_files = os.listdir(folder_path)
    list_files = [f for f in list_files if f.endswith(".pkl")]

    # add new files to index:
    for file in list_files:
        if file not in dict_index:
            dict_index[file] = create_index_entry(os.path.join(folder_path, file))

    # write to yml file:
    with open(index_file_path, "w") as f:
        yaml.dump(dict_index, f)


def mark_as_visualized(folder_path, list_file_names):
    """
    Mark the files in list_file_names as visualized in the index file for the folder at folder_path.
    file_name in here is expected to be an absolute path.

    :param folder_path:
    :param list_file_names:
    :return:
    """
    index_file_path = os.path.join(folder_path, "index.yml")
    assert os.path.exists(index_file_path), "Index file does not exist."

    # load index yml-file:
    with open(index_file_path, "r") as f:
        dict_index = yaml.load(f, yaml.SafeLoader)

    for file_name in list_file_names:
        # check if file lies in folder:
        if os.path.dirname(file_name) != folder_path:
            print(f"File {file_name} not in folder. Will not be marked as visualized.")
            continue

        file_basename = os.path.basename(file_name)
        assert file_basename in dict_index, f"File {file_basename} not in index."
        print(f"Marking {file_basename} as visualized.")
        dict_index[file_basename]["visualized"] = 1

    # write to yml file:
    with open(index_file_path, "w") as f:
        yaml.dump(dict_index, f)


def get_files_to_visualize(folder_path, sort_latest_first=True):
    """
    Get a list of files to visualize from the index file for the folder at folder_path.
    :param folder_path:
    :return:
    """
    index_file_path = os.path.join(folder_path, "index.yml")
    assert os.path.exists(index_file_path), "Index file does not exist."

    # load index yml-file:
    with open(index_file_path, "r") as f:
        dict_index = yaml.load(f, yaml.SafeLoader)

    list_files = list(dict_index.keys())
    if sort_latest_first:
        list_times = [dict_index[file]["ctime"] for file in list_files]
        list_files, _ = zip(*sorted(zip(list_files, list_times), key=lambda x: time.strptime(x[1]), reverse=True))

    list_files_to_visualize = [os.path.join(folder_path, file_name)
                               for file_name in list_files if dict_index[file_name]["visualized"] == 0]

    return list_files_to_visualize