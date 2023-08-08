import os

import pandas

from algorithms.symbolicTransformer.src.functionnal.data_preparation import retrieve_conte_dataset
from common.constant import Dialect
from common.constant import EnvType


def duplicate_sentence_detection(config, args):
    """
    find in database a test sentence also in training
    """
    training_data = retrieve_conte_dataset(
        EnvType.TRAINING.value,
        os.environ['HOME']+config['configuration_path']['application_path']+args['--app-path']+config['configuration_path']['application_path'],
        config["configuration_path"]["selected_db"],
        Dialect.LSF,
        config["learning_config"]["english_output"],
        False,
        10000)

    test_data = retrieve_conte_dataset(
        EnvType.TEST.value,
        os.environ['HOME']+config['configuration_path']['application_path']+args['--app-path']+config['configuration_path']['application_path'],
        config["configuration_path"]["selected_db"],
        Dialect.LSF,
        config["learning_config"]["english_output"],
        False,
        10000)

    print()

    duplicates = []
    for data in test_data:
        if data[0] in pandas.DataFrame(training_data)[0].tolist():
            duplicates.append(str(data[0]))

    return duplicates
