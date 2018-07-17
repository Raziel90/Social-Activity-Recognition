
import numpy as np
import pandas as pd
from os import path
from os import listdir


def read_skel_text_file(filepath):
    """Loads the data from the text file containing the skeleton of a
    single user. The returned array has dimension (T,3,J) where T is
    the length of the sequence and J is the number of joints the second
    dimension corresponds to the x y x of the camera view."""

    df = pd.read_csv(filepath, header=None)
    data = np.transpose(np.reshape(np.array([line.split(' ')
                                             for line in df[0].values]
                                            ).astype(float).T,
                                   newshape=(6, 15, -1)),
                        axes=(2, 0, 1))[:, 0:3, :]
    return data


"""    Function to extract the data of a single video both subjects"""


def extract_data_activity_video(base_path, sess, act):

    file_format = 'a' + str(act).zfill(2) + '_s' + str(sess).zfill(2)

    # print(file_format)
    # print(path.join(base_path))

    files_user1 = [read_skel_text_file(path.join(base_path, fname)) for fname
                   in listdir(base_path)
                   if fname.startswith(file_format)
                   and fname.endswith('user1.txt')]

    files_user2 = [read_skel_text_file(path.join(base_path, fname)) for fname
                   in listdir(base_path)
                   if fname.startswith(file_format)
                   and fname.endswith('user2.txt')]

    files_user1 = np.concatenate(files_user1, axis=0)
    files_user2 = np.concatenate(files_user2, axis=0)

    return dict(activity=act, session=sess, user1=files_user1, user2=files_user2)


def load_dataset(base_path):

    numact = 8
    numsess = 10
    dataset = pd.DataFrame([extract_data_activity_video(base_path, sess, act) for act in range(
        1, numact + 1) for sess in range(1, numsess + 1)])

    return dataset
