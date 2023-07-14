import argparse
import pandas as pd
import sklearn.feature_extraction.text as skt
import sklearn.metrics as skm
import sklearn.model_selection as skms
import tqdm


def _train_test(df, test_hid, hid_col='h_id'):
    test_mask = df[hid_col].isin(test_hid)
    return df[~test_mask], df[test_mask]


def read_data(data_file, y_file):
    # read the data
    df = pd.read_csv(data_file)
    # drop the labels
    df = df[["HADM_ID", "TEXT"]]
    df = df.rename(columns={'HADM_ID':'h_id', 'TEXT':'text'})
    # read annotation labels
    annot_labels = pd.read_csv(y_file)
    return df, annot_labels


def get_train_test(test_dir, i, df, y,
                   ohid_col="HADM_ID",
                   hid_col="h_id",
                   gt_col="ground_truth"):
    # load the file first
    test_file = test_dir + str(i) + ".csv"
    test_labels = pd.read_csv(test_file)
    # drop the duplicates
    test_labels = test_labels.drop_duplicates()
    test_hid = test_labels[ohid_col]
    # set the training / test split
    tmp_train_y, tmp_test_y = _train_test(y, test_hid)
    pat_y = pd.concat([tmp_train_y, tmp_test_y])
    train_y = tmp_train_y.drop(columns=hid_col) 
    # make sure the test hadm_id are there
    test_mask = test_labels[ohid_col].isin(tmp_test_y[hid_col])
    test_y = test_labels[test_mask][gt_col]
    # subset the df to be only those in pat_y
    pat_mask = df[hid_col].isin(pat_y[hid_col])
    df = df[pat_mask]
    # join the two
    train_df, test_df = _train_test(df, test_hid)
    return train_df, test_df, train_y, test_y


def _lower_text(text):
    text = text.lower()
    return text


def get_tfidf_mod(nfeats):
    return skt.TfidfVectorizer(stop_words='english',
                               max_features=nfeats,
                               token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',
                               preprocessor=_lower_text)
