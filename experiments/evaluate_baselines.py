import argparse
from datetime import datetime
from pymongo import MongoClient
import pandas as pd
import sklearn.ensemble as sken
import sklearn.linear_model as sklm
import sklearn.model_selection as skms
import sklearn.naive_bayes as sknb
import sklearn.neighbors as skknn
import sklearn.neural_network as sknn
import sklearn.preprocessing as skpp
import sklearn.svm as sksvm
import sklearn.tree as sktree
import sklearn.metrics as skm
import tqdm
import xgboost as xgb


from evalHelper import read_data, get_train_test, get_tfidf_mod
from medaboost.truth import MajorityVote, WeightedMajorityVote, DS


MODEL_PARAMS = {
    "logr-l1": {
         'model': sklm.LogisticRegression(penalty="l1",
                                          solver="liblinear",
                                          max_iter=10000),
         'params': {'C': [0.01, 0.1]}
     },
    "logr-l2": {
        'model': sklm.LogisticRegression(penalty="l2",
                                         solver="lbfgs",
                                         max_iter=10000),
        'params': {'C': [0.001, 0.01]}
    },
    "dt": {
        'model': sktree.DecisionTreeClassifier(min_samples_split=5,
                                               min_samples_leaf=5),
        'params': {'criterion': ['gini', 'entropy'],
                   'max_depth': range(3,8)}
    },
    "knn": {
        'model': skknn.KNeighborsClassifier(),
        'params': {'n_neighbors': range(3,10)}
    },
    "rf": {
         'model': sken.RandomForestClassifier(),
         'params': {'max_depth': [10],
                    'min_samples_leaf': [5, 10],
                    'n_estimators': [50, 100, 150]}
    },
    "xgboost": {
        "model": xgb.XGBClassifier(use_label_encoder=False),
        "params": {
            "max_depth": [6,7,8],
            "n_estimators": [100, 150],
            "learning_rate":[0.1],
            "eval_metric":["logloss"]
        }
    },
    "mlp": {
        "model": sknn.MLPClassifier(),
        "params": {
            "hidden_layer_sizes": [(25, ), (50,), (25, 25, ), (50, 50,)],
            "max_iter": [1000]
        }
    }
}


TRUTH_MODELS = {"mv": MajorityVote(),
                "wmv": WeightedMajorityVote(),
                "ds": DS(max_iter=500)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mongo_url")
    parser.add_argument("-mongo_db")
    parser.add_argument("-mongo_col")
    parser.add_argument("-data_file")
    parser.add_argument("-y_file")
    parser.add_argument("-test_dir")
    parser.add_argument("-mf", type=int, default=1000,
                        help="tfidf features")
    parser.add_argument("--labels", nargs='+',
                        default=["ehapi", "egem",
                                 "cantrip", "sotoodeh",
                                 "mv", "wmv", "ds"])
    
    args = parser.parse_args()

    df, annot_labels = read_data(args.data_file,
                                 args.y_file)
    perf_list = []
    # setup the mongo stuff
    mclient = MongoClient(args.mongo_url)
    mdb = mclient[args.mongo_db]
    mcol = mdb[args.mongo_col]
    # construct the output
    json_output = {
        "feat": "tfidf",
        "mf": args.mf,
        "ts": datetime.now()}

    for i in tqdm.tqdm(range(1, 11), desc="test-split"):
        # read the data
        tn_df, ts_df, tn_y, ts_y = get_train_test(args.test_dir,
                                                  i, df,
                                                  annot_labels)
        # transform to tf-idf
        tfidf_mod = get_tfidf_mod(args.mf)
        train_x = tfidf_mod.fit_transform(tn_df["text"])
        test_x = tfidf_mod.transform(ts_df["text"])

        # convert to dense
        train_x = train_x.toarray()
        test_x = test_x.toarray()
        
        for mk, mk_dict in tqdm.tqdm(MODEL_PARAMS.items(),
                                      desc="model-type", leave=False):

            for col in args.labels:
                cur_y = None
                if col in ["ehapi", "egem", "cantrip", "sotoodeh"]:
                    cur_y = tn_y[col]
                else:
                    truth_model = TRUTH_MODELS[col]
                    cur_y = truth_model.infer(tn_y)
                    cur_y = cur_y.astype(int)
                
                # setup grid search
                gs = skms.GridSearchCV(mk_dict["model"],
                                       mk_dict["params"],
                                       cv=5,
                                       n_jobs=4,
                                       scoring='roc_auc')
                # train the model with just the label of interest
                gs.fit(train_x, cur_y)
                # get the estimated prediction
                y_hat = gs.predict_proba(test_x)[:, 1]
                # score the model
                auc = skm.roc_auc_score(ts_y, y_hat)
                aps = skm.average_precision_score(ts_y, y_hat)
                opt_params = gs.best_params_
                # now print out file
                perf_dict = {"clf": mk,
                             "label": col,
                             "run": i,
                             "auc": auc,
                             "aps": aps}
                perf_res = {**json_output, **perf_dict, **opt_params}
                mcol.insert_one(perf_res)
            
    mclient.close()


if __name__ == '__main__':
    main()

