import argparse
from datetime import datetime
from pymongo import MongoClient
import sklearn.feature_extraction.text as skt
import sklearn.metrics as skm
import sklearn.model_selection as skms
import tqdm

from medaboost import MedaBoost, TruthAdaBoost
from evalHelper import read_data, get_train_test, get_tfidf_mod


PARAMS = {'n_estimators': [50, 100, 150, 200],
          'learning_rate': [0.05, 0.1, 0.2]}

MODELS = {
    "baseline_mv": TruthAdaBoost(truth_inf="MV"),
    "baseline_wmv": TruthAdaBoost(truth_inf="WMV"),
    "baseline_ds": TruthAdaBoost(truth_inf="DS", ds_iter=500),
    "meda_mv": MedaBoost(truth_inf="MV"),
    "meda_wmv": MedaBoost(truth_inf="WMV"),
    "meda_ds": MedaBoost(truth_inf="DS", ds_iter=500)
}


def main():
    parser = argparse.ArgumentParser()
    # mongo information
    parser.add_argument("-mongo_url")
    parser.add_argument("-mongo_db")
    parser.add_argument("-mongo_col")
    # default information
    parser.add_argument("-data_file")
    parser.add_argument("-y_file")
    parser.add_argument("-test_dir")
    parser.add_argument("-mf", type=int, default=1000, 
                        help="tfidf features")
    parser.add_argument("--models", nargs='+', 
                        default=["baseline_mv", "baseline_wmv",
                                 "baseline_ds", "meda_mv",
                                 "meda_wmv","meda_ds"])
    parser.add_argument("-st", type=int, default=0,
                        help="sample_weight calculation")
    parser.add_argument("-lt", default="exponential",
                        help="gradient boosting loss")
    
    
    args = parser.parse_args()

    df, annot_labels = read_data(args.data_file,
                                 args.y_file)

    mclient = MongoClient(args.mongo_url)
    mdb = mclient[args.mongo_db]
    mcol = mdb[args.mongo_col]

    # construct the output
    json_output = {
        "feat": "tfidf",
        "mf": args.mf,
        "label": "all",
        "ts": datetime.now(),
        "sample_type": args.st,
        "lt": args.lt
    }
    
    for i in tqdm.tqdm(range(1, 11)):
        # read the data
        tn_df, ts_df, tn_y, ts_y = get_train_test(args.test_dir,
                                                  i, df, annot_labels)
        # transform to tf-idf
        #tfidf_mod = skt.TfidfVectorizer(stop_words='english',
        #                                max_features=args.mf)
        tfidf_mod = get_tfidf_mod(args.mf)
        train_x = tfidf_mod.fit_transform(tn_df["text"])
        test_x = tfidf_mod.transform(ts_df["text"])

        for mk in tqdm.tqdm(args.models,
                            desc="model",
                            leave=False):
            # cycle through all the classifiers
            gs = skms.GridSearchCV(MODELS[mk],
                                   PARAMS,
                                   cv=5,
                                   n_jobs=10,
                                   scoring='roc_auc')
            # infer the label first
            single_y = MODELS[mk].infer_labels(tn_y)
            # train
            gs.fit(train_x, single_y, raw_y=tn_y,
                   sample_type=args.st, loss_func=args.lt)
            # fit on the probability
            y_hat = gs.predict_proba(test_x)[:, 1]
            # score the model
            clf_auc = skm.roc_auc_score(ts_y, y_hat)
            clf_aps = skm.average_precision_score(ts_y, y_hat)
            opt_params = gs.best_params_            
            # add it to the performance
            perf_dict = {"clf": mk,
                         "run": i,
                         "auc": clf_auc,
                         "aps": clf_aps}
            perf_res = {**json_output, **perf_dict, **opt_params}
            mcol.insert_one(perf_res)

    mclient.close()

if __name__ == '__main__':
    main()

