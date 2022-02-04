import numpy as np
from modAL.models import ActiveLearner
from modAL.models import CommitteeRegressor
from sklearn.ensemble import RandomForestRegressor
X_train_np = [] # image name
config_features = [] # configuration setting(input)
results = [] # response time(output)
initial_idx = [] #
max_std_sampling = 0


def activeLearner(config_features_raw, results, n_queries):
    # config_features_raw = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [10, 11, 12]]
    # results = [10, 11, 12, 13, 13]
    config_features = np.asarray(config_features_raw, dtype=np.float64)
    results = np.asarray(results, dtype=np.float64)
    initial_idx = [[1], [2, 0]]

    np.random.seed(666)
    sampled_config_ids = list(np.random.randint(len(config_features), size=2))
    learner_list = [ActiveLearner(
        estimator=RandomForestRegressor(),
        X_training=config_features[idx], y_training=results[idx]
    )
        for idx in initial_idx]

    # initializing the Committee
    committee = CommitteeRegressor(
        learner_list=learner_list,
        query_strategy=max_std_sampling
    )

    model = RandomForestRegressor()
    res_al = []
    for idx in range(n_queries):
        X_train = config_features[sampled_config_ids]
        y_train = results[sampled_config_ids]
        X_test = config_features[~np.isin(np.arange(len(config_features)), sampled_config_ids)]
        y_test = results[~np.isin(np.arange(len(config_features)), sampled_config_ids)]

        model.fit(X_train, y_train)
        res_al.append(model.score(X_test, y_test))
        query_idx, query_instance = committee.query(config_features)
        sampled_config_ids += list(query_idx)
        committee.teach(config_features[query_idx], results[query_idx])


