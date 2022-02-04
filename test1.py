import numpy as np
from sklearn.ensemble import RandomForestRegressor
from modAL.models import ActiveLearner, CommitteeRegressor
from modAL.disagreement import max_std_sampling
import cv2

ori0 = cv2.imread("ILSVRC/Data/DET/test/ILSVRC2017_test_00000001.JPEG")
ori0 = cv2.cvtColor(ori0, cv2.COLOR_BGR2RGB)
col, wid, rgb = ori0.shape
ori0 = cv2.copyMakeBorder(ori0, 0, 500-col, 0, 500-wid, cv2.BORDER_CONSTANT, value=[0, 0, 0])
ori0 = np.asarray(ori0)
ori0 = ori0.transpose(2,0,1).reshape(3,-1)


### Input format
config_features_raw = [ori0[0],ori0[1],ori0[2]]
results = [10, 11, 12]
config_features = np.asarray(config_features_raw, dtype=np.float64)
results = np.asarray(results, dtype=np.float64)
initial_idx = [[0, 1, 2],[0,2],[0,1]]

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
n_queries = 1
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