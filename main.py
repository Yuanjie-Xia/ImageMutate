import matplotlib.pyplot as plt
import image_transfer
import time
import os
import cv2
from random import randint
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from modAL.models import ActiveLearner, CommitteeRegressor
from modAL.disagreement import max_std_sampling


def image_mutate(mode, address):
    ori = cv2.imread(address)
    col, wid, rgb = ori.shape
    img = cv2.copyMakeBorder(ori, 0, 500 - col, 0, 500 - wid, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    for action in mode:
        if action == "positionConvert":
            img, mutated_image = image_transfer.all_position_convert(ori)
            ori = img
        if action == "smoothing":
            img, mutated_image = image_transfer.smooth(ori)
            ori = img
        if action == "Gaussian":
            img, mutated_image = image_transfer.gaussianFiltering(ori)
            ori = img
        if action == "Median":
            img, mutated_image = image_transfer.medianblur(ori)
            ori = img
        if action == "Bilateral":
            img, mutated_image = image_transfer.bilateralFiltering(ori)
            ori = img
        if action == "SelfDefine":
            img, mutated_image = image_transfer.reverse(ori)
            ori = img
    return img, mutated_image


def image_process(new_image,config):
    command = ''
    for item in config:
        command += str(item) + ' '
    plt.imsave("transformed_image.JPEG", new_image)
    #print('transformed image:')
    start_time = time.time()
    #print('djpeg ' + command + 'transformed_image.JPEG>002')
    os.system('djpeg ' + command + 'transformed_image.JPEG>002')
    running_time1 = time.time() - start_time
    return running_time1


def get_result(config_id, full_config_setting, mode, config, image_address):
    mode_selected = []
    config_selected = []
    item_list = []
    for id in config_id:
        item_list.append(full_config_setting[int(id)])
    #item_list = [full_config_setting[int(config_id)]]
    #print(item_list)
    for item in item_list:
        if item in mode:
            mode_selected.append(item)
        if item in config:
            config_selected.append(item)

    if len(mode_selected)>0:
        old_image, new_image = image_mutate(mode_selected, image_address)
    else:
        ori = cv2.imread(image_address)
        col, wid, rgb = ori.shape
        new_image = cv2.copyMakeBorder(ori, 0, 500 - col, 0, 500 - wid, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    #cv2.imwrite("new_image.JPEG", new_image)
    running_time = image_process(new_image, config_selected)
    #print(running_time)
    return running_time


def rand_config_generation(config_set):
    i = 0
    selection_set =[]
    for item_set in config_set:
        judge = randint(0,1)
        if judge == 1:
            pick = randint(0, len(item_set) - 1)
            point = [i, pick]
            selection_set.append(point)
        i += 1

    return selection_set


def predictionModel(config_features_raw, image_address, initial_idx):
    config_features = config_features_raw
    config_features_id = list(range(0, len(config_features_raw)))
    config_features_id = np.asarray(config_features_id)
    #results = np.asarray(results, dtype=np.float64)
    np.random.seed(777)
    sampled_config_ids = list(np.random.randint(len(config_features), size=2))
    x_train = [config_features_id[idx] for idx in initial_idx]
    result_train = []
    for item in x_train:
        result_train.append(get_result(item, config_features, config_features_raw, image_address))
        #print(result_train)
    learner_list = [ActiveLearner(
        estimator=RandomForestRegressor(),
        X_training=x_train, y_training=result_train
    )]
    # initializing the Committee
    committee = CommitteeRegressor(
        learner_list=learner_list,
        query_strategy=max_std_sampling
    )

    model = RandomForestRegressor()
    n_queries = 1
    res_al = []
    for idx in range(n_queries):
        X_train = config_features_id[sampled_config_ids]
        y_train = [get_result(X_train, config_features, config_features_raw, image_address)]
        X_test = config_features_id[~np.isin(np.arange(len(config_features)), sampled_config_ids)]
        y_test = [get_result(X_test, config_features, config_features_raw, image_address)]
        model.fit([X_train], y_train)
        res_al.append(model.score([X_test], y_test))
        query_idx, query_instance = committee.query(config_features)
        sampled_config_ids += list(query_idx)
        query_result = [get_result(config_features[query_idx], config_features
                                   , config_features_raw, image_address)]
        committee.teach([config_features[query_idx]], query_result)


def main():
    file1 = open('print.txt', 'r')
    configuration_set = [['-bmp', '-gif', '-os2', '-pnm'], ['-scale 1/2', '-scale 1/4', '-scale 1/8'],
                         ['-dct int', '-dct fast', '-dct float'],
                         ['-dither fs', '-dither ordered', '-dither none'],
                         ['-nosmooth']]
    count = 0
    response_time_list = []
    image_name_list = []
    while True:
        count += 1
        # Get next line from file
        line = file1.readline()
        # if line is empty
        # end of file is reached
        if not line:
            break
        if count > 1:
            item = line.split()
            image_address = item[1]
            image_name_list.append(image_address)
            # image_address = "ILSVRC/Data/DET/test/ILSVRC2017_test_00000028.JPEG"
            # rotate(image_address)
            # shearing(image_address)
            # print(image_address)

    mode = ['positionConvert', 'Gaussian', 'Median', 'Bilateral', 'SelfDefine']
    # positionConvert, smoothing, Gaussian, Median, Bilateral, SelfDefine
    
    initial_idx = [[0, 7, 3, 8], [4,6,15]]
    predictionModel(mode, configuration_set, "ILSVRC/Data/DET/test/ILSVRC2017_test_00000001.JPEG", initial_idx)
    # plt.axis('off')
    # plt.imshow(transformed_image)
    # plt.show()


if __name__ == "__main__":
    main()

