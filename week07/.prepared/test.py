

'''
feature_index = [i for i in range(n_features)]

# you should implement the function like Recursive Feature Elimination in sklearn
# http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html

# external loop to remove feature once at a time
for n_remove_fea in range(0, n_features - 2):
    print('========')
    history_score = []
    # internal loop to compare the performance of each feature
    for i in range(0, n_features - n_remove_fea):
        curr_train_x = train_x[:, feature_index[:i] + feature_index[i+1:]]
        curr_test_x = test_x[:, feature_index[:i] + feature_index[i+1:]]
        
        classifier = GaussianNB()
        curr_model = classifier.fit(curr_train_x, train_y)
        curr_score = classifier.score(curr_test_x, test_y)
        history_score.append(curr_score)
        
    # TODO: use np.argmax to get the least important feature and delete it and print the remaining feature index
    rm_feat_idx = np.argmax(history_score)
    feature_index = feature_index[:rm_feat_idx] + feature_index[rm_feat_idx+1:]
    print(n_remove_fea)
    print(feature_index)
    print(history_score)

'''
