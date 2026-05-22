import os
import pickle
import numpy as np



def load_articles(obj, adv_test_file=None):
    print('Dataset: ', obj)
    print("loading news articles")

    train_dict = pickle.load(open('data/news_articles/' + obj + '_train.pkl', 'rb'))
    test_dict = pickle.load(open('data/news_articles/' + obj + '_test.pkl', 'rb'))

    if adv_test_file is not None:
        restyle_dict = pickle.load(open(adv_test_file, 'rb'))
    else:
        restyle_dict = pickle.load(open('data/adversarial_test/' + obj + '_test_adv_A.pkl', 'rb'))
        # alternatively, switch to loading other adversarial test sets with '_test_adv_[B/C/D].pkl'

    x_train, y_train = train_dict['news'], train_dict['labels']
    x_test, y_test = test_dict['news'], test_dict['labels']

    # try to find appropriate key in restyle dict
    if isinstance(restyle_dict, dict) and 'news' in restyle_dict:
        x_test_res = restyle_dict['news']
    else:
        # fallback: if the pickle is a list or contains rewritten key
        if isinstance(restyle_dict, dict) and 'rewritten' in restyle_dict:
            x_test_res = restyle_dict['rewritten']
        elif isinstance(restyle_dict, (list, tuple)):
            x_test_res = restyle_dict
        else:
            raise ValueError('Unknown format for adversarial test file: {}'.format(adv_test_file))

    return x_train, x_test, x_test_res, y_train, y_test


def load_reframing(obj, mode='style', sentiment_dir='data/sentiment_data', target_len=None):
    print("loading news augmentations")
    print('Dataset: ', obj, 'Mode:', mode)

    if mode == 'style':
        restyle_dict_train1_1 = pickle.load(open('data/reframings/' + obj+ '_train_objective.pkl', 'rb'))
        restyle_dict_train1_2 = pickle.load(open('data/reframings/' + obj+ '_train_neutral.pkl', 'rb'))
        restyle_dict_train2_1 = pickle.load(open('data/reframings/' + obj+ '_train_emotionally_triggering.pkl', 'rb'))
        restyle_dict_train2_2 = pickle.load(open('data/reframings/' + obj+ '_train_sensational.pkl', 'rb'))

        finegrain_dict1 = pickle.load(open('data/veracity_attributions/' + obj+ '_fake_standards_objective_emotionally_triggering.pkl', 'rb'))
        finegrain_dict2 = pickle.load(open('data/veracity_attributions/' + obj+ '_fake_standards_neutral_sensational.pkl', 'rb'))

        x_train_res1 = np.array(restyle_dict_train1_1['rewritten'])
        x_train_res1_2 = np.array(restyle_dict_train1_2['rewritten'])
        x_train_res2 = np.array(restyle_dict_train2_1['rewritten'])
        x_train_res2_2 = np.array(restyle_dict_train2_2['rewritten'])

        y_train_fg, y_train_fg_m, y_train_fg_t = finegrain_dict1['orig_fg'], finegrain_dict1['mainstream_fg'], finegrain_dict1['tabloid_fg']
        y_train_fg2, y_train_fg_m2, y_train_fg_t2 = finegrain_dict2['orig_fg'], finegrain_dict2['mainstream_fg'], finegrain_dict2['tabloid_fg']

        replace_idx = np.random.choice(len(x_train_res1), len(x_train_res1) // 2, replace=False)

        x_train_res1[replace_idx] = x_train_res1_2[replace_idx]
        x_train_res2[replace_idx] = x_train_res2_2[replace_idx]
        y_train_fg[replace_idx] = y_train_fg2[replace_idx]
        y_train_fg_m[replace_idx] = y_train_fg_m2[replace_idx]
        y_train_fg_t[replace_idx] = y_train_fg_t2[replace_idx]

        return x_train_res1, x_train_res2, y_train_fg, y_train_fg_m, y_train_fg_t

    elif mode == 'sentiment':
        # sentiment data expected under sentiment_dir with filenames like '{obj}_train_positive.pkl'
        pos_path = os.path.join(sentiment_dir, obj + '_train_positive.pkl')
        neg_path = os.path.join(sentiment_dir, obj + '_train_negative.pkl')

        pos_dict = pickle.load(open(pos_path, 'rb'))
        neg_dict = pickle.load(open(neg_path, 'rb'))

        def extract_texts(d):
            if isinstance(d, dict):
                for key in ['rewritten', 'news', 'texts', 'articles', 'text']:
                    if key in d:
                        return np.array(d[key])
                # fallback: try common keys
                if len(d) == 0:
                    return np.array([])
                # if dict maps indices to strings
                values = list(d.values())
                if all(isinstance(x, str) for x in values):
                    return np.array(values)
            elif isinstance(d, (list, tuple, np.ndarray)):
                return np.array(d)
            raise ValueError('Unknown sentiment pkl format')

        x_train_res1 = extract_texts(pos_dict)
        x_train_res2 = extract_texts(neg_dict)

        # If target_len provided, sample or repeat to match length of original training set
        if target_len is not None:
            def sample_to_length(arr, L):
                if len(arr) == 0:
                    return np.array([''] * L)
                if len(arr) >= L:
                    idx = np.random.choice(len(arr), L, replace=False)
                else:
                    # sample with replacement to reach desired length
                    idx = np.random.choice(len(arr), L, replace=True)
                return np.array(arr)[idx]

            x_train_res1 = sample_to_length(x_train_res1, target_len)
            x_train_res2 = sample_to_length(x_train_res2, target_len)

        # sentiment files likely don't have fine-grain veracity attributions — use zeros
        n1 = len(x_train_res1)
        # create dummy zero arrays with 4 columns to match expected fine-grain shape
        y_train_fg = np.zeros((n1, 4))
        y_train_fg_m = np.zeros((n1, 4))
        y_train_fg_t = np.zeros((n1, 4))

        return x_train_res1, x_train_res2, y_train_fg, y_train_fg_m, y_train_fg_t

    else:
        raise ValueError('Unknown reframing mode: {}'.format(mode))
