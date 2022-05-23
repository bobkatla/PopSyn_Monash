from math import sqrt
import itertools
import time
import pandas as pd


def SRMSE(actual, pred):
    '''
    This calculate the SRMSE for 2 pandas.dataframe based on the list of their attributes
    The actual has to have the same or more columns than pred
    This will compare only the one that is in pred's colls
    BUT now I make it return -1 if they are not match (at least in length)
    '''
    if len(actual.columns) != len(pred.columns):
        return None
    start_time = time.time()

    total_att = 1
    full_list = {}
    attributes = pred.columns

    # Get the possible values for each att
    for att in attributes:
        possible_values = actual[att].unique()
        total_att *= len(possible_values)
        full_list[att] = possible_values

    # Generate all the possible combinations
    keys, values = zip(*full_list.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    #calculate
    hold = 0

    for instance in combinations:
        check_actual = None
        check_pred = None
        to_ttext = []
        for att in attributes:
            to_ttext.append(instance[att])
            if check_actual is not None:
                check_actual &= (actual[att] == instance[att])
                check_pred &= (pred[att] == instance[att])
            else:
                check_actual = (actual[att] == instance[att])
                check_pred = (pred[att] == instance[att])
        # This assumes that there will always be False result
        freq_actual = 1 - ((check_actual.value_counts()[False]) / len(check_actual))
        freq_pred = 1 - ((check_pred.value_counts()[False]) / len(check_pred))
        hold += (freq_actual - freq_pred)**2

    result = sqrt(hold * total_att)

    duration = time.time() - start_time
    print(f"Calculating the SRMSE for {len(attributes)} atts in {duration} seconds")

    return result

def update_SRMSE(actual, pred):
    if len(actual.columns) != len(pred.columns):
        return None
    
    start_time = time.time()
    attributes = actual.columns
    pred = pred[attributes]
    actual_vals = actual.value_counts(normalize=True)
    pred_vals = pred.value_counts(normalize=True)

    ls_pos_vals = []
    # Get the possible values for each att
    for att in attributes:
        possible_values = actual[att].unique()
        ls_pos_vals.append(set(possible_values))
    
    hold = 0
    all_com = pd.MultiIndex.from_product(ls_pos_vals, names=attributes)
    for com in all_com:
        actual_freq = actual_vals[com] if com in actual_vals else 0
        pred_freq = pred_vals[com] if com in pred_vals else 0
        hold += (actual_freq - pred_freq)**2
            
    result = sqrt(hold * len(all_com))
    duration = time.time() - start_time
    print(f"Calculating the SRMSE for {len(attributes)} atts in {duration} seconds")
    return result
