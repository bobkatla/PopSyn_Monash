import threading
from math import sqrt
import itertools
import time
import concurrent.futures


def thread_f(instance, actual, pred, attributes):
    # the 2 would be None or not None at the same time
    check_actual = None
    check_pred = None
    for att in attributes:
        if check_actual is not None:
            check_actual &= (actual[att] == instance[att])
            check_pred &= (pred[att] == instance[att])
        else:
            check_actual = (actual[att] == instance[att])
            check_pred = (pred[att] == instance[att])
    # This assumes that there will always be False result
    freq_actual = 1 - ((check_actual.value_counts()[False]) / len(check_actual))
    freq_pred = 1 - ((check_pred.value_counts()[False]) / len(check_pred))
    
    return (freq_actual - freq_pred)**2


def SRMSE(actual, pred):
    '''
    This calculate the SRMSE for 2 pandas.dataframe based on the list of their attributes
    The actual has to have the same or more columns than pred
    This will compare only the one that is in pred's colls
    '''
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
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for instance in combinations:
            future = executor.submit(thread_f, instance, actual, pred, attributes)
            hold += future.result()

    result = sqrt(hold * total_att)

    duration = time.time() - start_time
    print(f"Calculating the SRMSE for {len(attributes)} atts in {duration} seconds")

    return result
