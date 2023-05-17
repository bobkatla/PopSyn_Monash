"""
Learn BN from the given datasets
"""


def BN_l_struct(df):
    model = None
    return model


def BN_l_paras(df, model):
    NotImplemented


def BN_learn_all(df):
    BN_l_struct(df)
    model = BN_l_paras(df, model)
    return model


def test():
    # testing
    BN_model = BN_learn_all(df)


if __name__ == "__main__":
    test()
