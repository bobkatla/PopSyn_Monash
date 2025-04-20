"""From given paired pools, process them by demands which can be keeping the original or use generated ones like BN"""


def process_original_pools(ori_pools, method="original"):
    if method == "original":
        return ori_pools
    if method == "BN":
        # basically inflate to get results
        NotImplemented
    if method == "WGAN":
        NotImplemented
