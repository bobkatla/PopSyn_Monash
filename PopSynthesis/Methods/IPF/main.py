"""
The main file to do the work of generating using IPF-based
"""
from paras import loc_data, loc_controls, loc_output
from PopSynthesis.Methods.IPF.src.IPF import eval_based_on_full_pop


def main():
    results = eval_based_on_full_pop(loc_data=loc_data)
    print(results)


if __name__ == "__main__":
    main()