from collections import defaultdict
from PopSynthesis.Methods.connect_HH_PP.scripts.const import LS_GR_RELA, HANDLE_THE_REST_RELA
import polars as pl
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Union, Literal

MIN_PARENT_CHILD_GAP = 16
MIN_GRANDPARENT_GRANDCHILD_GAP = 33
MIN_COUPLE_GAP = 20 # This only apply when we do the conversion


class Person:
    def __init__(self, id: str, age: Any, sex: Literal["M", "F"], income: int, relationship: str) -> None:
        self.id: str = id
        self.age: int = int(age)
        self.is_male: bool = True if sex == "M" else False
        self.income: int = int(income)
        self.relationship: str = relationship

    
class Household:
    AVAILABLE_RELATIONSHIPS = ["Main", "Spouse", "Child", "Grandchild", "Sibling", "Others"]

    def __init__(self, persons: List[Person]) -> None:
        self.persons = persons
        # replace for all
        self.persons[0].relationship = "Main"
        self.initialize()

    def initialize(self) -> None:
        self.segment_by_rela = {relationship: [] for relationship in self.AVAILABLE_RELATIONSHIPS}
        for person in self.persons:
            assert person.relationship != "Self" # this should no longer exist
            self.segment_by_rela[person.relationship].append(person)
        assert len(self.segment_by_rela["Main"]) == 1
        self.main_person: Person = self.segment_by_rela["Main"][0]

        self.highest_income_person: Person = self.get_highest_income_person()
        self.is_implausible: bool = self.check_implausible()

    def get_highest_income_person(self) -> Person:
        converted_incomes = [p.income for p in self.persons] # same order as given persons
        idx_highest = idx_max_val_return(converted_incomes)
        return self.persons[idx_highest]
    
    def check_implausible(self) -> bool:
        # Check Child and GrandChild
        main_age = self.main_person.age
        is_implausible_case = False
        for person in self.persons:
            age_gap = abs(main_age - person.age)
            implausible_child = person.relationship == "Child" and age_gap < MIN_PARENT_CHILD_GAP
            implausible_grandchild =  person.relationship == "Grandchild" and age_gap < MIN_GRANDPARENT_GRANDCHILD_GAP
            if implausible_child or implausible_grandchild:
                is_implausible_case = True
                break
        return is_implausible_case

    def get_mapping_results_current(self) -> Dict[str, str]:
        return {p.id: p.relationship for p in self.persons}

    def get_person_idx(self, person_id: str) -> int:
        ordered_ids = [p.id for p in self.persons]
        possible_idx = find_idx_value(ordered_ids, person_id)
        assert len(possible_idx) == 1 # make sure it is unique
        return possible_idx[0]
        
    def update_relationship(self, person_id:str, new_relationship:str) -> None:
        target_person_idx = self.get_person_idx(person_id)
        self.persons[target_person_idx].relationship = new_relationship

    def update_all(self) -> None:
        self.initialize()


def idx_max_val_return(values: List[Any]) -> int:
    assert len(values) > 0
    max_idx = None
    max_val = None
    for idx, val in enumerate(values):
        if max_val is None or val > max_val:
            max_idx = idx
            max_val = val
    return max_idx


def find_idx_value(values: List[Any], find_val: Any) -> List[int]:
    results: List[Any] = []
    for idx, val in enumerate(values):
        if val == find_val:
            results.append(idx)
    return results


def check_rela_gb(gb_df: pd.DataFrame) -> None:
    for hhid, rela_gr in zip(gb_df.index, gb_df):
        check_dict = defaultdict(lambda: 0)
        for i in rela_gr:
            check_dict[i] += 1
        if check_dict["Self"] == 0:
            # print(hhid)
            print([f"{x} - {y}" for x, y in check_dict.items() if x != "Self"])
        elif check_dict["Self"] > 1:
            print("NOOOOOOOOOO", hhid, rela_gr)

    
def process_not_accept_values(pp_df: pd.DataFrame) -> pd.DataFrame:
    # Remove not accept value
    # At the moment we remove Null and Missing for income
    pp_df = pp_df.drop_nulls()
    pp_df_missing = pp_df.filter(pl.col("persinc")=="Missing/Refused")
    to_rm_hhid = list(pp_df_missing["hhid"].unique())
    pp_df = pp_df.filter(~pl.col("hhid").is_in(to_rm_hhid))
    return pp_df


def convert_simple_income(income_str: str) -> int:
    if "Negative" in income_str:
        return -1
    elif "Missing" in income_str:
        # This should not happen as we will filter no income
        return -2
    elif "Zero" in income_str:
        return 0
    elif "-" in income_str:
        return int(income_str.split("-")[0].replace("$", ""))
    elif "+" in income_str:
        return 2000
    else:
        raise ValueError("Weird")


def determine_any_be_spouse(others_idx, persons: List[Person]):
    # determine the spouse to be
    NotImplemented


def get_new_mapping_rela_based(household: Household) -> Union[None, Dict[str, str]]:
    # Find the ls idx of all possible relationship
    main_id = household.main_person.id
    highest_income_id = household.highest_income_person.id
    highest_inc_rela = household.highest_income_person.relationship
    household.update_relationship(highest_income_id, "Main")
    if highest_inc_rela == "Main":
        return None # No change needed
    elif highest_inc_rela == "Spouse":
        # Simple swap, rest is the same
        household.update_relationship(main_id, "Spouse")
    else:
        NotImplemented
        # elif highest_inc_rela == "Child":
        #     # NOTE: Assume that NO ONE consider their in-law their Child
        #     # If simple no other and no Grandchild we can simply swap, other Child will become Sibling
        #     # If Granchild but no Other, we check the gap, and if fit, it's the child, not then Others
        #     results_mapping.update({main_id: "Parent"})
        #     for spouse_idx in spouse_idxs:
        #         results_mapping.update({persons[spouse_idx].id: "Parent"})
        #     for child_idx in child_idxs:
        #         if persons[child_idx].id != highest_income_id:
        #             results_mapping.update({persons[child_idx].id: "Sibling"})
        #     for sibling_idx in sibling_idxs:
        #         results_mapping.update({persons[sibling_idx].id: "Others"})
        #     for other_idx in others_idxs:
        #         # We need to find can they be Spouse
        #         NotImplemented
            
    #     # If Others, we check the age gap and sex, if fit then turn them to Spouse (assert that the Child is above 18)
    #     # Check the Grandchild, if 
    household.update_all()
    return household.get_mapping_results_current()


def process_chosen_to_others(pp_data: pd.DataFrame, to_others_rela: List[str] = ["Other relative", "Unrelated", "Other"], other_name:str = "Others") -> pd.DataFrame:
    # Maybe let's keep sibling, as sibling defo cannot be spouse, and they maybe special
    filter_data_to_process = pp_data[pp_data["relationship"].isin(to_others_rela)]
    to_process_pp_data = pp_data.copy(deep=True) # ensure we don't modify the original data
    to_process_pp_data.loc[filter_data_to_process.index, "relationship"] = other_name
    return to_process_pp_data


def process_rela(pp_df: pl.DataFrame):
    # To handle relationship, generally we based on income, age and gender
    # Due to complexity, I will keep pandas here for now
    pp_df: pd.DataFrame = pp_df.to_pandas()
    pp_df = process_chosen_to_others(pp_df) # Note, it's Others not Other, minor to help tell diff

    pp_df["converted_income"] = pp_df["persinc"].apply(convert_simple_income)
    pp_df["combine_person_obj"] = pp_df.apply(lambda r: Person(r["persid"], r["age"], r["sex"], r["converted_income"], r["relationship"]), axis=1)
    gb_pid = pp_df.groupby("hhid")["combine_person_obj"].apply(lambda x: list(x))
    converted_hh_results = pd.DataFrame(gb_pid)
    converted_hh_results["hh_obj"] = converted_hh_results["combine_person_obj"].apply(lambda x: Household(x))
    converted_hh_results["implausible_case"] = converted_hh_results["hh_obj"].apply(lambda x: x.is_implausible)

    # Now we need to remove hh that has implauble case
    implausible_hh = converted_hh_results[converted_hh_results["implausible_case"]]
    print(f"Remove {len(implausible_hh)} households with implausible combinations")
    filtered_cases = converted_hh_results[~converted_hh_results["implausible_case"]]

    # Still need to think about how to handle case of Other
    filtered_cases["map_new_rela"] = filtered_cases["hh_obj"].apply(get_new_mapping_rela_based)
    print(filtered_cases)
    # And check for all of them the highest income is Main, again
    # return pp_df
