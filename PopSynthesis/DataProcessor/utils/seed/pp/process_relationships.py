from collections import defaultdict
import polars as pl
import pandas as pd
from typing import List, Dict, Any, Literal

MIN_PARENT_CHILD_GAP = 15
MIN_GRANDPARENT_GRANDCHILD_GAP = 33
MAX_COUPLE_GAP = 20  # This only apply when we do the conversion
MIN_PERMITTED_AGE_MARRIED = 16


class Person:
    def __init__(
        self, id: str, age: Any, sex: Literal["M", "F"], income: int, relationship: str
    ) -> None:
        self.id: str = id
        self.age: int = int(age)
        self.sex: Literal["M", "F"] = sex
        self.income: int = int(income)
        self.relationship: str = relationship

    def __str__(self) -> str:
        return f"{self.id}-{self.age}-{self.sex}-{self.income}-{self.relationship}"


class Household:
    AVAILABLE_RELATIONSHIPS = [
        "Main",
        "Spouse",
        "Child",
        "Grandchild",
        "Sibling",
        "Others",
        "Parent",
        "Grandparent",
    ]

    def __init__(self, persons: List[Person]) -> None:
        self.persons = persons
        # replace for all
        self.persons[0].relationship = "Main"
        self.initialize()

    def __str__(self) -> str:
        return f"{len(self.persons)} people"

    def initialize(self) -> None:
        self.segment_by_rela = {
            relationship: [] for relationship in self.AVAILABLE_RELATIONSHIPS
        }
        for person in self.persons:
            assert person.relationship != "Self"  # this should no longer exist
            self.segment_by_rela[person.relationship].append(person)
        assert len(self.segment_by_rela["Main"]) == 1
        self.main_person: Person = self.segment_by_rela["Main"][0]

        self.highest_income_person: Person = self.get_highest_income_person()
        self.is_implausible: bool = self.check_implausible()

    def get_highest_income_person(self) -> Person:
        converted_incomes = [
            p.income for p in self.persons
        ]  # same order as given persons
        idx_highest = idx_max_val_return(converted_incomes)
        highest_inc_person = self.persons[idx_highest]
        assert (
            not self.main_person.income > highest_inc_person.income
        )  # if wrong, sth wrong with the func
        if self.main_person.income == highest_inc_person.income:
            highest_inc_person = (
                self.main_person
            )  # This is to ensure we prefer Main for highest income
        return highest_inc_person

    def check_implausible(self) -> bool:
        # Check Child and GrandChild
        main_age = self.main_person.age
        is_implausible_case = False
        for person in self.persons:
            age_gap = abs(main_age - person.age)
            implausible_child = (
                person.relationship == "Child" and age_gap < MIN_PARENT_CHILD_GAP
            )
            implausible_grandchild = (
                person.relationship == "Grandchild"
                and age_gap < MIN_GRANDPARENT_GRANDCHILD_GAP
            )
            implausible_parent = (
                person.relationship == "Parent" and age_gap < MIN_PARENT_CHILD_GAP
            )
            implausible_grandparent = (
                person.relationship == "Grandparent"
                and age_gap < MIN_GRANDPARENT_GRANDCHILD_GAP
            )
            if (
                implausible_child
                or implausible_grandchild
                or implausible_parent
                or implausible_grandparent
            ):
                is_implausible_case = True
                break
        return is_implausible_case

    def get_mapping_results_current(self) -> Dict[str, str]:
        return {p.id: p.relationship for p in self.persons}

    def get_person_idx(self, person_id: str) -> int:
        ordered_ids = [p.id for p in self.persons]
        possible_idx = find_idx_value(ordered_ids, person_id)
        assert len(possible_idx) == 1  # make sure it is unique
        return possible_idx[0]

    def update_relationship(self, person_id: str, new_relationship: str) -> None:
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
    pp_df_missing = pp_df.filter(pl.col("persinc") == "Missing/Refused")
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


def get_new_update_household(household: Household) -> Household:
    # Find the ls idx of all possible relationship
    main_id = household.main_person.id
    highest_inc_person = household.highest_income_person
    highest_income_id = highest_inc_person.id
    highest_inc_rela = highest_inc_person.relationship
    household.update_relationship(highest_income_id, "Main")

    if household.is_implausible or highest_inc_rela == "Main":
        return household  # No change needed

    elif highest_inc_rela == "Spouse":
        # Simple swap, sibling to Other, rest is the same
        household.update_relationship(main_id, "Spouse")
        for sibling in household.segment_by_rela["Sibling"]:
            household.update_relationship(sibling.id, "Others")

    elif highest_inc_rela == "Child":
        num_spouse = 0
        # NOTE: Assume that NO ONE consider their in-law their Child
        household.update_relationship(main_id, "Parent")
        # Spouse would be Parent
        for spouse in household.segment_by_rela["Spouse"]:
            household.update_relationship(spouse.id, "Parent")
        # Child would be Sibling
        for child in household.segment_by_rela["Child"]:
            if child.id != highest_income_id:
                household.update_relationship(child.id, "Sibling")
        # Sibling would be uncle/aunt -> Others
        for sibling in household.segment_by_rela["Sibling"]:
            household.update_relationship(sibling.id, "Others")
        # Others can be others or Spouse
        for other in household.segment_by_rela["Others"]:
            # NOTE: for this conversion we made a lot of assumptions for better conversion
            if num_spouse > 0:
                break
            else:
                age_gap = abs(highest_inc_person.age - other.age)
                in_permited_age = (
                    highest_inc_person.age > MIN_PERMITTED_AGE_MARRIED
                    and other.age > MIN_PERMITTED_AGE_MARRIED
                    and age_gap <= MAX_COUPLE_GAP
                )
                diff_sex = other.sex != highest_inc_person.sex
                if in_permited_age and diff_sex:
                    household.update_relationship(other.id, "Spouse")
                    num_spouse += 1
        # Grandchild can be the Child or Others
        for grandchild in household.segment_by_rela["Grandchild"]:
            # We need to check, is it the hh inc child, simple in correct age gap is fine
            age_gap = abs(highest_inc_person.age - grandchild.age)
            if age_gap < MIN_PARENT_CHILD_GAP:
                household.update_relationship(grandchild.id, "Others")
            else:
                household.update_relationship(grandchild.id, "Child")

    elif highest_inc_rela == "Sibling":
        num_spouse = 0
        # We simply ignore the parents and grandparent existence, consider no Child
        household.update_relationship(main_id, "Sibling")

        # Spouse, Child and Grandchild of the sibling would be Others
        all_to_others_pp = (
            household.segment_by_rela["Spouse"]
            + household.segment_by_rela["Child"]
            + household.segment_by_rela["Grandchild"]
        )
        for to_other_person in all_to_others_pp:
            household.update_relationship(to_other_person.id, "Others")
        # Others can be the spouse

        for other in household.segment_by_rela["Others"]:
            # NOTE: for this conversion we made a lot of assumptions for better conversion
            if num_spouse > 0:
                break
            else:
                age_gap = abs(highest_inc_person.age - other.age)
                in_permited_age = (
                    highest_inc_person.age > MIN_PERMITTED_AGE_MARRIED
                    and other.age > MIN_PERMITTED_AGE_MARRIED
                    and age_gap <= MAX_COUPLE_GAP
                )
                diff_sex = other.sex != highest_inc_person.sex
                if in_permited_age and diff_sex:
                    household.update_relationship(other.id, "Spouse")
                    num_spouse += 1

    elif highest_inc_rela == "Grandchild":
        num_spouse = 0
        num_parent = 0
        # We simply ignore the parents and grandparent existence, consider no Child
        household.update_relationship(main_id, "Grandparent")
        # Spouse would be Others
        for spouse in household.segment_by_rela["Spouse"]:
            household.update_relationship(spouse.id, "Grandparent")
        # Child would be Others
        for child in household.segment_by_rela["Child"]:
            if num_parent > 0:  # we find 1 parent only, too much assumptions
                household.update_relationship(child.id, "Others")
            else:
                age_gap = abs(highest_inc_person.age - child.age)
                if age_gap >= MIN_PARENT_CHILD_GAP:
                    household.update_relationship(child.id, "Parent")
                    num_parent += 1
        for other in household.segment_by_rela["Others"]:
            # NOTE: for this conversion we made a lot of assumptions for better conversion
            if num_spouse > 0:
                break
            else:
                age_gap = abs(highest_inc_person.age - other.age)
                in_permited_age = (
                    highest_inc_person.age > MIN_PERMITTED_AGE_MARRIED
                    and other.age > MIN_PERMITTED_AGE_MARRIED
                    and age_gap <= MAX_COUPLE_GAP
                )
                diff_sex = other.sex != highest_inc_person.sex
                if in_permited_age and diff_sex:
                    household.update_relationship(other.id, "Spouse")
                    num_spouse += 1
        for grandchild in household.segment_by_rela["Grandchild"]:
            if grandchild.id != highest_income_id:
                household.update_relationship(grandchild.id, "Sibling")

    elif highest_inc_rela == "Others":
        # Convert all to others
        for person in household.persons:
            if person.id != highest_income_id:
                household.update_relationship(person.id, "Others")
    else:
        raise ValueError(f"There is no relationship: {highest_inc_rela}")

    household.update_all()
    return household


def process_chosen_to_others(
    pp_data: pd.DataFrame,
    to_others_rela: List[str] = ["Other relative", "Unrelated", "Other"],
    other_name: str = "Others",
) -> pd.DataFrame:
    # Maybe let's keep sibling, as sibling defo cannot be spouse, and they maybe special
    filter_data_to_process = pp_data[pp_data["relationship"].isin(to_others_rela)]
    to_process_pp_data = pp_data.copy(
        deep=True
    )  # ensure we don't modify the original data
    to_process_pp_data.loc[filter_data_to_process.index, "relationship"] = other_name
    return to_process_pp_data


def process_rela(pp_df: pl.DataFrame) -> pl.DataFrame:
    # To handle relationship, generally we based on income, age and gender
    # Due to complexity, I will keep pandas here for now
    pp_df: pd.DataFrame = pp_df.to_pandas()
    to_process_pp_df = process_chosen_to_others(
        pp_df
    )  # Note, it's Others not Other, minor to help tell diff

    to_process_pp_df["converted_income"] = to_process_pp_df["persinc"].apply(
        convert_simple_income
    )
    to_process_pp_df["converted_persons"] = to_process_pp_df.apply(
        lambda r: Person(
            r["persid"], r["age"], r["sex"], r["converted_income"], r["relationship"]
        ),
        axis=1,
    )

    groups_by_hhid = to_process_pp_df.groupby("hhid")["converted_persons"].apply(
        lambda x: list(x)
    )
    converted_hh_results = pd.DataFrame(groups_by_hhid)
    converted_hh_results["converted_households"] = converted_hh_results[
        "converted_persons"
    ].apply(lambda x: Household(x))

    # Still need to think about how to handle case of Other
    converted_hh_results["converted_households"] = converted_hh_results[
        "converted_households"
    ].apply(get_new_update_household)

    # Now we need to remove hh that has implauble case
    converted_hh_results["implausible_case"] = converted_hh_results[
        "converted_households"
    ].apply(lambda x: x.is_implausible)
    implausible_hh = converted_hh_results[converted_hh_results["implausible_case"]]
    print(f"Will remove {len(implausible_hh)} households with implausible combinations")
    filtered_cases = converted_hh_results[~converted_hh_results["implausible_case"]]
    filtered_cases.loc[:, ["highest_inc_rela"]] = converted_hh_results[
        "converted_households"
    ].apply(lambda x: x.highest_income_person.relationship)
    assert (
        filtered_cases["highest_inc_rela"] == "Main"
    ).all()  # assert all highest inc is Main

    # use the mapping to new rela based on cases
    filtered_cases.loc[:, ["new_mapping_from_id"]] = converted_hh_results[
        "converted_households"
    ].apply(lambda x: x.get_mapping_results_current())
    result_mapping = {}
    for mapping in filtered_cases["new_mapping_from_id"].to_list():
        result_mapping.update(mapping)
    pp_df["relationship"] = pp_df["persid"].map(result_mapping)

    return pl.from_pandas(pp_df)
