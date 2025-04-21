COUNT_COL = "count"
MAIN_PERSON = "Main"
HH_TAG = "HH"
EXPECTED_RELATIONSHIPS = [
    "Spouse",
    "Child",
    "Parent",
    "Grandparent",
    "Grandchild",
    "Sibling",
    "Others",
] + [MAIN_PERSON]
EPXECTED_CONNECTIONS = [
    # there is order here to ensure what is sampled first
    (HH_TAG, MAIN_PERSON),
    (MAIN_PERSON, "Spouse"),
    (MAIN_PERSON, "Child"),
    ("Child", "Grandchild"),
    (MAIN_PERSON, "Parent"),
    ("Parent", "Grandparent"),
    (MAIN_PERSON, "Grandparent"),
    (MAIN_PERSON, "Grandchild"),
    (MAIN_PERSON, "Sibling"),
    (MAIN_PERSON, "Others"),
]
BACK_CONNECTIONS = {
    HH_TAG: None, # top
    MAIN_PERSON: [HH_TAG],
    "Spouse": [MAIN_PERSON],
    "Child": [MAIN_PERSON],
    "Parent": [MAIN_PERSON],
    "Grandparent": ["Parent", MAIN_PERSON],
    "Grandchild": ["Child", MAIN_PERSON],
    "Sibling": [MAIN_PERSON],
    "Others": [MAIN_PERSON],
}
RELA_BY_RELA = [
    [MAIN_PERSON],
    ["Spouse", "Child", "Parent", "Sibling", "Others"],
    ["Grandparent", "Grandchild"]
]