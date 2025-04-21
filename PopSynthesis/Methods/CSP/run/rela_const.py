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