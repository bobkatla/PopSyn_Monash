"""Create mock data for quick testing, small sample of populations with the known activities"""

import pandas as pd


def create_mock_data():
    # Create people dataframe, not random, with both person and household attributes
    people = pd.DataFrame({
        "person_id": [1, 2, 3, 4, 5],
        "household_id": [1, 1, 2, 2, 3],
        "age": [20, 21, 22, 23, 24],
        "gender": ["male", "female", "male", "female", "male"],
        "employment_status": ["employed", "unemployed", "employed", "employed", "unemployed"],
        "education_level": ["high_school", "university", "high_school", "university", "high_school"],
    })
    
    # Create locations dataframe, not random, with location attributes
    locations = pd.DataFrame({
        "location_id": [1, 2, 3, 4, 5],
        "attractiveness": [0.6, 0.5, 0.5, 0.5, 0.5],
    })

    # Create activity dataframe, not random, tell activity type and location and sequence in a day
    activity_df = pd.DataFrame({
        "person_id_in_sequence": [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5],
        # last activity is home
        "start_time": [8, 13, 17, 9, 16, 8, 18, 14, 18, 8, 13, 17],  # Times in 24h format
        "purpose": ["work", "shopping", "home", "shopping", "home", "school", "home", "leisure", "home", "work", "shopping", "home"],
        # 5 locations
        "destination_location_id": [1, 2, 3, 4, 3, 1, 2, 1, 2, 5, 1, 2],
        "joint_activity": [False, False, False, False, False, False, False, True, True, False, False, False],
    })

    # Create location edges, they are all connected to each other
    location_edges = pd.DataFrame({
        "location_id_1": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        "location_id_2": [2, 3, 4, 1, 3, 5, 1, 2, 4, 1, 2, 3, 1, 2, 3],
        "distance": [1, 1, 3, 1, 1, 1, 2, 1, 5, 1, 9, 7, 1, 1, 5],
        "travel_time": [1, 1, 3, 1, 1, 1, 2, 1, 5, 1, 9, 7, 1, 1, 5],
    })

    return people, locations, activity_df, location_edges

