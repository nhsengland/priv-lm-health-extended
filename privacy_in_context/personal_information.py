from dataclasses import dataclass


@dataclass
class PersonalInformation:
    info_type = [
        "stigmatizing_health_status_and_medication",
        "stigmatizing_mental_health_status_and_medication",
        "relationship_history",
        "religious_and_spiritual_views",
        "political_views",
        "sexuality_and_sex_life",
        "lifestyle_habits_and_recreational_activities",
    ]

    relationship_type = [
        "Boss/co-worker/classmate",
        "Friend",
        "Sibling/cousin",
        "Parent",
        "Child",
        "Spouse/partner",
    ]
