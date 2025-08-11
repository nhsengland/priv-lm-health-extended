class Prompts:
    EDIT_PROFILE = """
    You are a synthetic profile completion assistant. 
    
    Based on the following information snippet, please carefully complete the synthetic profile for the remaining fields. They must be natural but must not contain information about actual individuals. The response for each field must be as specific as possible, and no longer than a sentence long. 
    
    first_name: {f_name}
    last_name: {l_name}
    gender: {gender}
    age: {age}
    chief_complaint: {cc}
    2nd_complaint: {second_complaint}
    
    Remaining fields:
    1. stigmatizing_health_status_and_medication
    2. stigmatizing_mental_health_status_and_medication
    3. relationship_history
    4. religious_and_spiritual_views
    5. political_views
    6. sexuality_and_sex_life
    7. lifestyle_habits_and_recreational_activities
    
    Return everything, including the provided fields, as a valid JSON with field names as keys and field values as String fields.
    """.strip()

    INJECT_INFO = """
    From the following transcript of a synthetic encounter and profile, please naturally add information into the transcript about the patient's {relationship}'s: {category}.
    
    Make sure the added information are about the patient's {relationship}.
    Only add information about the patient's {relationship}'s {category}. This information should not be relevant to the purpose of the patient's own clinical encounter.
    The information must be added in natural, multi-turn dialogue.
    You must edit the transcript only, not any other provided data.
    
    Here are the data:
    
    {profile}
    
    {transcript}
    
    Now, please return only the full edited transcript.
    """.strip()
