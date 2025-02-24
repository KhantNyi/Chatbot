import pandas as pd
import json

# Function to generate JSON from course data
def generate_json(df):
    json_output = []
    
    for _, row in df.iterrows():
        course_id = row["Courses"]
        course_name = row["Title"]
        prereq_64 = row["Prerequisite for 621 - 643"] if pd.notna(row["Prerequisite for 621 - 643"]) else ""
        prereq_65 = row["Prerequisite for 651 - 652"] if pd.notna(row["Prerequisite for 651 - 652"]) else ""
        
        # Infer program from course_id prefix
        if course_id.startswith("CSX"):
            program = "CS"
        elif course_id.startswith("ITX"):
            program = "IT"
        else:
            program = "Unknown"  # Adjust this logic based on your full dataset
        
        # Process for year 64 (621–643)
        prereqs_64 = prereq_64.strip()
        if prereqs_64:
            # Split prerequisites if multiple are listed with "and"
            prereq_list = [p.strip() for p in prereqs_64.split(" and ")]
        else:
            prereq_list = ["None"]
        
        json_entry_64 = {
            "id": f"64-{program}-{course_id}",
            "text": f"Course {course_id} ({course_name}) in the {program} program, for 64. Prerequisites: {prereqs_64 or 'None'}.",
            "metadata": {
                "year": "64",
                "program": program,
                "course_id": course_id,
                "course_name": course_name,
                "prerequisites": prereq_list
            }
        }
        json_output.append(json_entry_64)
        
        # Process for year 65 (651–652)
        prereqs_65 = prereq_65.strip()
        if prereqs_65:
            # Split prerequisites if multiple are listed with "and"
            prereq_list = [p.strip() for p in prereqs_65.split(" and ")]
        else:
            prereq_list = ["None"]
        
        json_entry_65 = {
            "id": f"65-{program}-{course_id}",
            "text": f"Course {course_id} ({course_name}) in the {program} program, for 65. Prerequisites: {prereqs_65 or 'None'}.",
            "metadata": {
                "year": "65",
                "program": program,
                "course_id": course_id,
                "course_name": course_name,
                "prerequisites": prereq_list
            }
        }
        json_output.append(json_entry_65)
    
    return json_output

# Read the Excel file
file_path = r""  # Replace with your actual file path
df = pd.read_excel(file_path)

# Generate JSON
result = generate_json(df)

# Save to a file
with open("courses.json", "w") as f:
    json.dump(result, f, indent=2)
