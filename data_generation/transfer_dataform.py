
import json

def generate_combined_output(example_reaction_data):
    outputs = []
    combined_texts = []
    x_min = y_min = float('inf')
    x_max = y_max = 0
    has_agent = False
    has_all_conditions = False

    for bbox in example_reaction_data["bboxes"]:
        if 'label' not in bbox:
            continue
        img_path = example_reaction_data["file_name"]
        dataset_name = "reaction_image_OCR"
        height = example_reaction_data["height"]
        width = example_reaction_data["width"]
        
        # Extract text and labels from bounding box
        bbox_text = sum(bbox.get("text", []), [])
        label = bbox.get("label", "")
        expression_parts = []

        # Determine the correct verb to use based on the count of items
        verb = "is" if len(bbox_text) == 1 else "are"

        if bbox_text:
            combined_texts.append(f"{', '.join(bbox_text)}")
            expression_parts.append(f"{', '.join(bbox_text)} {verb} written in the box")

        if label == "agent":
            # Generate output for agent
            has_agent = True
            expression_parts.append(f"{', '.join(bbox_text)} {verb} agent")
        
        if label == "all_conditions":
            has_all_conditions = True
            # Generate output for all_conditions
            if bbox["text"][0]:
                solvents = ', '.join(bbox["text"][0])
                solvent_verb = "is" if len(bbox["text"][0]) == 1 else "are"
                expression_parts.append(f"{solvents} {solvent_verb} solvent")

            for item in bbox["text"][1]:
                item_verb = "is"
                if item.endswith('h'):
                    expression_parts.append(f"{item} {item_verb} time")
                elif item.endswith('%'):
                    expression_parts.append(f"{item} {item_verb} yield")
                else:
                    expression_parts.append(f"{item} {item_verb} temperature")

        # Assemble the full expression from parts and update bounding box coordinates
        full_expression = "; ".join(expression_parts)
        x1, y1, x2, y2 = bbox["bbox"]
        x_min, y_min = min(x_min, x1), min(y_min, y1)
        x_max, y_max = max(x_max, x2), max(y_max, y2)

        outputs.append({
            "img_path": img_path,
            "expression": full_expression,
            "bbox": [x1, y1, x2, y2],
            "dataset_name": dataset_name,
            "height": height,
            "width": width
        })

    # Combine all expressions into one
    combined_expression = f"{', '.join(combined_texts)} are written in the box; " + "; ".join(part for output in outputs for part in output["expression"].split("; ")[1:])
    combined_output = {
        "img_path": img_path,
        "expression": combined_expression,
        "bbox": [x_min, y_min, x_max, y_max],
        "dataset_name": dataset_name,
        "height": height,
        "width": width
    }
    if has_agent and has_all_conditions:
        outputs.append(combined_output)

    return outputs, combined_output



def process_data_and_write_jsonl(input_file_path, output_file_path):
    # Load the JSON data from the provided file
    with open(input_file_path, 'r') as file:
        data = json.load(file)

    # List to hold all processed entries
    processed_entries = []

    # Process each entry in the JSON data
    for entry in data:
        # Generate outputs for entries that have labeled boxes
        try:
            outputs, combined_output = generate_combined_output(entry)
            print(outputs, combined_output)
            # Append each processed entry to the list
            processed_entries.extend(outputs)
            #processed_entries.extend(combined_output)
        except: pass

    # Write the processed entries to a JSONL file
    with open(output_file_path, 'w') as outfile:
        for processed_entry in processed_entries:
            json.dump(processed_entry, outfile)
            outfile.write('\n')  # Write a newline character after each JSON object

    return f"Processed data has been written to {output_file_path}"

# Assuming the function `generate_output` is defined as before and modifies the entries correctly
# Let's execute the function with the paths to the input and a new output file
input_path = 'output_single_20240422.json'
output_path = 'train_OCR.jsonl'

# Call the function to process the data and write to a JSONL file
process_result = process_data_and_write_jsonl(input_path, output_path)


# # Use the function to generate outputs based on the corrected and improved data handling logic
# updated_example_reaction_data = {
#     "id": 1403,
#     "width": 1333,
#     "height": 450,
#     "file_name": "US20010000038A1_0256.png",
#     "license": 0,
#     "bboxes": [
#         {"id": 0, "bbox": [9, 169, 243, 288], "category_id": 1},
#         {"id": 1, "bbox": [303, 159, 563, 265], "category_id": 1},
#         {"id": 2, "bbox": [905, 177, 1321, 332], "category_id": 1},
#         {"id": 3, "bbox": [595, 149, 873, 207], "category_id": 2, "label": "agent", "text": [["methylene chloride", "aluminum chloride"]]},
#         {"id": 4, "bbox": [572, 242, 890, 375], "category_id": 2, "label": "all_conditions","text": [["ice", "methylene chloride"], ["30-40C","14.0h", "86.0%"]]}
#     ],
#     "reactions": [
#         {"reactants": [0, 1], "conditions": [3, 4], "products": [2]}
#     ],
#     "corefs": [],
#     "caption": "",
#     "pdf": {},
#     "diagram_type": "single"
# }

# # output_results = generate_output(updated_example_reaction_data)
# # combined_output = combine_output(updated_example_reaction_data)
# # print(output_results)
# # print(combined_output)


# individual_outputs, combined_output = generate_combined_output(updated_example_reaction_data)
# print(individual_outputs, combined_output)