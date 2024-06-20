from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from demo1 import generate_indigo_image

def generate_json_from_multiple_reactions_with_render(i, title, results):
    # Initialize the main dictionary structure
    output_dict = {
        "id": i + 1378,
        "width": 1333,
        "height": 500,
        "file_name": f"{title}.png",
        "license": 0,
        "bboxes": [],
        "reactions": [],
        "corefs": [],
        "caption": "",
        "pdf": {},
        "diagram_type": "single"
    }

    unique_id = 0
    previous_products_ids = []
    next_reactants_ids = []

    for result in results:
        reactants_ids = []
        products_ids = []
        conditions_ids = []

        # Reactants
        for reactant_bbox in result['reactants']:
            reactant_data = {
                "id": unique_id,
                "bbox": reactant_bbox[0] + reactant_bbox[1],
                "category_id": 1
            }
            output_dict['bboxes'].append(reactant_data)
            reactants_ids.append(unique_id)
            unique_id += 1

        # Products
        for product_bbox in result['products']:
            product_data = {
                "id": unique_id,
                "bbox": product_bbox[0] + product_bbox[1],
                "category_id": 1
            }
            output_dict['bboxes'].append(product_data)
            products_ids.append(unique_id)
            unique_id += 1

        # Conditions
        for condition_type, condition_bbox in result['conditions'].items():
            if condition_type in ['agent', 'all_conditions']:
                condition_data = {
                    "id": unique_id,
                    "bbox": condition_bbox[0] + condition_bbox[1],
                    "category_id": 2
                }
                output_dict['bboxes'].append(condition_data)
                conditions_ids.append(unique_id)
                unique_id += 1

        render_ids = []
        for render_key in ['render_agent', 'render_solvent']:
            if render_key in result['conditions']:
                render_data = {
                    "id": unique_id,
                    "bbox": result['conditions'][render_key][0] + result['conditions'][render_key][1],
                    "category_id": 2
                }
                output_dict['bboxes'].append(render_data)
                render_ids.append(unique_id)
                unique_id += 1
        output_dict['reactions'].append({
            "reactants": reactants_ids,
            "conditions": conditions_ids,
            "products": products_ids})

        if next_reactants_ids:
            output_dict['reactions'].append({
                "reactants": products_ids,
                "conditions": render_ids,
                "products": next_reactants_ids
            })
        # Populating reactions and corefs



        # If there's a render_agent or render_solvent, create a new reaction connecting previous products to current reactants



        previous_products_ids = products_ids
        next_reactants_ids = reactants_ids
    # Currently, corefs is empty as we don't have the identifiers
    output_dict['corefs'] = []

    return output_dict

def generate_json_from_multiple_reactions_with_render2(i, title, results):
    # Initialize the main dictionary structure
    output_dict = {
        "id": i + 1378,
        "width": 1333,
        "height": 500,
        "file_name": f"{title}.png",
        "license": 0,
        "bboxes": [],
        "reactions": [],
        "corefs": [],
        "caption": "",
        "pdf": {},
        "diagram_type": "single"
    }

    unique_id = 0
    all_reactants_ids = []
    all_products_ids = []

    # First pass: Assign IDs for all reactants, products and conditions
    for result in results:
        reactants_ids = []
        products_ids = []

        # Reactants
        for reactant_bbox in result['reactants']:
            reactant_data = {
                "id": unique_id,
                "bbox": reactant_bbox[0] + reactant_bbox[1],
                "category_id": 1
            }
            output_dict['bboxes'].append(reactant_data)
            reactants_ids.append(unique_id)
            unique_id += 1

        # Products
        for product_bbox in result['products']:
            product_data = {
                "id": unique_id,
                "bbox": product_bbox[0] + product_bbox[1],
                "category_id": 1
            }
            output_dict['bboxes'].append(product_data)
            products_ids.append(unique_id)
            unique_id += 1

        all_reactants_ids.append(reactants_ids)
        all_products_ids.append(products_ids)

    # Second pass: Build reactions and connecting reactions
    for index, result in enumerate(results):
        conditions_ids = []
        # Conditions
        for condition_type, condition_bbox in result['conditions'].items():
            if condition_type in ['agent', 'all_conditions']:
                condition_data = {
                    "id": unique_id,
                    "bbox": condition_bbox[0] + condition_bbox[1],
                    "category_id": 2
                }
                output_dict['bboxes'].append(condition_data)
                conditions_ids.append(unique_id)
                unique_id += 1

        render_ids = []
        for render_key in ['render_agent', 'render_solvent']:
            if render_key in result['conditions']:
                render_data = {
                    "id": unique_id,
                    "bbox": result['conditions'][render_key][0] + result['conditions'][render_key][1],
                    "category_id": 2
                }
                output_dict['bboxes'].append(render_data)
                render_ids.append(unique_id)
                unique_id += 1

        # Appending current reaction details
        output_dict['reactions'].append({
            "reactants": all_reactants_ids[index],
            "conditions": conditions_ids,
            "products": all_products_ids[index]})

        # Check for the next reaction's reactants and create a new reaction
        if index < len(results) - 1:  # Ensure we aren't on the last reaction
            output_dict['reactions'].append({
                "reactants": all_products_ids[index],
                "conditions": render_ids,
                "products": all_reactants_ids[index + 1]
            })

    # Currently, corefs is empty as we don't have the identifiers
    output_dict['corefs'] = []

    return output_dict


# Sample execution for the provided data
sample_results = [{'reactants': [[(11, 139), (252, 372)], [(312, 132), (609, 221)]],
                   'products': [[(809, 54), (1313, 303)]],
                   'conditions':
                       {'agent': ((678, 182), (740, 210)),
                        'solvent': ((667, 232), (751, 260)),
                        'temperature': ((610, 267), (679, 295)), 'time': ((618, 307), (704, 335)),
                        'all_conditions': ((610, 232), (751, 335)), 'arrow': ((619, 225), (799, 225)),
                        'render_agent': ((619, 182), (679, 210))}
                        #'render_solvent': ((619, 232), (704, 260))}
                   },

                    {'reactants': [[(11, 139), (252, 372)], [(312, 132), (609, 221)]],
                   'products': [[(809, 54), (1313, 303)]],
                   'conditions':
                       {'agent': ((678, 182), (740, 210)),
                        'solvent': ((667, 232), (751, 260)),
                        'temperature': ((610, 267), (679, 295)), 'time': ((618, 307), (704, 335)),
                        'all_conditions': ((610, 232), (751, 335)), 'arrow': ((619, 225), (799, 225)),
                        'render_agent': ((620, 182), (679, 210)),
                        'render_solvent': ((620, 232), (704, 260))}
                     },

                  {'reactants': [[(232, 546), (440, 722)], [(500, 650), (649, 684)]],
                   'products': [[(1123, 562), (1314, 740)]],
                   'conditions': {'agent': ((664, 602), (1108, 660)),
                                  'solvent': ((820, 682), (952, 740)),
                                  'temperature': ((802, 747), (860, 775)),
                                  'time': ((881, 747), (952, 775)),
                                  'all_conditions': ((802, 682), (952, 775))},
                   'arrow': ((659, 675), (1113, 675))
                   }

                  ]


sample_output = generate_json_from_multiple_reactions_with_render2(0, "sample_title", sample_results)
print(sample_output)




