from indigo import Indigo
from indigo.renderer import IndigoRenderer

indigo = Indigo()
renderer = IndigoRenderer(indigo)
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import os
import cv2
import time
import random
import re
import string
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import glob
from demo1 import generate_reaction_image,update_coordinates_for_vertical_merge,merge_all_coordinates,draw_boxes_on_reaction_image
# 加载反应

def get_bounds(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray < 255))
    if coords.size > 0:
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        #cv2.rectangle(image, (y_min, x_min), (y_max, x_max), (0, 255, 255), 2)

    return [(y_min-1, x_min-1), (y_max+1, x_max+1)]




def generate_indigo_image(smiles, mol_augment=True, default_option=False, shuffle_nodes=False, pseudo_coords=False,
                          include_condensed=True, debug=False):
    indigo = Indigo()
    renderer = IndigoRenderer(indigo)
    indigo.setOption('render-output-format', 'png')
    #indigo.setOption("render-image-size", "384,384")
    indigo.setOption('render-background-color', '1,1,1')
    indigo.setOption('render-stereo-style', 'none')
    indigo.setOption('render-label-mode', 'hetero')
    indigo.setOption('render-font-family', 'Arial')
    indigo.setOption('render-atom-ids-visible', False)
    if not default_option:
        thickness = random.uniform(1, 2)  # limit the sum of the following two parameters to be smaller than 4
        indigo.setOption('render-relative-thickness', thickness)
        indigo.setOption('render-bond-line-width', random.uniform(1, 4 - thickness))
        if random.random() < 0.5:
            indigo.setOption('render-font-family', random.choice(['Arial', 'Times', 'Courier', 'Helvetica']))
        indigo.setOption('render-label-mode', random.choice(['hetero', 'terminal-hetero']))
        indigo.setOption('render-implicit-hydrogens-visible', random.choice([True, False]))
        if random.random() < 0.1:
            indigo.setOption('render-stereo-style', 'old')
        #if random.random() < 0.2:
            #indigo.setOption('render-atom-ids-visible', True)


    mol = indigo.loadMolecule(smiles)

    opts = Draw.MolDrawOptions()
    opts.includeAtomTags = False
    mol1 = Chem.MolFromSmiles(smiles)
    mol1 = Draw.MolToImage(mol1,options=opts)
    img_np = np.array(mol1)
    img_cv2 = img_np[:, :, ::-1]
    #print(img_cv2.shape)
    bounds_cv2 = get_bounds(img_cv2)

    reaction  = renderer.renderToBuffer(mol)
    #img = renderer.renderToBuffer(reaction)
    img = cv2.imdecode(np.asarray(bytearray(reaction), dtype=np.uint8), 1)
    #print(img.shape)
    bounds = get_bounds(img)
    #cv2.imwrite("molecule4.png", mol1)
    mol1.save('molecule4.png')
    # decode buffer to image
    # img = np.repeat(np.expand_dims(img, 2), 3, axis=2)  # expand to RGB
    #cv2.rectangle(img, bounds[0], bounds[1], (0, 0, 255), 2)
    #cv2.imshow('Bound Image', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    success = True

    return img, bounds, smiles, success



def generate_reaction_image_tree1(reactants_smiles, products_smiles, condition):

    global text_size
    full_img = np.ones((450, 1333, 3), dtype=np.uint8) * 255

    reactant_data = [generate_indigo_image(r) for r in reactants_smiles]
    product_data = [generate_indigo_image(p) for p in products_smiles]

    # 获取每个分子图像的宽度
    reactant_widths = [r[1][1][0] - r[1][0][0] for r in reactant_data]
    product_widths = [p[1][1][0] - p[1][0][0] for p in product_data]

    reactant_heights = [r[1][1][1] - r[1][0][1] for r in reactant_data]
    product_heights = [p[1][1][1] - p[1][0][1] for p in product_data]
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
             cv2.FONT_HERSHEY_TRIPLEX]
    agents_font = random.choice(fonts)
    solvents_font = random.choice(fonts)
    #font = cv2.FONT_HERSHEY_SIMPLEX
    agents = condition.get('agents', [])
    all_agents_bounds = [float('inf'), float('inf'), float('-inf'), float('-inf')]  # [min_x, min_y, max_x, max_y]
    all_solvents_bounds = [float('inf'), float('inf'), float('-inf'), float('-inf')]  # [min_x, min_y, max_x, max_y]

    text_y = int(450 / 2) - 20  # 初始y坐标
    # text_size = cv2.getTextSize('sample', font, 0.9, 2)[0]  # 估算文本大小

    identifier = random.randint(1,20)

    if agents:
        for agent in agents:
            text_w = cv2.getTextSize(agent, agents_font, 0.9, 2)[0][0]
            text_size = cv2.getTextSize(agent, agents_font, 0.9, 2)[0]
            text_x = 100

            # Update bounds
            all_agents_bounds[0] = min(all_agents_bounds[0], text_x - 1)
            all_agents_bounds[1] = min(all_agents_bounds[1], text_y - text_size[1] - 3)
            all_agents_bounds[2] = max(all_agents_bounds[2], text_x + text_size[0] + 1)
            all_agents_bounds[3] = max(all_agents_bounds[3], text_y + 5)

    solvents = condition.get('solvents', [])
    text_y = int(450 / 2) + 30  # 初始y坐标
    if solvents:
        for solvent in solvents:
            text_w = cv2.getTextSize(solvent, solvents_font, 0.9, 2)[0][0]
            text_size = cv2.getTextSize(solvent, solvents_font, 0.9, 2)[0]
            text_x = 100

            all_solvents_bounds[0] = min(all_solvents_bounds[0], text_x - 1)
            all_solvents_bounds[1] = min(all_solvents_bounds[1], text_y - text_size[1] - 3)
            all_solvents_bounds[2] = max(all_solvents_bounds[2], text_x + text_size[0] + 1)
            all_solvents_bounds[3] = max(all_solvents_bounds[3], text_y + 5)

    text_lenth = max(all_agents_bounds[2]-all_agents_bounds[0],all_solvents_bounds[2]-all_solvents_bounds[0])

    # 计算总的分子图像宽度（反应物和产物）
    total_reactant_width = sum(reactant_widths) #+ 60 * (len(reactants_smiles) - 1)
    total_product_width = sum(product_widths)
    if text_lenth < 180:
        total_width = 1333 - 230 - 60 * (len(reactants_smiles) - 1) - 60 * (len(products_smiles) - 1)
    else:
        total_width = 1333 - text_lenth - 50 - 60 * (len(reactants_smiles) - 1) - 60 * (len(products_smiles) - 1)
    # 计算放缩比例，以便所有分子图像都适合在图像中
    scale_factor = total_width / (total_reactant_width + total_product_width)

    # 当前位置
    x_offset = random.randint(0,20)

    # 用于存储坐标的字典
    coords = {'reactants': [],
        'products': [],
        'conditions': {}}

    agents = condition.get('agents', [])
    all_agents_bounds = [float('inf'), float('inf'), float('-inf'), float('-inf')]  # [min_x, min_y, max_x, max_y]
    all_solvents_bounds = [float('inf'), float('inf'), float('-inf'), float('-inf')]  # [min_x, min_y, max_x, max_y]

    text_y = int(450 / 2) - 20  # 初始y坐标



    solvents = condition.get('solvents', [])
    text_y = int(450 / 2) + 30  # 初始y坐标

    # 渲染反应物
    for idx, data in enumerate(reactant_data):
        img, bounds = data[0], data[1]
        new_width = int((reactant_widths[idx]) * scale_factor)
        new_height = int((reactant_heights[idx]) * scale_factor)

        img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        new_height, new_width, _ = img_resized.shape

        # 放置图像
        y_random = random.randint(-50, 50)
        y_offset = (450 - new_height) // 2 + y_random
        full_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = img_resized
        identifier_y = y_offset + new_height + 30  # 20像素是留给标识符的空间
        cv2.putText(full_img, str(identifier), (x_offset + new_width // 2 - 10, min(identifier_y,480)), cv2.FONT_HERSHEY_SIMPLEX, random.uniform(0.7, 1.2),
                    (0, 0, 0), 2, cv2.LINE_AA)
        identifier += 1

        # 调整并记录坐标
        adjusted_bounds = [
            (x_offset, y_offset),
            (x_offset + new_width, y_offset + new_height)]
        ########################################
        coords['products'].append(adjusted_bounds)
        x_offset += new_width

        if idx < len(reactants_smiles) - 1:
            plus_y = 450 // 2
            cv2.putText(full_img, '+', (x_offset + 5, plus_y +20), random.choice(fonts), random.uniform(1, 2.5), (0, 0, 0), random.randint(1, 3),
                        cv2.LINE_AA)
            x_offset += 60  # "+"

        # 添加箭头
    if text_lenth < 180:
        arrow_start = (x_offset + 10, int(450 / 2))
        arrow_end = (x_offset + 190, int(450 / 2))
        cv2.arrowedLine(full_img, arrow_end, arrow_start, (0, 0, 0), thickness = random.randint(2, 5), tipLength=0.05)
        coords['arrow'] = ( arrow_end, arrow_start)

        x_offset += 200 # 留出空间给箭头
    else:
        arrow_start = (x_offset + 10, int(450 / 2))
        arrow_end = (x_offset + 20 + text_lenth, int(450 / 2))
        cv2.arrowedLine(full_img, arrow_end,arrow_start,  (0, 0, 0), thickness = random.randint(2, 5), tipLength=0.05)
        coords['arrow'] = (arrow_end, arrow_start)
        x_offset += (30 + text_lenth) # 留出空间给箭头

    # 渲染产物
    for idx, data in enumerate(product_data):
        img, bounds = data[0], data[1]
        new_width = int((product_widths[idx])  * scale_factor)
        new_height = int((product_heights[idx])  * scale_factor)

        img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 放置图像
        y_random = random.randint(-50, 50)
        y_offset = (450 - new_height) // 2 + y_random
        full_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = img_resized
        identifier_y = y_offset + new_height + 30  # 20像素是留给标识符的空间
        cv2.putText(full_img, str(identifier), (x_offset + new_width // 2 - 10, identifier_y), cv2.FONT_HERSHEY_SIMPLEX, random.uniform(0.7, 1.2),
                    (0, 0, 0), 2, cv2.LINE_AA)
        identifier += 1
        # 调整并记录坐标
        adjusted_bounds = [
            (x_offset, y_offset),
            (x_offset + new_width, y_offset + new_height)]
        #########################################
        coords['reactants'].append(adjusted_bounds)
        x_offset += new_width

        if idx < len(products_smiles) - 1:
            plus_y = 450 // 2
            cv2.putText(full_img, '+', (x_offset + 5, plus_y + 20 ), random.choice(fonts), 2, (0, 0, 0), 2,
                        cv2.LINE_AA)
            x_offset += 60  # "+"
    #coords['products'] = product_coords

    # 在箭头上方添加agents
    agents = condition.get('agents', [])
    all_agents_bounds = [float('inf'), float('inf'), float('-inf'), float('-inf')]  # [min_x, min_y, max_x, max_y]
    all_solvents_bounds = [float('inf'), float('inf'), float('-inf'), float('-inf')]  # [min_x, min_y, max_x, max_y]

    text_y = int(450 / 2) - 20  # 初始y坐标
    #text_size = cv2.getTextSize('sample', font, 0.9, 2)[0]  # 估算文本大小
    if agents:
        for agent in agents:
            text_w = cv2.getTextSize(agent, agents_font, 0.9, 2)[0][0]
            text_size = cv2.getTextSize(agent, agents_font, 0.9, 2)[0]
            text_x = arrow_start[0] + (arrow_end[0] - arrow_start[0]) // 2 - text_w // 2  # 居中于箭头

            # Update bounds
            all_agents_bounds[0] = min(all_agents_bounds[0], text_x - 1)
            all_agents_bounds[1] = min(all_agents_bounds[1], text_y - text_size[1] - 3)
            all_agents_bounds[2] = max(all_agents_bounds[2], text_x + text_size[0] + 1)
            all_agents_bounds[3] = max(all_agents_bounds[3], text_y + 5)

            cv2.putText(full_img, agent, (text_x, text_y), agents_font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)


            text_y -= text_size[1] + 10  # 更新y坐标
        coords['conditions']['agent'] = ((all_agents_bounds[0], all_agents_bounds[1]),
                                              (all_agents_bounds[2], all_agents_bounds[3]))
    #else:
        #coords['conditions']['agent'] = ()

    # 在箭头下方添加solvents
    solvents = condition.get('solvents', [])
    text_y = int(450 / 2) + 30  # 初始y坐标

    if solvents:
        for solvent in solvents:
            text_w = cv2.getTextSize(solvent, solvents_font, 0.9, 2)[0][0]
            text_size = cv2.getTextSize(solvent, solvents_font, 0.9, 2)[0]
            text_x = arrow_start[0] + (arrow_end[0] - arrow_start[0]) // 2 - text_w // 2  # 居中于箭头

            all_solvents_bounds[0] = min(all_solvents_bounds[0], text_x - 1)
            all_solvents_bounds[1] = min(all_solvents_bounds[1], text_y - text_size[1] - 3)
            all_solvents_bounds[2] = max(all_solvents_bounds[2], text_x + text_size[0] + 1)
            all_solvents_bounds[3] = max(all_solvents_bounds[3], text_y + 5)


            cv2.putText(full_img, solvent, (text_x, text_y), solvents_font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)


            text_y += text_size[1] + 10  # 更新y坐标

        coords['conditions']['solvent'] = ((all_solvents_bounds[0], all_solvents_bounds[1]),
                                            (all_solvents_bounds[2], all_solvents_bounds[3]))
    #else:
        #coords['conditions']['solvent'] = ()#
    # ... [略去其他部分]
    # 继续添加temperature和time
    start_x = arrow_start[0]
    total_text_width = 0
    for key in ['temperature', 'time']:
        if key in condition and condition[key]:
            text = condition[key]
            text_w = cv2.getTextSize(text + ', ', random.choice(fonts), 0.9, 1)[0][0]  # 加上逗号和空格的宽度
            total_text_width += text_w

    # 计算箭头的中点
    arrow_mid = arrow_start[0] + (arrow_end[0] - arrow_start[0]) // 2

    # 从箭头的中点开始，向左偏移半个文本的宽度
    start_x = arrow_mid - total_text_width // 2

    for i, key in enumerate(['temperature', 'time']):
        if key in condition and condition[key]:

            text = condition[key]
            text_w = cv2.getTextSize(text + ', ', random.choice(fonts), 0.9, 1)[0][0]  # 加上逗号和空格的宽度

            # 如果超出了箭头的右端，则换行
            if start_x + text_w > arrow_end[0]:
                start_x = arrow_start[0]
                text_y += text_size[1] + 20

            cv2.putText(full_img, text, (start_x, text_y + 5), random.choice(fonts), 0.9, (0, 0, 0), 2, cv2.LINE_AA)
            coords['conditions'][key] = (
            (start_x - 1, text_y + 5 - text_size[1] - 3), (start_x + text_w - 22, text_y + 5 + 5))
            start_x += text_w
            if i < 1:
                cv2.putText(full_img, ',', (start_x - 20, text_y + 5), random.choice(fonts), 0.9, (0, 0, 0), 2, cv2.LINE_AA)
                # start_x += 10
    all_conditions_bounds = [float('inf'), float('inf'), float('-inf'), float('-inf')]  # [min_x, min_y, max_x, max_y]

    for condition_key in ['solvent', 'temperature', 'time']:
        if condition_key in coords['conditions']:
            condition_bounds = coords['conditions'][condition_key]
            all_conditions_bounds[0] = min(all_conditions_bounds[0], condition_bounds[0][0])
            all_conditions_bounds[1] = min(all_conditions_bounds[1], condition_bounds[0][1])
            all_conditions_bounds[2] = max(all_conditions_bounds[2], condition_bounds[1][0])
            all_conditions_bounds[3] = max(all_conditions_bounds[3], condition_bounds[1][1])

    if all_conditions_bounds != [float('inf'), float('inf'), float('-inf'), float('-inf')]:
        coords['conditions']['all_conditions'] = ((all_conditions_bounds[0], all_conditions_bounds[1]),
                                                  (all_conditions_bounds[2], all_conditions_bounds[3]))

    if text_lenth > 450:
        return None
    else:
        return full_img, coords


def combine_reaction_images(images, upper_products_coords, lower_reactants_coords,max_overlap = 0):
    """
    Combine multiple reaction images vertically.
    :param images: list of images to combine
    :param max_overlap: maximum allowed overlap between images in pixels
    :return: combined image
    """
    total_width = max(img.shape[1] for img in images)
    total_height = sum(img.shape[0] for img in images)

    # Adjust the total height considering overlaps
    for _ in range(len(images) - 1):
        overlap = random.randint(0, max_overlap)
        total_height -= overlap

    combined_image = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

    y_offset = 0
    for img in images:
        height, width, _ = img.shape
        combined_image[y_offset:y_offset + height, :width] = img

        y_offset += height

        # Subtract a random overlap for the next image
        if len(images) > 1:
            overlap = random.randint(0, max_overlap)
            y_offset -= overlap
    print(upper_product_coords)
    midpoints_x1 = [int((coord[1][0] + coord[0][0]) / 2) for coord in upper_products_coords]
    midpoint_x1 = int(sum(midpoints_x1) / len(midpoints_x1))

    # Calculate the y value for upper products as the lowest point
    midpoint_y1 = max(coord[1][1] for coord in upper_products_coords)

    # Calculate the midpoint for lower reactants
    midpoints_x2 = [int((coord[1][0] + coord[0][0]) / 2) for coord in lower_reactants_coords]
    midpoint_x2 = int(sum(midpoints_x2) / len(midpoints_x2))

    # Calculate the y value for lower reactants as the highest point
    midpoint_y2 = min(coord[0][1] for coord in lower_reactants_coords)

    arrow_start = (midpoint_x1, midpoint_y1 + 40)
    arrow_end = (midpoint_x2, midpoint_y2 + 430)
    cv2.arrowedLine(combined_image, arrow_start, arrow_end, (0, 0, 0), 2, tipLength=0.05)


    return combined_image

def combine_tree_reaction_images(images, upper_products_coords, lower_reactants_coords, max_overlap = 0):
    """
    Combine two reaction images vertically and draw an arrow between the products of the first image and the reactants of the second.
    :param images: list of images to combine
    :param upper_products_coords: bounding box coordinates of the products in the first image
    :param lower_reactants_coords: bounding box coordinates of the reactants in the second image
    :param max_overlap: maximum allowed overlap between images in pixels
    :return: combined image
    """
    total_width = max(img.shape[1] for img in images)
    total_height = sum(img.shape[0] for img in images)

    # Adjust the total height considering overlaps
    for _ in range(len(images) - 1):
        overlap = random.randint(0, max_overlap)
        total_height -= overlap

    combined_image = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

    y_offset = 0
    for idx, img in enumerate(images):
        height, width, _ = img.shape
        combined_image[y_offset:y_offset + height, :width] = img

        # If it's the first image, draw an arrow to the second
        if idx == 0:
            arrow_start = (upper_products_coords[1][0], y_offset + upper_products_coords[1][1])
            arrow_end = (lower_reactants_coords[0][0], y_offset + height + lower_reactants_coords[0][1] - max_overlap)
            cv2.arrowedLine(combined_image, arrow_start, arrow_end, (0, 0, 0), 2, tipLength=0.05)

        y_offset += height

        # Subtract a random overlap for the next image
        if len(images) > 1:
            overlap = random.randint(0, max_overlap)
            y_offset -= overlap

    return combined_image



def generate_json_from_results(i,title,result):
    # Initialize the main dictionary structure
    output_dict = {"id": i + 1378,"width": 1333,"height": 450,
                   "file_name": f"{title}.png",
                   "license": 0,"bboxes": [],
                   "reactions": [],
                   "corefs": [],
                   "caption": "",
                   "pdf": {},
                   "diagram_type": "single"}

    # Assigning unique IDs
    unique_id = 0

    # Reactants
    reactants_ids = []
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
    products_ids = []
    for product_bbox in result['products']:
        product_data = {
            "id": unique_id,
            "bbox": product_bbox[0] + product_bbox[1],
            "category_id": 1
        }
        output_dict['bboxes'].append(product_data)
        products_ids.append(unique_id)
        unique_id += 1

    # Conditions (agent, solvent, temperature, time)
    conditions_ids = []
    for condition_type, condition_bbox in result['conditions'].items():
        condition_data = {
            "id": unique_id,
            "bbox": condition_bbox[0] + condition_bbox[1],
            "category_id": 2
        }
        output_dict['bboxes'].append(condition_data)
        conditions_ids.append(unique_id)
        unique_id += 1

    # Populating reactions and corefs
    output_dict['reactions'].append({
        "reactants": reactants_ids,
        "conditions": conditions_ids,
        "products": products_ids
    })

    # Currently, corefs is empty as we don't have the identifiers
    output_dict['corefs'] = []

    return output_dict

def read_json(filename):
    """读取JSON文件内容"""
    with open(filename, 'r') as json_file:
        return json.load(json_file)

def write_json(data, filename):
    """将数据写入JSON文件"""
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def append_to_json(new_data, filename):
    """将新数据追加到现有的JSON文件中"""
    # 读取当前文件内容
    try:
        current_data = read_json(filename)
    except (FileNotFoundError, json.JSONDecodeError):
        current_data = {}

    # 将新数据追加到文件内容
    current_data.append(new_data)

    # 保存更新后的内容
    write_json(current_data, filename)


def save_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)



def generate_json_from_multiple_reactions_with_render5(i, title, results):
    # Initialize the main dictionary structure
    output_dict = {
        "id": i + 50000,
        "width": 1333,
        "height": 450 * len(results),
        "file_name": f"{title}.png",
        "license": 0,
        "bboxes": [],
        "reactions": [],
        "corefs": [],
        "caption": "",
        "pdf": {},
        "diagram_type": "tree",
    }

    unique_id = 0
    previous_products_ids = []

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

        if previous_products_ids:
            output_dict['reactions'].append({
                "reactants": previous_products_ids,
                "conditions": render_ids,
                "products": reactants_ids
            })
        # Populating reactions and corefs
        output_dict['reactions'].append({
            "reactants": reactants_ids,
            "conditions": conditions_ids,
            "products": products_ids
        })

        # If there's a render_agent or render_solvent, create a new reaction connecting previous products to current reactants



        previous_products_ids = products_ids

    # Currently, corefs is empty as we don't have the identifiers
    output_dict['corefs'] = []

    return output_dict


#########################################

reactions_data = []
reaction_count = 0  # 初始化计数器为0

# 使用glob模块遍历每一年的目录
#for year in range(2002):  # 从2001到2023
    # 构建文件路径的模式
 #   pattern = f"pistachio/data/extract/applications/{year}/*_txt_reactions_wbib.json"
for year in range(2021):  # 从2001到2023
    # 构建文件路径的模式
    pattern = f"pistachio/data/extract/applications/2007/*_txt_reactions_wbib.json"
    # 使用glob查找匹配的文件
    for filename in glob.glob(pattern):
        with open(filename, "r") as file:
            for line in file:
                try:
                    reaction = json.loads(line)
                    reactions_data.append(reaction)
                    reaction_count += 1
                    print(f"reactions number: {reaction_count}")
                except:
                    pass
                if reaction_count > 10000:
                    break

print(f"Total reactions: {reaction_count}")

# Modify the extraction process to handle missing "actions" attribute


results = []
for reaction in reactions_data:
    extracted_data = {}
    title = reaction["title"]
    extracted_data["title"] = title
    # Reactants and Products
    reactants_smiles = []
    products_smiles = []
    solvents = set()
    agents = set()
    for component in reaction["components"]:
        if "smiles" in component:  # Check if smiles attribute exists
            if component["role"] == "Reactant":
                reactants_smiles.append(component["smiles"])
            elif component["role"] == "Product":
                products_smiles.append(component["smiles"])
            elif component["role"] == "Solvent":
                sname = component["name"]
                if "water" in sname.lower():
                    sname = "H20"
                solvents.add(sname)
            elif component["role"] == "Agent":
                aname =  component["name"]
                if "water" in aname.lower():
                    aname = "H20"
                agents.add(aname)
                if len(agents) > 3:
                    break
    extracted_data["solvents"] = list(solvents)
    extracted_data["agents"] = list(agents)

    # Removing duplicates
    reactants_smiles = list(set(reactants_smiles))
    products_smiles = list(set(products_smiles))

    extracted_data["reactants_smiles"] = reactants_smiles
    extracted_data["products_smiles"] = products_smiles

    # Solvent, Agent, Temperature and Time

    temperatures = []
    times = []
    max_time = 0  # 最长时间

    if "actions" in reaction:
        for action in reaction["actions"]:
           # if "components" in action:
            #    for component in action["components"]:
             #       if component["role"] == "Solvent":
              #          solvents.append(component["name"])
               #     elif component["role"] == "Agent":
               #         agents.append(component["name"])
            if "parameters" in action:
                for parameter in action["parameters"]:
                    if parameter["type"] == "Temperature":
                        temp_text = parameter["text"].replace("° ", "").strip()
                        if "room temperature" in temp_text.lower():
                            temp_text = "rt"
                        temperatures.append(temp_text)
                    elif parameter["type"] == "Time" and "value" in parameter:
                        # 将时间转换为小时，并找到最大时间
                        current_time = round(parameter["value"] / 3600,1)
                        if current_time > max_time:
                            max_time = current_time

    solvents = list(set(solvents))
    agents = list(set(agents))
    if temperatures:
        temperatures = [temperatures[0]]
    if max_time:
        times = [f"{max_time}h"]

    extracted_data["solvents"] = solvents
    extracted_data["agents"] = agents
    extracted_data["temperatures"] = temperatures
    extracted_data["times"] = times

    results.append(extracted_data)
print(results[:10])
print(len(results))


output_folder = "generated_images_debug_tree1"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_dicts = []
coordinates_list = []
images_to_combine = []
combine_count = 2

for index, reaction in enumerate(results):
    # 提取所需的数据
    reactants = reaction["reactants_smiles"]
    products = reaction["products_smiles"]
    conditions = {
        "agents": reaction["agents"],
        "solvents": reaction["solvents"],
        "temperature": reaction["temperatures"][0] if reaction["temperatures"] else "",
        "time": reaction["times"][0] if reaction["times"] else ""
    }
    title = reaction["title"]
    # 生成图片
    try:
        if len(images_to_combine) == 0:
            image, coordinates = generate_reaction_image(reactants, products, conditions)
            upper_product_coords = coordinates['products']
            print(upper_product_coords)

            images_to_combine.append(image)
            coordinates_list.append(coordinates)
        elif len(images_to_combine) == 1:
            image, coordinates = generate_reaction_image_tree1(reactants, products, conditions)
            down_product_coords = coordinates['reactants']

            images_to_combine.append(image)
            coordinates_list.append(coordinates)
        if len(images_to_combine) == combine_count:
            combined_image = combine_reaction_images(images_to_combine, upper_product_coords, down_product_coords)
            updated_coordinates = update_coordinates_for_vertical_merge(coordinates_list, image_height=450)
            combined_coords = merge_all_coordinates(updated_coordinates)
            combined_filename = os.path.join(output_folder, f"{title}.png")
            cv2.imwrite(combined_filename, combined_image)
            #box_img = draw_boxes_on_reaction_image(combined_image, combined_coords)
            #combined_filename1 = os.path.join(output_folder, f"box_{title}.png")
            #cv2.imwrite(combined_filename1, box_img)
            images_to_combine = []
            coordinates_list = []
            combine_count = 2

            output_dict = generate_json_from_multiple_reactions_with_render5(index, title, updated_coordinates)
        #append_to_json(output_dict, 'output_filename.json')
        #print(output_dict)
            output_dicts.append(output_dict)
        #filename1 = os.path.join(output_folder, f"{title}.png")
        #cv2.imwrite(filename1, image)
    except Exception as e:
        print(f"Error at index {index} with title {title}. Error: {e}")
        index = index - 1
        pass
    print(index)
    if index > 4000:
        break
    #box_img = draw_boxes_on_reaction_image(image, coordinates)
    #filename = os.path.join(output_folder, f"reaction_{index + 1}.png")
    #print(coordinates)
    #cv2.imwrite(filename, box_img)
print(output_dicts)
save_to_json(output_dicts, 'outputdebug_tree.json')
