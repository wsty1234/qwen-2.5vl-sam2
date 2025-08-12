
import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import numpy as np
import os
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
import os
from qwen_vl_utils import process_vision_info  # 官方工具函数
import re
from collections import defaultdict


def generate_points_with_qwen(model, processor, image, user_prompot):
    """
    使用 Qwen2.5-VL 模型生成指定对象的边界框（点提示）。
    通过更明确的提示词，旨在获取规范的 JSON 输出。

    Args:
        model: 已加载的Qwen2.5-VL模型
        processor: 已加载的处理器
        image (PIL.Image): 待检测的RGB图像
        user_prompot (str or list): 需要检测的对象类别列表（如["car", "person"]）或单个字符串。

    Returns:
        list: Qwen-VL 的原始输出列表，通常包含一个字符串（JSON格式）。
    """

    # Ensure user_prompot is a string for the prompt
    if isinstance(user_prompot, list):
        # Join multiple prompts, e.g., ["car", "person"] -> "car或person"
        user_prompot_str = "、".join(user_prompot)
    else:
        user_prompot_str = user_prompot

    # # Create messages
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": image},
    #             {
    #                 "type": "text",
    #                 "text": f"请从图片中检测所有的{user_prompot}。对于每一个检测到的{user_prompot}，根据目标大小，请生成 **1到5个代表其完整轮廓的点**。这些点应该分布在目标的各个部分，例如车头、车身和车尾等。将结果以 JSON 格式输出，其中每个对象的 'points' 键的值必须是一个包含点坐标的**列表的列表**，例如：[{{\"points\": [[x1, y1], [x2, y2]], \"label\": \"label_name\"}}, {{ \"points\": [[x3, y3], [x4, y4], [x5, y5]], \"label\": \"label_name\"}}]"
    #             },
    #         ],
    #     }
    # ]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": f"请从图片中检测所有的{user_prompot_str}。\n"
                              "对于每一个检测到的目标，请生成 **3到5个代表其完整轮廓的关键点**。\n"
                              "这些点应该分布在目标的各个部分，例如汽车的四个角和中心。\n"
                              "请严格按照以下 JSON 数组格式输出结果。**不要有任何额外文字，只返回 JSON。**\n"
                              "**每个 JSON 对象必须精确地包含 'points' 和 'label' 两个键。**\n"
                              "   - 'points' 的值必须是一个包含 `[x, y]` 坐标对的列表，例如 `[[x1, y1], [x2, y2], [x3, y3]]`。\n"
                              "     **每个点必须是一个独立的 `[x, y]` 子列表，即使只有一个点。**\n"
                              "   - 'label' 的值是对象的类别字符串，例如 `\"car\"` 或 `\"road\"`，**不要包含额外的方括号或引号。**\n"
                              "以下是严格的 JSON 示例（请严格遵循此格式，包括所有逗号和括号）：\n"
                              "```json\n"
                              "[\n"
                              "  {\"points\": [[100, 200], [150, 250], [100, 250], [150, 200]], \"label\": \"car\"},\n"
                              "  {\"points\": [[300, 400]], \"label\": \"road\"}  \n" # Example with a single point (still as [[x,y]])
                              "]\n"
                              "```\n"
                              "**再次强调：在 'points' 列表结束的 `]` 之后，必须紧跟一个逗号 `,`，然后才是 'label' 键。**"
                },
            ],
        }
    ]    
    # Execute inference
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Assuming process_vision_info is defined elsewhere and works correctly
    # You'll need to make sure this function is correctly imported or defined.
    # For now, I'll put a placeholder if it's not provided.
    try:
        image_inputs, video_inputs = process_vision_info(messages)
    except NameError:
        print("Warning: 'process_vision_info' not found. Assuming simple image input structure.")
        # Fallback if process_vision_info is not provided
        # This part depends heavily on how your 'messages' are structured for processor
        # For typical single image input, it might look like this:
        image_inputs = [image] # Pass the PIL image directly if processor handles it
        video_inputs = None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate output
    print("------------------Qwen2.5b_vl正在生成输出------------------")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=2048
        
        # Potentially add temperature=0.1 or top_p=0.9 to encourage more deterministic output
        # num_beams=1 # Or increase for more diverse, but potentially slower, generation
    )

    # Decode generated IDs
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def pro_vl_results(rl_output):
    """
    处理 Qwen-VL 的原始输出，提取 JSON 内容，并将其格式化为 SAM2 所需的点提示。
    增强了对 'points' 嵌套结构、'label' 字符串化列表的鲁棒性，并过滤掉负数坐标点。

    Args:
        rl_output (list): Qwen-VL 模型的原始输出列表，通常包含一个字符串。

    Returns:
        dict: 包含 'points' 和 'label' 键的字典。
              'points' 是一个包含所有对象点的列表，每个子列表是 [[x, y], ...]。
              'label' 是一个包含每个点集对应标签长度的列表，例如 [[N1], [N2], ...]。
    """
    processed_points_for_sam2 = {'points': [], 'label': []}
    
    # 1. 提取 JSON 内容
    json_content_str = None
    if rl_output and isinstance(rl_output[0], str):
        json_pattern = r'```json\n(.*?)\n```'
        json_match = re.search(json_pattern, rl_output[0], re.DOTALL)
        
        if json_match:
            extracted_json_content = json_match.group(1)
            # 尝试修复 ']]"label"' 变成 ']], "label"' 的情况
            repaired_json_content = re.sub(r'\]\]\s*"label"', r']], "label"', extracted_json_content)
            repaired_json_content = re.sub(r'\]\s*"label"', r'], "label"', repaired_json_content)
            
            try:
                raw_point_results = json.loads(repaired_json_content)
            except json.JSONDecodeError as e:
                print(f"警告: 提取的 JSON 内容解析失败: {e}. 内容: {repaired_json_content[:200]}...")
                raw_point_results = []
        else:
            # 如果没有 ```json``` 标记，尝试直接解析，尽管Qwen-VL通常会带
            try:
                raw_point_results = json.loads(rl_output[0])
            except json.JSONDecodeError:
                print("未在 ```json 标记之间或作为纯 JSON 字符串找到 JSON 内容。")
                raw_point_results = []
    else:
        print(f"警告: Qwen-VL 输出格式异常: {rl_output}. 期望一个包含字符串的列表。")
        raw_point_results = []

    # 确保 raw_point_results 是一个列表，以便统一处理
    if not isinstance(raw_point_results, list):
        raw_point_results = [raw_point_results]

    # 2. 遍历解析后的结果，提取并格式化点信息
    for obj_data in raw_point_results:
        if 'points' in obj_data and isinstance(obj_data['points'], list):
            current_obj_raw_points = obj_data['points']
            
            # --- 扁平化 points 列表 START ---
            flattened_points = []
            def flatten(items):
                """递归扁平化列表，直到遇到非列表元素或[x, y]对"""
                for item in items:
                    if isinstance(item, list) and len(item) == 2 and all(isinstance(coord_val, (int, float)) for coord_val in item):
                        # --- 新增的负数坐标过滤逻辑 START ---
                        if item[0] >= 0 and item[1] >= 0: # 确保 x 和 y 坐标都非负
                            # 找到了一个 [x, y] 对，且坐标有效，直接添加
                            flattened_points.append(item)
                        else:
                            print(f"警告: 发现负数或无效坐标，已过滤: {item}")
                        # --- 新增的负数坐标过滤逻辑 END ---
                    elif isinstance(item, list):
                        # 如果是列表，继续递归扁平化
                        flatten(item)
                    # 否则，跳过非列表或非 [x, y] 的项，或可以添加警告
            
            flatten(current_obj_raw_points)
            # --- 扁平化 points 列表 END ---

            # --- 解析 label 字段 START ---
            extracted_label = "unknown" # 默认值
            raw_label = obj_data.get('label', '')
            if isinstance(raw_label, str):
                # 尝试解析 "['road']" 或 "road"
                try:
                    # 将单引号替换为双引号以便JSON解析
                    parsed_label_list = json.loads(raw_label.replace("'", '"')) 
                    if isinstance(parsed_label_list, list) and len(parsed_label_list) > 0:
                        extracted_label = str(parsed_label_list[0]) # 取列表的第一个元素
                    else:
                        extracted_label = raw_label # 如果解析失败或不是列表，就用原始字符串
                except json.JSONDecodeError:
                    extracted_label = raw_label # 如果不是有效的JSON字符串，就用原始字符串
            elif raw_label is not None:
                # 如果 label 直接是字符串（如 "road"），或其他非字符串类型
                extracted_label = str(raw_label)
            # --- 解析 label 字段 END ---

            if flattened_points: # 如果当前对象有有效的点
                processed_points_for_sam2['points'].append(flattened_points)
                # SAM2通常期望每个点一个标签，且都是前景（1）
                # 我们这里存储的是该点集包含的点数量，在merge_annotations中会转化为1
                processed_points_for_sam2['label'].append([len(flattened_points)])
            else:
                print(f"警告: 对象数据中 'points' 字段为空或无效，或者未能成功扁平化: {obj_data}")
        else:
            print(f"警告: 对象数据中缺少 'points' 键或其格式不正确: {obj_data}")

    return processed_points_for_sam2


import json
import re

def pro_vl_results_32B(rl_output):
    """
    处理 Qwen-VL 的原始输出，提取 JSON 内容，并将其格式化为 SAM2 所需的点提示。
    增强了对 'points' 嵌套结构、'label' 字符串化列表的鲁棒性，过滤负数坐标，
    并更灵活地修复 JSON 格式中缺失的逗号。

    Args:
        rl_output (list): Qwen-VL 模型的原始输出列表，通常包含一个字符串。

    Returns:
        dict: 包含 'points' 和 'label' 键的字典。
              'points' 是一个包含所有对象点的列表，每个子列表是 [[x, y], ...]。
              'label' 是一个包含每个点集对应标签长度的列表，例如 [[N1], [N2], ...]。
    """
    processed_points_for_sam2 = {'points': [], 'label': []}
    # print(f"😊😊😊😊{rl_output[0]}")
    # 1. 提取 JSON 内容
    json_content_str = None
    if rl_output and isinstance(rl_output[0], str):
        json_pattern = r'```json\n(.*?)\n```'
        json_match = re.search(json_pattern, rl_output[0], re.DOTALL)
        
        if json_match:
            extracted_json_content = json_match.group(1)
            
            # --- 更通用的 JSON 修复逻辑 START ---
            # 修复 '[...]"label"' 变成 '[...], "label"' 的情况
            # 匹配任何一个或多个 ']' 后面直接跟着一个 '"' (如果前面不是 ',')
            # 这是一个更通用的模式，以捕捉 ']]"label"' 和 ']"label"' 甚至可能是 ']]]"label"'
            repaired_json_content = re.sub(r'\]\s*(?<!,)"', r'], "', extracted_json_content)
            # 再次修复，以防有嵌套的 ]] 且之前没有逗号，例如 [[x,y]],"label" 变成 [[x,y]],,"label"
            # 确保只在必要时添加逗号，避免重复
            repaired_json_content = re.sub(r'\]\s*,\s*\]\s*,\s*"', r']], "', repaired_json_content) # For cases like ']],,"label"' from previous repair
            
            repaired_json_content = re.sub(r'"label":\s*"\[\'(.*?)\'\]"', r'"label": "\1"', repaired_json_content)

            # --- 更通用的 JSON 修复逻辑 END ---
            
            try:
                raw_point_results = json.loads(repaired_json_content)
            except json.JSONDecodeError as e:
                print(f"警告: 提取的 JSON 内容解析失败: {e}. 内容: {repaired_json_content[:200]}...")
                raw_point_results = []
        else:
            # If no ```json``` marker, try direct parsing, although Qwen-VL usually provides one
            try:
                raw_point_results = json.loads(rl_output[0])
            except json.JSONDecodeError:
                print("未在 ```json 标记之间或作为纯 JSON 字符串找到 JSON 内容。")
                raw_point_results = []
    else:
        print(f"警告: Qwen-VL 输出格式异常: {rl_output}. Expected a list containing a string.")
        raw_point_results = []

    # Ensure raw_point_results is a list for uniform processing
    if not isinstance(raw_point_results, list):
        raw_point_results = [raw_point_results]

    # 2. Iterate through parsed results, extract and format point information
    for obj_data in raw_point_results:
        if 'points' in obj_data and isinstance(obj_data['points'], list):
            current_obj_raw_points = obj_data['points']
            
            # --- Flatten points list START ---
            flattened_points = []
            def flatten(items):
                """Recursively flattens lists until non-list elements or [x, y] pairs are found."""
                for item in items:
                    # Check for [x, y] pair and non-negative coordinates
                    if isinstance(item, list) and len(item) == 2 and all(isinstance(coord_val, (int, float)) for coord_val in item):
                        if item[0] >= 0 and item[1] >= 0:
                            flattened_points.append(item)
                        else:
                            print(f"警告: 发现负数或无效坐标，已过滤: {item}")
                    elif isinstance(item, list):
                        flatten(item)
                    # Else, skip non-list or non-[x, y] items
            
            flatten(current_obj_raw_points)
            # --- Flatten points list END ---

            # --- Parse label field START ---
            extracted_label = "unknown" # Default value
            raw_label = obj_data.get('label', '')
            # If label is a string, try to clean it
            if isinstance(raw_label, str):
                # We tried to repair `["'road'"]` to `"road"` earlier with regex.
                # If that failed, this attempts to parse it as JSON
                try:
                    parsed_label_list = json.loads(raw_label.replace("'", '"')) 
                    if isinstance(parsed_label_list, list) and len(parsed_label_list) > 0:
                        extracted_label = str(parsed_label_list[0])
                    else:
                        extracted_label = raw_label 
                except json.JSONDecodeError:
                    extracted_label = raw_label # If not a valid JSON string, use original
            elif raw_label is not None:
                extracted_label = str(raw_label)
            # --- Parse label field END ---

            if flattened_points: # If current object has valid points
                processed_points_for_sam2['points'].append(flattened_points)
                # SAM2 usually expects one label per point, all foreground (1)
                # We store the number of points in this set; it will be converted to 1s in merge_annotations
                processed_points_for_sam2['label'].append([len(flattened_points)]) # Keep the length for now
            else:
                print(f"警告: 对象数据中 'points' 字段为空或无效，或者未能成功扁平化: {obj_data}")
        else:
            print(f"警告: 对象数据中缺少 'points' 键或其格式不正确: {obj_data}")

    return processed_points_for_sam2
def timeit(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result
    return wrapper

def merge_annotations(annotations):
    """
    合并具有相同 frame_idx 的注解。
    将每个 frame_idx 下的所有点的 'points' 和 'labels' 聚合起来。
    生成的 'labels' 格式为 [1, 1, ..., 1]，数量与合并后的点数匹配。

    Args:
        annotations (list): 原始的注解列表，每个元素包含 'obj_id', 'frame_idx', 'points', 'labels', 'box'。

    Returns:
        list: 合并后的注解列表。每个元素代表一个 frame_idx 下的所有对象。
              其 'points' 包含所有对象的扁平化点列表，'labels' 包含与这些点一一对应的 '1' 标签。
              'obj_id' 表示该帧中原始对象的总数（用于元数据，SAM2可能不直接使用此ID）。
    """
    merged_by_frame = defaultdict(
        lambda: {'points': [], 'labels': [], 'box': [], 'original_obj_count': 0}
    )

    for annot in annotations:
        frame_idx = annot['frame_idx']
        
        # 将当前注解的点添加到该帧的总点列表中 (扁平化)
        # annot['points'] 已经是像 [[x1, y1], [x2, y2]] 这样的列表
        merged_by_frame[frame_idx]['points'].extend(annot['points'])
        
        # 根据当前注解的点数量，添加相应数量的 '1' 到该帧的总标签列表中
        # annot['points'] 的长度就是该原始对象所包含的点数
        num_points_in_current_object = len(annot['points'])
        merged_by_frame[frame_idx]['labels'].extend([1] * num_points_in_current_object)
        
        # 增加原始对象计数
        merged_by_frame[frame_idx]['original_obj_count'] += 1
        
        # 框信息（如果存在，这里简单地取第一个，或者根据需要进行合并）
        if annot['box']:
            # 这里的逻辑需要根据你实际的 box 处理需求来定
            pass # 保持不动，或根据你的合并策略处理

    final_merged_annotations = []
    for frame_idx in sorted(merged_by_frame.keys()):
        merged_data = merged_by_frame[frame_idx]
        final_merged_annotations.append({
            'obj_id': merged_data['original_obj_count'], # 此处的 obj_id 作为元数据，表示原始对象的总数
            'frame_idx': frame_idx,
            'points': merged_data['points'], # 包含所有原始对象的扁平化点列表
            'labels': merged_data['labels'], # 包含与所有点一一对应的 '1' 标签，如 [1, 1, 1, 1, 1, 1, 1, 1, 1]
            'box': merged_data['box'] # 保持为原始收集的box（如果存在）
        })
    return final_merged_annotations

def init_models():
    """初始化所有模型"""
    # SAM2
    SAM2_CHECKPOINT = "/home/iking/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_predictor = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT)
    
    return sam_predictor

CONFIG = {
    "prompts": ["sky"], # 提示词
    "keyframe_interval": 20,   # 提示帧的频率
    "sam_config": "checkpoints/sam2.1_hiera_large.pt",
    "sam_checkpoint": "configs/sam2.1/sam2.1_hiera_l.yaml"
}

def save_colored_mask_png(mask_bool, out_path, color=(255, 0, 0)):
    """
    mask_bool: (H, W) 的 bool/0-1 掩码
    color: RGB 固定颜色
    out_path: 保存路径（.png）
    """
    h, w = mask_bool.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = np.array(color, dtype=np.uint8)
    rgba[..., 3] = (mask_bool.astype(np.uint8) * 255)  # 前景不透明，背景透明
    Image.fromarray(rgba, mode='RGBA').save(out_path)

# 4B) 保存为单通道掩码（灰度/索引）
def save_binary_mask_png(mask_bool, out_path, fg_value=255):
    """
    单通道 8-bit，前景 fg_value，背景 0
    """
    img = (mask_bool.astype(np.uint8) * fg_value)
    Image.fromarray(img, mode='L').save(out_path)

def quad_to_xyxy(quad):
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        return [min(xs), min(ys), max(xs), max(ys)]

def main():
    # 0. 配置
    img_file = "/home/iking/qwen_vl_sam/demo0.png"
    output_dir = "./outputs"
    # 1. 初始化SAM2模型
    sam_predictor= init_models()
    sam2_predictor = SAM2ImagePredictor(sam_predictor)
    
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    qwen_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct"
        )

    annotations = []
    
    # 2. 读取图像
    frame = cv2.imread(img_file)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil_from_array = Image.fromarray(frame_rgb)
    objects_to_detect = CONFIG["prompts"]
    print(f"objects_to_detect:{objects_to_detect}")
    # 使用Qwen模型生成点提示
    vl_results = generate_points_with_qwen(
        model=qwen_model,
        processor=qwen_processor,
        image=frame_pil_from_array,
        user_prompot=objects_to_detect
    )
    print(f"vl_results:{vl_results}")
    # processed_points = pro_vl_results(vl_results)
    processed_points = pro_vl_results_32B(vl_results)
    print(f"处理后的LLM结果:{processed_points}")

    boxes = [quad_to_xyxy(quad) for quad in processed_points['points']]
    boxes_np = np.array(boxes, dtype=np.float32)  # shape: (N, 4), format [x_min, y_min, x_max, y_max    

    raw_labels = [l[0] for l in processed_points['label']]  # 展平每个的首元素
    names = [str(x) for x in raw_labels]  # 这里直接转字符串

    sam2_predictor.set_image(frame_pil_from_array)
    if torch.cuda.is_available(): # 或者 torch.cuda.is_available() for GPU
        torch.cuda.empty_cache() # 或者 torch.cuda.empty_cache()
    print("--------------------- Qwen-VL 模型显存已释放---------------------")
    # --- 显存释放结束 ---

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes_np,
        multimask_output=False,
    )
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    save_dir = "/home/iking/qwen_vl_sam"
    fixed_color = (0, 255, 0)  # 固定颜色：绿色
    for i, mask in enumerate(masks):
        # 确保是 bool
        mask_bool = (mask > 0.5) if mask.dtype != np.bool_ else mask
        name = names[i] if i < len(names) else f"mask_{i}"

        # A) 彩色带透明
        out_path_rgba = os.path.join(save_dir, f"{name}.png")
        save_colored_mask_png(mask_bool, out_path_rgba, color=fixed_color)

if __name__ == "__main__":
    main()