from flask import Flask, request, session, render_template_string, redirect, url_for, jsonify, make_response
from werkzeug.utils import secure_filename
import os
import math
import csv
import glob
import warnings
import certifi
import requests
import rasterio
import pandas as pd
import calendar
import re
import base64
import io
import logging
from datetime import datetime, timedelta
from math import exp
from ultralytics import YOLO
from pyproj import Transformer
from threading import Thread, Event
import uuid
import sys
import random
from gunicorn.app.base import BaseApplication
import json
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

DEFAULT_TIFF_PATH = "/Users/michael_z/Downloads/PlantTools/snum_sand_silt_clay_drytime_1440_480_5.tif"
PLANT_DATA_CSV = "/Users/michael_z/Downloads/PlantTools/Plant Database/numbers_soil_clean_processed.csv"
OPENWEATHER_API_KEY = ""
YOLO_MODEL_PATH = "/Users/michael_z/Downloads/PlantTools/plant-growth-stage-classifier-yolov8x-best.pt"
PLANTNET_RESULT_FILE = "/Users/michael_z/Downloads/PlantTools/plantnet_result.csv"
YOLO_TRANSLATED_TEXT_FILE = "results_translated.txt"
YOLO_OUTPUT_TEXT_FILE = "output.txt"
YOLO_PROB_TEXT_FILE = "prob.txt"
YOLO_RESULT_TEXT_FILE = "results.txt"
SOIL_CSV_FILE = "/Users/michael_z/Downloads/PlantTools/soil.csv"
TEMPERATURE_CSV_FILE = "/Users/michael_z/Downloads/PlantTools/TEMP.csv"
SOIL_WATER_CONTENT_CSV_FILE = "/Users/michael_z/Downloads/PlantTools/SWC.csv"
EVAPOTRANSPIRATION_CSV_FILE = "/Users/michael_z/Downloads/PlantTools/ET0.csv"
WATERING_CSV_FILE = "/Users/michael_z/Downloads/PlantTools/watering.csv"
AREA_CSV_FILE = "/Users/michael_z/Downloads/PlantTools/area.csv"
SOIL_PH_TIFF_PATH = "/Users/michael_z/Downloads/PlantTools/phh2o_0-5cm_mean_1000.tif"
PLANT_DATABASE_PATH = "/Users/michael_z/Downloads/PlantTools/Plant Database/numbers_soil_clean_updated_processed.csv"
VISUAL_CROSSING_API_KEY = ""
WATERING_PLAN_CSV_FILE = "/Users/michael_z/Downloads/PlantTools/watering_plan.csv"

# Global variables for area data, disease info and plant database.
area_data = {"pots": [], "yard": None}
global_disease_info = None
global_yolo_model = None
plant_database_dataframe = None

# ---------------- Soil Multiplier Mapping ----------------

soil_multiplier_mapping = {
    "Sand": 1.8,
    "Loamy sand": 1.6,
    "Sandy loam": 1.4,
    "Loam": 1.0,
    "Silt loam": 1.2,
    "Silt": 1.1,
    "Sandy clay loam": 0.85,
    "Clay loam": 0.9,
    "Silty clay loam": 0.8,
    "Sandy clay": 0.75,
    "Silty clay": 0.7,
    "Clay": 0.65,
}

# ---------------- YOLO Translation Map (German to English) ----------------

translation_mapping = {
    "Aufgang": "shoot emergence",
    "Aussaat": "sowing",
    "Austrieb": "sprouting",
    "Blattentfaltung": "leaf unfolding",
    "Blattfall": "leaf fall",
    "Blattverfärbung": "leaf discoloration",
    "Blühbeginn": "flowering onset",
    "Blüte": "flowering",
    "Ernte": "harvest",
    "Fruchtreife": "fruit maturity",
    "Keimung": "germination",
    "Knospen": "bud formation",
    "Vollblüte": "full bloom"
}

# ---------------- Disease Detection using ResNet-34 ----------------

disease_detection_model = None
disease_detection_labels = None


def load_disease_detection_model():
    labels = [
        "Strawberry___healthy",
        "Grape___Black_rot",
        "Potato___Early_blight",
        "Blueberry___healthy",
        "Corn_(maize)___healthy",
        "Tomato___Target_Spot",
        "Peach___healthy",
        "Potato___Late_blight",
        "Tomato___Late_blight",
        "Tomato___Tomato_mosaic_virus",
        "Pepper,_bell___healthy",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Tomato___Leaf_Mold",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Cherry_(including_sour)___Powdery_mildew",
        "Apple___Cedar_apple_rust",
        "Tomato___Bacterial_spot",
        "Grape___healthy",
        "Tomato___Early_blight",
        "Corn_(maize)___Common_rust_",
        "Grape___Esca_(Black_Measles)",
        "Raspberry___healthy",
        "Tomato___healthy",
        "Cherry_(including_sour)___healthy",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Apple___Apple_scab",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Peach___Bacterial_spot",
        "Pepper,_bell___Bacterial_spot",
        "Tomato___Septoria_leaf_spot",
        "Squash___Powdery_mildew",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Apple___Black_rot",
        "Apple___healthy",
        "Strawberry___Leaf_scorch",
        "Potato___healthy",
        "Soybean___healthy"
    ]
    model = models.resnet34(pretrained=False)
    features = model.fc.in_features
    model.fc = nn.Linear(features, len(labels))
    checkpoint_path = "/Users/michael_z/Downloads/PlantTools/plantDisease-resnet34.pth"
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[len("network."):] if key.startswith("network.") else key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    model.eval()
    return model, labels


def predict_disease_from_image(image_path, model, labels):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        # sys.exit(1) # Do not exit the Flask app
        raise ValueError(f"Could not load image: {e}")
    transform_pipeline = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transformed_img = transform_pipeline(img).unsqueeze(0)
    with torch.no_grad():
        output = model(transformed_img)
        _, predicted_idx = torch.max(output, 1)
    return labels[predicted_idx.item()]


def get_disease_information(image_path, lang='en'):
    global disease_detection_model, disease_detection_labels
    if disease_detection_model is None:
        disease_detection_model, disease_detection_labels = load_disease_detection_model()
    try:
        predicted = predict_disease_from_image(image_path, disease_detection_model, disease_detection_labels)
    except Exception as e:
        logging.error(f"Error predicting disease: {e}")
        return {"disease": get_text("unknown_disease", lang), "remedy": get_text("error_disease_prediction", lang)}

    if "healthy" in predicted.lower():
        remedy_key = "remedy_healthy"
    else:
        # Construct the key for the remedy mapping, e.g., "remedy_Potato___Early_blight"
        remedy_key = f"remedy_{predicted}"

    remedy = get_text(remedy_key, lang)  # This will fetch the translated remedy

    # Also translate the disease name itself if it's complex or has underscores
    display_disease_name = predicted.replace("___", " - ").replace("_", " ")
    return {"disease": display_disease_name, "remedy": remedy}


# ---------------- Global Language Dictionary ----------------
# This dictionary holds all translatable strings for the UI.
# Each key represents a unique string identifier, and its value is another dictionary
# containing the English ('en') and Chinese ('zh') translations.
global_lang_strings = {
    'string_key': {'en': 'English text', 'zh': 'Chinese text'},
    'app_title': {'en': "Agriculture Tool", 'zh': "农业工具"},
    'welcome_message': {'en': "Welcome to the Agriculture Tool", 'zh': "欢迎使用农业工具"},
    'select_function': {'en': "Please select a functionality:", 'zh': "请选择一个功能:"},
    'btn_watering_recommendations': {'en': "Watering Recommendations", 'zh': "浇水建议"},
    'btn_crop_recommendations': {'en': "Crop Recommendations", 'zh': "作物推荐"},
    'btn_plant_analysis': {'en': "Plant Analysis", 'zh': "植物分析"},
    'btn_back_to_main_menu': {'en': "Back to Main Menu", 'zh': "返回主菜单"},
    'btn_english': {'en': "English", 'zh': "英语"},
    'btn_chinese': {'en': "中文", 'zh': "中文"},

    # Watering Form
    'watering_title': {'en': "Watering Recommendations", 'zh': "浇水建议"},
    'watering_step1_title': {'en': "Step 1: Enter Location and Area Details", 'zh': "第一步: 输入地理位置和区域详情"},
    'label_latitude': {'en': "Latitude:", 'zh': "纬度:"},
    'label_longitude': {'en': "Longitude:", 'zh': "经度:"},
    'latitude_pattern_title': {'en': "Enter valid latitude (-90 to 90)", 'zh': "输入有效纬度 (-90 到 90)"},
    'longitude_pattern_title': {'en': "Enter valid longitude (-180 to 180)", 'zh': "输入有效经度 (-180 到 180)"},
    'pots_data_heading': {'en': "Pots Data", 'zh': "盆栽数据"},
    'pots_data_desc1': {'en': "Enter one pot per line. Use format:", 'zh': "每行输入一个盆栽。格式如下:"},
    'pots_data_desc2_c': {'en': "<code>c, radius_in_cm, plant_name</code> (for Circular pots)",
                          'zh': "<code>c, 半径_厘米, 植物名称</code> (圆形盆栽)"},
    'pots_data_desc2_r': {'en': "<code>r, length_in_cm, width_in_cm, plant_name</code> (for Rectangular pots)",
                          'zh': "<code>r, 长度_厘米, 宽度_厘米, 植物名称</code> (矩形盆栽)"},
    'pots_data_example_title': {'en': "<em>Example:</em>", 'zh': "<em>示例:</em>"},
    'pots_data_example_c': {'en': "<code>c, 15, Tomato</code>", 'zh': "<code>c, 15, 西红柿</code>"},
    'pots_data_example_r': {'en': "<code>r, 30, 20, Basil</code>", 'zh': "<code>r, 30, 20, 罗勒</code>"},
    'yard_data_heading': {'en': "Yard Data", 'zh': "院子数据"},
    'yard_data_desc1': {
        'en': "Define the total yard area. Enter dimensions (length, width) in meters, one section per line if irregular.",
        'zh': "定义院子总面积。输入尺寸（长、宽）以米为单位，如果形状不规则则每行输入一个区域。"},
    'yard_data_example_title': {'en': "<em>Example (for a 5m x 10m yard):</em>",
                                'zh': "<em>示例 (5米 x 10米院子):</em>"},
    'yard_data_example': {'en': "<code>5, 10</code>", 'zh': "<code>5, 10</code>"},
    'label_yard_plants': {'en': "Plants in Yard:", 'zh': "院子里的植物:"},
    'yard_plants_placeholder': {'en': "Comma-separated, e.g., Roses, Lavender, Grass",
                                'zh': "逗号分隔, 例如：玫瑰, 薰衣草, 草坪"},
    'yard_plants_small_text': {'en': "(Leave blank if yard area is not planted or for generic calculation)",
                               'zh': "(如果院子未种植植物或进行通用计算，请留空)"},
    'yard_plant_areas_heading': {'en': "Yard Plant Areas (Optional)", 'zh': "院子植物面积 (可选)"},
    'yard_plant_areas_desc': {'en': "Enter one plant per line in the format: <code>plant_name, area_in_m²</code>",
                              'zh': "每行输入一个植物，格式为：<code>植物名称, 面积_平方米</code>"},
    'yard_plant_areas_placeholder': {'en': "Roses, 2.5&#10;Lavender, 1.2", 'zh': "玫瑰, 2.5&#10;薰衣草, 1.2"},
    'btn_calculate_daily_needs': {'en': "Calculate Daily Needs", 'zh': "计算每日需求"},

    # Watering Select Days
    'watering_select_days_title': {'en': "Watering Recommendations - Select Days", 'zh': "浇水建议 - 选择日期"},
    'watering_step2_title': {'en': "Step 2: Select Watering Days", 'zh': "第二步: 选择浇水日期"},
    'info_daily_needs_calculated': {
        'en': "Below are the <strong>calculated daily watering needs (in Liters)</strong> based on the forecast and your inputs (Soil Type: <strong>{}</strong>).",
        'zh': "以下是根据天气预报和您的输入（土壤类型：<strong>{}</strong>）计算出的<strong>每日浇水需求（升）</strong>。"},
    'info_yard_plant_individual': {'en': "For the yard, each plant's watering recommendation is provided individually.",
                                   'zh': "对于院子，每种植物的浇水建议单独提供。"},
    'info_check_boxes_adjust': {'en': "Check the boxes for the days you plan to water. Adjust as needed.",
                                'zh': "勾选您计划浇水的日期。根据需要调整。"},
    'table_header_area_plant': {'en': "Area / Plant", 'zh': "区域 / 植物"},
    'table_header_frequency_guideline': {'en': "Frequency Guideline", 'zh': "频率指南"},
    'table_header_potted_plants': {'en': "Potted Plants", 'zh': "盆栽植物"},
    'table_header_yard_plants': {'en': "Yard Plants", 'zh': "院子植物"},
    'btn_generate_final_plan': {'en': "Generate Final Plan", 'zh': "生成最终计划"},
    'btn_back_to_inputs': {'en': "Back to Inputs", 'zh': "返回输入"},

    # Watering Final Plan
    'final_plan_title': {'en': "Your Final Watering Plan", 'zh': "您的最终浇水计划"},
    'plan_selected_days_only': {'en': "This plan includes only the days you selected.",
                                'zh': "此计划仅包含您选择的日期。"},
    'csv_message_success': {'en': "Watering plan saved to watering.csv.", 'zh': "浇水计划已保存到 watering.csv。"},
    'csv_message_error': {'en': "Error saving watering plan to CSV: {}", 'zh': "保存浇水计划到 CSV 失败: {}"},
    'no_watering_days_selected': {'en': "No watering days were selected.", 'zh': "没有选择任何浇水日期。"},
    'table_header_area_type': {'en': "Area Type", 'zh': "区域类型"},
    'table_header_identifier': {'en': "Identifier", 'zh': "标识符"},
    'table_header_plant_name': {'en': "Plant Name", 'zh': "植物名称"},
    'table_header_watering_date': {'en': "Watering Date", 'zh': "浇水日期"},
    'table_header_water_amount': {'en': "Water Amount (L)", 'zh': "水量 (升)"},
    'btn_back_to_day_selection': {'en': "Back to Day Selection", 'zh': "返回日期选择"},
    'Pot': {'en': "Pot", 'zh': "盆栽"},  # Specific translation for 'Pot'
    'Yard': {'en': "Yard", 'zh': "院子"},  # Specific translation for 'Yard'

    # Crop Search
    'crop_search_title': {'en': "Search and Select Plants", 'zh': "搜索并选择植物"},
    'label_search_plants': {'en': "Search for plants:", 'zh': "搜索植物:"},
    'btn_search': {'en': "Search", 'zh': "搜索"},
    'heading_select_plants': {'en': "Select Plants", 'zh': "选择植物"},
    'btn_submit_selection': {'en': "Submit Selection", 'zh': "提交选择"},
    'no_crop_data_found': {'en': "No crop data found", 'zh': "未找到作物数据"},
    'no_matching_crop_data_found': {'en': "No matching crop data found.", 'zh': "未找到匹配的作物数据。"},
    'please_select_at_least_one_plant': {'en': "Please select at least one plant", 'zh': "请至少选择一种植物"},

    # Data Choice
    'data_choice_title': {'en': "Choose Data Input", 'zh': "选择数据输入方式"},
    'heading_choose_data_input_method': {'en': "Choose Data Input Method", 'zh': "选择数据输入方法"},
    'btn_use_location_data': {'en': "Use Location Data", 'zh': "使用位置数据"},
    'btn_enter_manual_data': {'en': "Enter Manual Data", 'zh': "手动输入数据"},

    # Location Form
    'location_form_title': {'en': "Enter Location Data", 'zh': "输入位置数据"},
    'label_month': {'en': "Month (1-12):", 'zh': "月份 (1-12):"},
    'label_year': {'en': "Year:", 'zh': "年份:"},
    'btn_submit': {'en': "Submit", 'zh': "提交"},
    'invalid_input': {'en': "Invalid input", 'zh': "输入无效"},

    # Sensor Form
    'sensor_form_title': {'en': "Enter Sensor Data", 'zh': "输入传感器数据"},
    'label_inst_temp': {'en': "Instantaneous Temperature (C):", 'zh': "瞬时温度 (摄氏度):"},
    'label_ambient_humidity': {'en': "Ambient Humidity (%):", 'zh': "环境湿度 (%):"},
    'label_atm_pressure': {'en': "Atmospheric Pressure (hPa):", 'zh': "大气压力 (百帕):"},
    'label_monthly_avg_temp': {'en': "Monthly Avg Temperature (C):", 'zh': "月平均温度 (摄氏度):"},
    'label_total_precipitation': {'en': "Total Precipitation (mm):", 'zh': "总降水量 (毫米):"},
    'label_soil_ph': {'en': "Soil pH:", 'zh': "土壤酸碱度:"},

    # Crop Result
    'recommendation_title': {'en': "Crop Recommendation", 'zh': "作物推荐"},
    'heading_recommendation': {'en': "Recommendation", 'zh': "推荐"},
    'area_not_fit_for_gardening': {'en': "This area may not be fit for gardening now; consider planting indoors.",
                                   'zh': "该区域目前可能不适合园艺；请考虑室内种植。"},
    'recommended_crop': {'en': "Recommended Crop: {} (Score: {:.4f})", 'zh': "推荐作物: {} (分数: {:.4f})"},

    # Plant Analysis
    'plant_analysis_tool_title': {'en': "Plant Analysis Tool", 'zh': "植物分析工具"},
    'select_image': {'en': "Select an image:", 'zh': "选择图片:"},
    'btn_upload_analyze': {'en': "Upload & Analyze", 'zh': "上传并分析"},
    'analysis_title': {'en': "Analysis Result", 'zh': "分析结果"},
    'error_no_file_part': {'en': "No file part in the request!", 'zh': "请求中没有文件部分！"},
    'error_no_selected_file': {'en': "No selected file!", 'zh': "未选择文件！"},
    'error_failed_to_save_file': {'en': "Failed to save file.", 'zh': "保存文件失败。"},
    'error_occurred_during_processing': {'en': "An error occurred during processing.", 'zh': "处理过程中发生错误。"},
    'no_yolo_result': {'en': "No YOLO result", 'zh': "无YOLO结果"},
    'detected_disease': {'en': "Detected disease: {}", 'zh': "检测到的疾病: {}"},
    'recommended_remedy': {'en': "Recommended remedy: {}", 'zh': "推荐疗法: {}"},
    'no_disease_detected': {'en': "No disease detected.", 'zh': "未检测到疾病。"},
    'no_remedy_provided': {'en': "No remedy provided", 'zh': "未提供疗法"},
    'common_name_unknown': {'en': "Unknown", 'zh': "未知"},
    'unknown_soil_type': {'en': "Unknown", 'zh': "未知"},
    'unknown_disease': {'en': "Unknown Disease", 'zh': "未知疾病"},
    'error_disease_prediction': {'en': "Could not retrieve remedy information.", 'zh': "无法获取疗法信息。"},
    'image_num': {'en': "plant {}", 'zh': "植物 {}"},  # For dynamic YOLO output
    'yolo_growth_stage': {'en': "The predicted growth stage of {} is {}", 'zh': "{}的预测生长阶段是 {}"},
    # For dynamic YOLO output
    'error_fetching_weather_data': {'en': "Failed to fetch weather data", 'zh': "获取天气数据失败"},
    'error_retrieving_soil_values': {'en': "Error retrieving soil values: {}", 'zh': "获取土壤值失败: {}"},
    'error_no_et0_data': {'en': "No ET₀ data available", 'zh': "无ET₀数据可用"},
    'error_missing_fields': {'en': "Missing required fields: latitude, longitude, watering_start, watering_end",
                             'zh': "缺少必填字段: 纬度, 经度, 浇水开始时间, 浇水结束时间"},
    'error_failed_to_save_csv': {'en': "Failed to save watering plan to CSV: {}", 'zh': "保存浇水计划到 CSV 失败: {}"},
    'error_server_error': {'en': "Server error: {}", 'zh': "服务器错误: {}"},
    'error_invalid_data': {'en': "Invalid data: {}", 'zh': "无效数据: {}"},
    'error_no_json_payload': {'en': "No JSON payload provided", 'zh': "未提供JSON负载"},
    'error_latitude_longitude_required': {'en': "Latitude and longitude are required", 'zh': "需要提供纬度和经度"},
    'error_provide_query': {'en': "Please provide a 'query' in the JSON payload.", 'zh': "请在JSON负载中提供'query'。"},
    'error_no_function_selected': {'en': "No function selected", 'zh': "未选择功能"},
    'watering_frequency_times_per_week': {'en': "{}-{} times per week", 'zh': "每周 {}-{} 次"},

    # Disease Remedies (original English text)
    'remedy_healthy': {'en': "Plant is healthy, no treatment needed.", 'zh': "植物健康，无需治疗。"},
    "remedy_Potato___Early_blight": {
        'en': "Prune or stake plants to improve air circulation and reduce fungal problems. Make sure to disinfect your pruning shears (one part bleach to 4 parts water) after each cut. Keep the soil under plants clean and free of garden debris. Add a layer of organic compost to prevent the spores from splashing back up onto vegetation. Drip irrigation and soaker hoses can be used to help keep the foliage dry. For best control, apply copper-based fungicides early, two weeks before disease normally appears or when weather forecasts predict a long period of wet weather. Alternatively, begin treatment when disease first appears, and repeat every 7-10 days for as long as needed.",
        'zh': "修剪或支撑植物，以改善空气流通并减少真菌问题。每次修剪后务必用漂白剂（1份漂白剂兑4份水）消毒修枝剪。保持植物下方的土壤清洁，没有园艺垃圾。添加一层有机堆肥，以防止孢子溅回植物上。可以使用滴灌和浸泡软管，以帮助保持叶片干燥。为获得最佳控制效果，请在病害通常出现前两周或天气预报有长时间潮湿天气时，尽早施用铜基杀菌剂。或者，在病害首次出现时开始治疗，并根据需要每7-10天重复一次。"
    },
    "remedy_Potato___Late_blight": {
        'en': "Plant resistant cultivars when available. Remove volunteers from the garden prior to planting and space plants far enough apart to allow for plenty of air circulation. Water in the early morning hours, or use soaker hoses, to give plants time to dry out during the day — avoid overhead irrigation. Destroy all tomato and potato debris after harvest",
        'zh': "在条件允许的情况下种植抗病品种。在种植前清除花园中的自生植物，并确保植物间距足够大，以利于空气流通。在清晨浇水，或使用浸泡软管，让植物在白天有时间干燥——避免高空灌溉。收获后销毁所有番茄和马铃薯残骸。"
    },
    "remedy_Tomato___Early_blight": {
        'en': "Use pathogen-free seed, or collect seed only from disease-free plants. Rotate out of tomatoes and related crops for at least two years. Control susceptible weeds such as black nightshade and hairy nightshade, and volunteer tomato plants throughout the rotation. Fertilize properly to maintain vigorous plant growth. Particularly, do not over-fertilize with potassium and maintain adequate levels of both nitrogen and phosphorus. Avoid working in plants when they are wet from rain, irrigation, or dew. Use drip irrigation instead of overhead irrigation to keep foliage dry. Stake the plants to increase airflow around the plant and facilitate drying. Staking will also reduce contact between the leaves and spore-contaminated soil.",
        'zh': "使用无病种子，或只从无病植物上收集种子。至少两年内避免在番茄和相关作物上轮作。在整个轮作期间控制易感杂草，如黑茄和毛茄，以及自生番茄植株。适当施肥以保持植物旺盛生长。特别是，不要过度施用钾肥，并保持足够的氮和磷水平。避免在植物因雨水、灌溉或露水而湿润时进行操作。使用滴灌代替高空灌溉，以保持叶片干燥。给植物打桩，以增加植物周围的空气流通并促进干燥。打桩还可以减少叶片与受孢子污染的土壤之间的接触。"
    },
    "remedy_Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        'en': "The most effective treatments used to control the spread of TYLCV are insecticides and resistant crop varieties. Symptomatic plants should be carefully covered by a clear or black plastic bag and tied at the stem at soil line. If symptomatic plants have no obvious whiteflies on the lower leaf surface, these plants can be cut from the garden and BURIED in the compost.",
        'zh': "控制TYLCV传播最有效的治疗方法是杀虫剂和抗病作物。有症状的植物应小心地用透明或黑色塑料袋覆盖，并在土壤线处系在茎上。如果患病植物叶片下表面没有明显的粉虱，这些植物可以从花园中剪下并埋入堆肥中。"
    },
    "remedy_Tomato___Bacterial_spot": {
        'en': "Using pathogen-free seed and disease-free transplants, when possible, is the best way to avoid bacterial spot on tomato. Avoiding sprinkler irrigation and cull piles near greenhouse or field operations, and rotating with a nonhost crop also helps control the disease.",
        'zh': "在可能的情况下，使用无病原体种子和无病种苗是避免番茄细菌性斑点的最佳方法。避免在温室或田间作业附近进行喷灌和堆放废弃物，以及与非寄主作物轮作也有助于控制病害。"
    },
    "remedy_Tomato___Leaf_Mold": {
        'en': "Upon noticing the infected areas, the first step is to let the plants air out and dry. If they are being cultivated in a greenhouse, expose them to dry air conditions, because the humidity that the fungus needs to survive and thrive is dried up in the open air. If the tomatoes are being cultivated outdoors, try to keep the leaves dry when watering the plants. One thing you can do to help keep the leaves as dry as possible is to water in the early morning hours, that way the plant has plenty of time to dry before the sun comes out, which will keep the humidity around the leaves low. You can also try drip irrigation methods, or soak watering methods to attempt to water the soil without ever wetting the leaves of the plant.",
        'zh': "一旦发现感染区域，第一步是让植物通风干燥。如果它们在温室中种植，请将其暴露在干燥的空气条件下，因为真菌生存和繁殖所需的湿度在露天中会干燥。如果在户外种植番茄，浇水时尽量保持叶片干燥。您可以采取的一个措施是在清晨浇水，这样植物在阳光出来之前有充足的时间干燥，这将使叶片周围的湿度保持在低水平。您还可以尝试滴灌或浸泡浇水方法，以尝试浇灌土壤而不会弄湿植物叶片。"
    },
    "remedy_Tomato___Target_Spot": {
        'en': "Remove old plant debris at the end of the growing season; otherwise, the spores will travel from debris to newly planted tomatoes in the following growing season, thus beginning the disease anew. Dispose of the debris properly and don't place it on your compost pile unless you're sure your compost gets hot enough to kill the spores. Rotate crops and don't plant tomatoes in areas where other disease-prone plants have been located in the past year– primarily eggplant, peppers, potatoes or, of course– tomatoes. Rutgers University Extension recommends a three year rotation cycle to reduce soil-borne fungi. Pay careful attention to air circulation, as target spot of tomato thrives in humid conditions. Grow the plants in full sunlight. Be sure the plants aren't crowded and that each tomato has plenty of air circulation. Cage or stake tomato plants to keep the plants above the soil.",
        'zh': "在生长季节结束时清除旧的植物残骸；否则，孢子将从残骸传播到下个生长季节新种植的番茄上，从而再次引发病害。正确处理残骸，不要将其放入堆肥堆中，除非您确定堆肥足够热以杀死孢子。轮作作物，不要在过去一年中种植过其他易患病植物（主要是茄子、辣椒、土豆，当然还有番茄）的区域种植番茄。罗格斯大学推广部建议采用三年轮作周期来减少土传真菌。特别注意空气流通，因为番茄目标斑点病在潮湿条件下生长旺盛。在充足的阳光下种植植物。确保植物不拥挤，并且每个番茄都有充足的空气流通。给番茄植物搭建笼子或支撑，以使植物远离土壤。"
    },
    "remedy_Tomato___Spider_mites Two-spotted_spider_mite": {
        'en': "The best way to begin treating for two-spotted mites is to apply a pesticide specific to mites called a miticide. Ideally, you should start treating for two-spotted mites before your plants are seriously damaged. Apply the miticide for control of two-spotted mites every 7 days or so. Since mites can develop resistance to chemicals, switch to another type of miticide after three applications.",
        'zh': "治疗二斑叶螨的最佳方法是施用一种专门针对螨虫的杀螨剂。理想情况下，您应该在植物受到严重损害之前开始治疗二斑叶螨。每7天左右施用一次杀螨剂以控制二斑叶螨。由于螨虫可能对化学物质产生抗药性，因此在三次施用后切换到另一种杀螨剂。"
    },
    "remedy_Tomato___Tomato_mosaic_virus": {
        'en': "The use of ToMV-resistant varieties is generally the best way to reduce losses from this disease. Avoid planting in soil from previous crops that were infected with ToMV. Steam sterilizing the potting soil and containers as well as all equipment after each crop can reduce disease incidence. Before handling containers or plants be sure all workers wash with soap and water. Sterilizing pruning utensils or snapping off suckers without touching the plant instead of knife pruning help reduce disease incidence. Direct seeding in the field can help reduce the spread of ToMV.",
        'zh': "使用抗ToMV品种通常是减少该病害损失的最佳方法。避免在以前感染ToMV的作物土壤中种植。每季作物后对盆栽土壤、容器以及所有设备进行蒸汽消毒可以减少病害发生率。在处理容器或植物之前，务必确保所有工人用肥皂和水洗手。对修剪工具进行消毒或不接触植物直接掰掉吸芽而不是用刀修剪有助于减少病害发生率。田间直播有助于减少ToMV的传播。"
    },
    "remedy_Tomato___Septoria_leaf_spot": {
        'en': "Remove infected leaves immediately, and be sure to wash your hands and pruners thoroughly before working with uninfected plants. Fungicides containing either copper or potassium bicarbonate will help prevent the spreading of the disease. Begin spraying as soon as the first symptoms appear and follow the label directions for continued management. While chemical options are not ideal, they may be the only option for controlling advanced infections. One of the least toxic and most effective is chlorothalonil (sold under the names Fungonil and Daconil).",
        'zh': "立即清除受感染的叶片，并在处理未感染植物之前务必彻底洗净双手和修枝剪。含有铜或碳酸氢钾的杀菌剂将有助于防止病害蔓延。一旦出现首批症状，立即开始喷洒，并遵循标签说明进行持续管理。虽然化学方法不理想，但它们可能是控制晚期感染的唯一选择。毒性最低且最有效的方法之一是百菌清（以Fungonil和Daconil名称出售）。"
    },
    "remedy_Tomato___Late_blight": {
        'en': "Plant resistant cultivars when available. Remove volunteers from the garden prior to planting and space plants far enough apart to allow for plenty of air circulation. Water in the early morning hours, or use soaker hoses, to give plants time to dry out during the day — avoid overhead irrigation. Destroy all tomato and potato debris after harvest. Apply a copper based fungicide (2 oz/ gallon of water) every 7 days or less, following heavy rain or when the amount of disease is increasing rapidly. If possible, time applications so that at least 12 hours of dry weather follows application.",
        'zh': "在有抗病品种时种植。种植前清除园中的自生植物，并使植物间距足够大，以利于空气流通。在清晨浇水，或使用浸泡软管，让植物在白天有时间干燥——避免高空灌溉。收获后销毁所有番茄和马铃薯残骸。在暴雨后或病害迅速增加时，每7天或更短时间施用一次铜基杀菌剂（2盎司/加仑水）。如果可能，请安排施药时间，使其在施药后至少有12小时的干燥天气。"
    },
    "remedy_Cherry_(including_sour)___Powdery_mildew": {
        'en': "Plant resistant cultivars in sunny locations whenever possible. Prune or stake plants to improve air circulation. Make sure to disinfect your pruning tools (one part bleach to 4 parts water) after each cut. Remove diseased foliage from the plant and clean up fallen debris on the ground. Use a thick layer of mulch or organic compost to cover the soil after you have raked and cleaned it well. Mulch will prevent the disease spores from splashing back up onto the leaves. Milk sprays, made with 40% milk and 60% water, are an effective home remedy for use on a wide range of plants. For best results, spray plant leaves as a preventative measure every 10-14 days. Wash foliage occasionally to disrupt the daily spore-releasing cycle. Neem oil and PM Wash, used on a 7 day schedule, will prevent fungal attack on plants grown indoors.",
        'zh': "尽可能在阳光充足的地方种植抗病品种。修剪或支撑植物以改善空气流通。每次修剪后务必用漂白剂（1份漂白剂兑4份水）消毒修枝工具。从植物上清除病叶并清理地面上的落叶残骸。在耙地和清理干净后，用厚厚一层覆盖物或有机堆肥覆盖土壤。覆盖物将防止病害孢子溅回到叶片上。牛奶喷雾（40%牛奶和60%水）是适用于多种植物的有效家庭疗法。为获得最佳效果，每10-14天喷洒植物叶片作为预防措施。偶尔清洗叶片以中断每日孢子释放周期。印楝油和PM Wash，按照7天计划使用，将防止室内植物受到真菌侵害。"
    },
    "remedy_Pepper,_bell___Bacterial_spot": {
        'en': "Seed treatment with sodium hypochlorite (for example, Clorox) is effective in reducing bacterial populations on seed surfaces. Control of bacterial spot on greenhouse transplants is an essential step for preventing spread of the leaf spot bacteria in the field. Transplants should be inspected regularly to identify symptomatic seedlings. Transplants with symptoms may be removed and destroyed or treated with streptomycin. Products containing microorganisms can be used to enhance plant growth and reduce the negative effects of diseases. These products may contain plant-growth-promoting rhizobacteria (PGPR) or biological agents",
        'zh': "用次氯酸钠（例如，Clorox）处理种子可有效减少种子表面的细菌数量。控制温室移栽苗上的细菌斑点是防止叶斑细菌在田间传播的重要步骤。应定期检查移栽苗以识别有症状的幼苗。有症状的移栽苗可以被移除和销毁或用链霉素处理。含有微生物的产品可用于促进植物生长和减少疾病的负面影响。这些产品可能含有植物生长促进根际细菌（PGPR）或生物制剂。"
    },
    "remedy_Grape___Black_rot": {
        'en': "Prune away infected branches and remove mummified fruit from your grapevines to cut down on fungal spores. Always disinfect your pruning shears with one part bleach to four parts water after each cut. Keep the area under the vines clear of fallen leaves and debris, and add a light layer of organic compost to minimize spore splash. Water at the base with drip irrigation or soaker hoses to keep the foliage dry. When dark lesions first appear, apply an organic copper-based fungicide two weeks before the disease usually shows or when extended wet weather is forecast. Alternatively, treat as soon as symptoms appear, and repeat every 7-10 days until the infection subsides.",
        'zh': "修剪受感染的枝条，并从葡萄藤上清除木乃伊状的果实，以减少真菌孢子。每次修剪后务必用一份漂白剂兑四份水的比例消毒修枝剪。保持藤蔓下方的区域没有落叶和碎片，并添加一层薄薄的有机堆肥，以尽量减少孢子飞溅。使用滴灌或浸泡软管在根部浇水，以保持叶片干燥。当首次出现黑色病斑时，在病害通常出现前两周或预报有长时间潮湿天气时，施用有机铜基杀菌剂。或者，在症状一出现时就进行处理，并每7-10天重复一次，直到感染消退。"
    },
    "remedy_Orange___Haunglongbing_(Citrus_greening)": {
        'en': "Prune out any branches with mottled yellow leaves or dieback, and remove symptomatic fruit to reduce disease spread. Disinfect your pruning tools with a bleach solution after every cut. Clear fallen leaves and fruit from the area and avoid leaving debris that could harbor the bacteria. Water using drip irrigation at the base of the tree to keep the foliage dry. At the first signs of greening, apply an organic insecticidal spray such as neem oil to deter the Asian citrus psyllid, and repeat treatments every 7-10 days during extended wet or pest-active periods. Keeping your tree healthy is key to slowing the progress of HLB.",
        'zh': "剪掉任何带有斑驳黄叶或枯死现象的枝条，并清除有症状的果实以减少疾病传播。每次修剪后用漂白剂溶液消毒修剪工具。清除该区域的落叶和果实，避免留下可能藏匿细菌的碎屑。使用滴灌在树根部浇水，以保持叶片干燥。在黄龙病首次出现迹象时，施用有机杀虫剂（如印楝油）以阻止亚洲柑橘木虱，并在长时间潮湿或病虫害活跃期间每7-10天重复处理。保持树木健康是减缓HLB进展的关键。"
    },
    "remedy_Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        'en': "Prune off leaves and shoots that show early signs of blight to improve air flow and reduce fungal spread. Disinfect your shears with a one-to-four bleach-to-water solution after each cut. Remove all fallen leaves and debris from under the vines, and lightly top-dress with organic compost to help block spore splash. Use drip irrigation to water at the base and keep foliage dry. As soon as leaf blight symptoms emerge, treat with an organic copper fungicide two weeks before the disease typically develops, or when wet weather is forecast, repeating every 7-10 days until control is achieved.",
        'zh': "修剪掉出现早期枯萎迹象的叶片和新梢，以改善空气流通并减少真菌传播。每次修剪后用1:4的漂白水溶液消毒剪刀。清除藤蔓下方的所有落叶和碎屑，并轻微追施有机堆肥，以帮助阻止孢子飞溅。使用滴灌在根部浇水，保持叶片干燥。一旦叶枯病症状出现，在病害通常发生前两周或预报潮湿天气时，用有机铜杀菌剂处理，每7-10天重复一次，直到控制住病害。"
    },
    "remedy_Apple___Cedar_apple_rust": {
        'en': "Prune and remove any branches bearing rust-colored spots and clear out all infected leaves and fruit from around the tree. Always disinfect your pruning tools with a bleach solution after each cut to prevent cross-contamination. Rake up and discard fallen debris, then add a thin layer of organic compost to deter spore splash. Water at the base using drip irrigation to keep the foliage dry. When rust symptoms first appear, treat with an organic copper-based fungicide two weeks before the usual rust season or during prolonged wet spells, and reapply every 7-10 days until the disease is controlled.",
        'zh': "修剪并清除任何带有锈色斑点的枝条，并清除树周围所有受感染的叶片和果实。每次修剪后务必用漂白剂溶液消毒修剪工具，以防止交叉污染。耙起并丢弃落叶残骸，然后添加一层薄薄的有机堆肥以阻止孢子飞溅。使用滴灌在树根部浇水，保持叶片干燥。当锈病症状首次出现时，在通常锈病季节前两周或在长时间潮湿天气期间，用有机铜基杀菌剂处理，并每7-10天重复施用，直到病害得到控制。"
    },
    "remedy_Corn_(maize)___Common_rust_": {
        'en': "Thin your corn to improve air circulation and remove any rusted leaves immediately to reduce fungal load. Discard infected material and disinfect any tools used with a 1:4 bleach solution. Keep the area free of debris and apply a light layer of organic compost to lessen spore buildup. Water at the soil level with drip irrigation, avoiding overhead methods that wet the leaves. As soon as small rust pustules appear, begin treatment with an organic copper fungicide two weeks before common rust usually appears or when extended wet conditions are forecast, and repeat every 7-10 days until the infection is managed.",
        'zh': "稀植玉米以改善空气流通，并立即清除任何生锈的叶片以减少真菌负荷。丢弃受感染的材料，并用1:4的漂白剂溶液消毒所有使用的工具。保持该区域无碎屑，并施加一层薄薄的有机堆肥以减少孢子积聚。用滴灌在土壤表面浇水，避免使用会弄湿叶片的高空浇水方法。一旦出现小的锈病脓疱，在普通锈病通常出现前两周或预报有长时间潮湿条件时，开始用有机铜杀菌剂处理，并每7-10天重复一次，直到感染得到控制。"
    },
    "remedy_Grape___Esca_(Black_Measles)": {
        'en': "Prune out vine sections with dark streaks and spotted symptoms, cutting well below the infected area. Disinfect your pruning shears with a bleach solution after every cut to avoid spreading the pathogen. Remove all fallen, diseased leaves and debris from around the vine, and top-dress with organic compost to help suppress spore movement. Water at the base using drip irrigation to keep foliage dry. At the first sign of Esca, treat with an organic copper-based fungicide two weeks before symptoms typically occur or when prolonged wet weather is forecast, reapplying every 7-10 days until the disease is controlled.",
        'zh': "修剪掉有黑色条纹和斑点症状的藤蔓部分，修剪位置应远低于受感染区域。每次修剪后用漂白剂溶液消毒修枝剪，以避免病原体传播。清除藤蔓周围所有掉落的病叶和碎屑，并用有机堆肥覆盖土壤，以帮助抑制孢子移动。使用滴灌在根部浇水，保持叶片干燥。在埃斯卡病首次出现迹象时，在症状通常发生前两周或预报有长时间潮湿天气时，用有机铜基杀菌剂处理，并每7-10天重复施用，直到病害得到控制。"
    },
    "remedy_Apple___Apple_scab": {
        'en': "Prune away infected branches and remove any leaves or fruit with scabby lesions to limit apple scab. Always disinfect your shears using a bleach solution after each cut. Keep the area beneath the tree clear of fallen debris, and spread a thin layer of organic compost to minimize spore splash. Use drip irrigation to water only at the base, avoiding wetting the leaves. When early symptoms of scab appear, apply an organic sulfur or copper fungicide two weeks before scab usually develops or when extended wet weather is predicted, and repeat every 7-10 days until the disease subsides.",
        'zh': "修剪受感染的枝条，并清除任何带有痂状病斑的叶片或果实，以限制苹果黑星病。每次修剪后务必用漂白剂溶液消毒剪刀。保持树下区域没有落叶残骸，并铺一层薄薄的有机堆肥，以尽量减少孢子飞溅。使用滴灌只在根部浇水，避免弄湿叶片。当黑星病早期症状出现时，在通常发病前两周或预报有长时间潮湿天气时，施用有机硫或铜杀菌剂，并每7-10天重复一次，直到病害消退。"
    },
    "remedy_Corn_(maize)___Northern_Leaf_Blight": {
        'en': "Thin your corn to promote air circulation and remove any lower leaves that show tan, elongated lesions. Discard all infected material and disinfect your tools with a bleach solution after use. Keep the ground free of debris and apply a light layer of organic compost to reduce spore buildup. Water using drip irrigation at the base to prevent prolonged leaf wetness. As soon as blight symptoms are visible, treat with an organic copper-based fungicide two weeks before the disease normally appears or during extended wet periods, repeating every 7-10 days until the infection is under control.",
        'zh': "稀植玉米以促进空气流通，并清除任何出现黄褐色、细长病斑的下部叶片。丢弃所有受感染的材料，并在使用后用漂白剂溶液消毒工具。保持地面没有碎屑，并施加一层薄薄的有机堆肥以减少孢子积聚。使用滴灌在根部浇水，以防止叶片长时间潮湿。一旦枯萎病症状可见，在病害通常出现前两周或在长时间潮湿期间，用有机铜基杀菌剂处理，并每7-10天重复一次，直到感染得到控制。"
    },
    "remedy_Peach___Bacterial_spot": {
        'en': "Prune off branches showing dark lesions and oozing spots, and remove any fallen fruit or leaves to reduce bacterial spread. Disinfect your pruning shears with a 1:4 bleach-to-water solution after each cut. Clear the area beneath the tree of debris, and add a thin layer of organic compost to block spore splash. Water at the base with drip irrigation to keep the foliage dry. At the first sign of bacterial spot, treat the tree with an organic copper-based bactericide two weeks before the disease typically emerges or during prolonged wet weather, and repeat every 7-10 days until the symptoms diminish.",
        'zh': "剪掉出现黑色病斑和渗液斑点的枝条，并清除任何掉落的果实或叶片，以减少细菌传播。每次修剪后用1:4的漂白水溶液消毒修枝剪。清除树下区域的碎屑，并添加一层薄薄的有机堆肥以阻止孢子飞溅。使用滴灌在根部浇水，保持叶片干燥。在细菌斑点病首次出现迹象时，在病害通常发生前两周或在长时间潮湿天气期间，用有机铜基杀菌剂处理树木，并每7-10天重复一次，直到症状减轻。"
    },
    "remedy_Squash___Powdery_mildew": {
        'en': "Prune away leaves that show the early white, powdery coating to reduce mildew spread. Disinfect your tools with a bleach solution after each cut, and remove all fallen leaves and debris from around the plants. Lightly top-dress the area with organic compost to prevent spores from splashing onto new growth. Water at the base using drip irrigation or soaker hoses to keep the foliage dry. When powdery mildew first appears, treat with an organic remedy like a baking soda spray or copper-based fungicide two weeks before the disease typically occurs or when prolonged wet conditions are forecast, and reapply every 7-10 days as needed.",
        'zh': "修剪掉出现早期白色粉状覆盖物的叶片，以减少白粉病的传播。每次修剪后用漂白剂溶液消毒工具，并清除植物周围所有掉落的叶片和碎屑。轻微追施有机堆肥，以防止孢子溅到新生长物上。使用滴灌或浸泡软管在根部浇水，保持叶片干燥。当白粉病首次出现时，在病害通常发生前两周或预报有长时间潮湿条件时，用有机疗法（如小苏打喷雾或铜基杀菌剂）处理，并根据需要每7-10天重复施用。"
    },
    "remedy_Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        'en': "Thin your corn plants to enhance air circulation and remove any leaves showing rectangular gray spots immediately. Discard infected material and disinfect your tools with a bleach solution after each use. Clear away plant debris from around the corn and apply a light layer of organic compost to reduce spore buildup. Water only at the base with drip irrigation to keep the foliage dry. At the first appearance of gray leaf spot, treat with an organic copper-based fungicide two weeks before the disease normally sets in or when extended wet weather is expected, and repeat every 7-10 days until the disease is controlled.",
        'zh': "稀植玉米以增强空气流通，并立即清除任何出现矩形灰色斑点的叶片。丢弃受感染的材料，并在每次使用后用漂白剂溶液消毒工具。清除玉米周围的植物碎屑，并施加一层薄薄的有机堆肥以减少孢子积聚。仅在根部使用滴灌浇水，以保持叶片干燥。在灰叶斑病首次出现时，在病害通常发生前两周或预计有长时间潮湿天气时，用有机铜基杀菌剂处理，并每7-10天重复一次，直到病害得到控制。"
    },
    "remedy_Apple___Black_rot": {
        'en': "Prune out any branches or fruit that exhibit dark, sunken lesions with concentric rings, and immediately remove the infected material. Disinfect your pruning shears with a one-to-four bleach solution after each cut to avoid spreading the rot. Keep the ground clear of fallen leaves and decaying fruit, and apply a thin layer of organic compost to reduce spore splash. Water at the base using drip irrigation to prevent moisture on the leaves. When black rot appears, treat the tree with an organic copper-based fungicide two weeks before the disease typically emerges or during extended wet weather, and reapply every 7-10 days until the infection is contained.",
        'zh': "修剪掉任何表现出深色、凹陷、同心圆病斑的枝条或果实，并立即清除受感染的材料。每次修剪后用1:4的漂白剂溶液消毒修枝剪，以避免腐烂传播。保持地面没有落叶和腐烂果实，并施加一层薄薄的有机堆肥以减少孢子飞溅。使用滴灌在根部浇水，以防止叶片潮湿。当黑腐病出现时，在病害通常发生前两周或在长时间潮湿天气期间，用有机铜基杀菌剂处理树木，并每7-10天重复施用，直到感染得到控制。"
    },
    "remedy_Strawberry___Leaf_scorch": {
        'en': "Prune away any strawberry leaves showing scorch damage and remove all plant debris from the bed to cut down on disease spread. Disinfect your tools with a bleach solution after each cut, and clear the area of excess mulch that might harbor spores. Lightly top-dress with organic compost to help prevent spore splash. Water at the base using drip irrigation to keep the foliage dry. At the first sign of leaf scorch, treat the plants with an organic copper-based fungicide two weeks before symptoms usually develop or when extended wet weather is forecast, and repeat every 7-10 days until the disease is controlled.",
        'zh': "修剪掉任何出现灼烧损伤的草莓叶片，并清除苗床上所有植物残骸，以减少疾病传播。每次修剪后用漂白剂溶液消毒工具，并清除可能藏匿孢子的多余覆盖物。轻微追施有机堆肥，以帮助防止孢子飞溅。使用滴灌在根部浇水，保持叶片干燥。当叶枯病首次出现迹象时，在症状通常发生前两周或预报有长时间潮湿天气时，用有机铜基杀菌剂处理植物，并每7-10天重复一次，直到病害得到控制。"
    },
    # Missing from original mapping, added as healthy for completeness
    "remedy_Strawberry___healthy": {'en': "Plant is healthy, no treatment needed.", 'zh': "植物健康，无需治疗。"},
    "remedy_Grape___healthy": {'en': "Plant is healthy, no treatment needed.", 'zh': "植物健康，无需治疗。"},
    "remedy_Blueberry___healthy": {'en': "Plant is healthy, no treatment needed.", 'zh': "植物健康，无需治疗。"},
    "remedy_Corn_(maize)___healthy": {'en': "Plant is healthy, no treatment needed.", 'zh': "植物健康，无需治疗。"},
    "remedy_Peach___healthy": {'en': "Plant is healthy, no treatment needed.", 'zh': "植物健康，无需治疗。"},
    "remedy_Pepper,_bell___healthy": {'en': "Plant is healthy, no treatment needed.", 'zh': "植物健康，无需治疗。"},
    "remedy_Raspberry___healthy": {'en': "Plant is healthy, no treatment needed.", 'zh': "植物健康，无需治疗。"},
    "remedy_Tomato___healthy": {'en': "Plant is healthy, no treatment needed.", 'zh': "植物健康，无需治疗。"},
    "remedy_Cherry_(including_sour)___healthy": {'en': "Plant is healthy, no treatment needed.",
                                                 'zh': "植物健康，无需治疗。"},
    "remedy_Apple___healthy": {'en': "Plant is healthy, no treatment needed.", 'zh': "植物健康，无需治疗。"},
    "remedy_Potato___healthy": {'en': "Plant is healthy, no treatment needed.", 'zh': "植物健康，无需治疗。"},
    "remedy_Soybean___healthy": {'en': "Plant is healthy, no treatment needed.", 'zh': "植物健康，无需治疗。"}
}


# Helper function to get translated text
def get_text(key, lang=None):
    """
    Retrieves the translated string for a given key based on the current session language.
    Defaults to English if the language is not set or the key is not found for the selected language.
    """
    if lang is None:
        lang = session.get('lang', 'en')
    return global_lang_strings.get(key, {}).get(lang, global_lang_strings.get(key, {}).get('en', key))


# ---------------- Weather, Soil, Crop, and Recommendation Functions ----------------

def get_open_meteo_forecast(lat, lon):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "shortwave_radiation,windspeed_10m,cloudcover,temperature_2m,relativehumidity_2m",
            "daily": "temperature_2m_max,temperature_2m_min",
            "windspeed_unit": "ms",
            "timezone": "auto"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data.get("daily") or not data.get("hourly"):
            raise ValueError("Incomplete data from Open-Meteo")
        return data
    except Exception as e:
        print(f"Open-Meteo API error: {e}. Falling back to Visual Crossing API.")
        vc_url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}"
        vc_params = {
            "unitGroup": "metric",
            "include": "days,hours",
            "key": VISUAL_CROSSING_API_KEY,
            "contentType": "json"
        }
        try:
            vc_resp = requests.get(vc_url, params=vc_params, timeout=10)
            vc_resp.raise_for_status()
            vc_data = vc_resp.json()
            transformed = {
                "daily": {
                    "time": [day["datetime"] for day in vc_data.get("days", [])],
                    "temperature_2m_max": [day["tempmax"] for day in vc_data.get("days", [])],
                    "temperature_2m_min": [day["tempmin"] for day in vc_data.get("days", [])],
                },
                "hourly": {}
            }
            if vc_data.get("days") and vc_data["days"][0].get("hours"):
                times = []
                rad = []
                ws = []
                cc = []
                temp = []
                hum = []
                for day in vc_data["days"]:
                    for hour in day.get("hours", []):
                        times.append(hour["datetime"])
                        rad.append(hour.get("solarradiation", None))
                        ws.append(hour.get("windspeed", None))
                        cc.append(hour.get("cloudcover", None))
                        temp.append(hour.get("temp", None))
                        hum.append(hour.get("humidity", None))
                transformed["hourly"] = {
                    "time": times,
                    "shortwave_radiation": rad,
                    "windspeed_10m": ws,
                    "cloudcover": cc,
                    "temperature_2m": temp,
                    "relativehumidity_2m": hum
                }
            return transformed
        except Exception as e_vc:
            print(f"Visual Crossing API error: {e_vc}")
            return None


def convert_wind_speed_to_two_meters(v10):
    exponent = 1.0 / 7.0
    return v10 * ((2.0 / 10.0) ** exponent)


def calculate_dew_point_temperature(T, RH):
    a, b = 17.27, 237.7
    if RH <= 0:
        return None
    alpha = math.log(RH / 100.0) + (a * T) / (b + T)
    return (b * alpha) / (a - alpha)


def write_hourly_data_to_csv(times, solar, wind10, wind2, cloud, rh, dew):
    with open(SOIL_WATER_CONTENT_CSV_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Time", "Solar Radiation (W/m²)", "Wind Speed 10m (m/s)",
                         "Estimated Wind Speed 2m (m/s)", "Cloud Cover (%)",
                         "Relative Humidity (%)", "Dew Point (°C)"])
        for t, s, w10_val, w2_val, c, r, d in zip(times, solar, wind10, wind2, cloud, rh, dew):
            writer.writerow([t, s, w10_val, w2_val, c, r, d])


def write_daily_data_to_csv(dates, tmax, tmin):
    with open(TEMPERATURE_CSV_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Date", "Max Temperature (°C)", "Min Temperature (°C)"])
        for d, mx, mn in zip(dates, tmax, tmin):
            writer.writerow([d, mx, mn])


def get_soil_values(tiff_path, lon, lat):
    try:
        with rasterio.open(tiff_path) as src:
            row, col = src.index(lon, lat)
            return src.read(2)[row, col], src.read(3)[row, col], src.read(4)[row, col]
    except Exception as e:
        raise Exception(f"Error reading soil data: {e}")


def determine_soil_classification(sand, silt, clay):
    total = sand + silt + clay
    if total <= 0:
        return "Invalid"
    if abs(total - 100) > 1e-6:
        sand, silt, clay = 100 * sand / total, 100 * silt / total, 100 * clay / total
    if sand >= 85 and clay < 10:
        return "Sand"
    if 70 <= sand < 90 and clay < 15 and silt <= 30:
        return "Loamy sand"
    if 43 <= sand < 85 and clay < 20 and silt <= 50:
        return "Sandy loam"
    if 23 <= sand <= 52 and 28 <= silt <= 50 and 7 <= clay <= 27:
        return "Loam"
    if 50 <= silt <= 80 and clay <= 27 and sand < 50:
        return "Silt loam"
    if silt >= 80 and clay < 12 and sand < 20:
        return "Silt"
    if 45 <= sand <= 65 and 20 <= clay < 35 and silt < 28:
        return "Sandy clay loam"
    if 27 <= clay <= 40 and 20 <= sand <= 45 and 15 <= silt <= 53:
        return "Clay loam"
    if 27 <= clay <= 40 and silt >= 40 and sand <= 20:
        return "Silty clay loam"
    if 45 <= sand <= 65 and clay >= 35 and silt <= 20:
        return "Sandy clay"
    if clay >= 40 and silt >= 40 and sand <= 20:
        return "Silty clay"
    if clay >= 40 and sand < 45 and silt < 40:
        return "Clay"
    return get_text("unknown_soil_type")  # Translate "Unclassified"


def get_soil_ph(lat, lon, tiff_path=SOIL_PH_TIFF_PATH):
    try:
        with rasterio.open(tiff_path) as src:
            raster_crs = src.crs
            transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
            row, col = src.index(x, y)
            band1 = src.read(1)
            ph_value = float(band1[row, col])
            if ph_value < 5 or ph_value > 10:
                # Use triangular distribution: min=5, max=9, mode=7
                ph_value = random.triangular(5, 9, 7)
            # Round to one decimal place
            return round(ph_value, 1)
    except Exception as e:
        print(f"Error fetching soil pH: {e}")
        return None


def write_soil_to_csv(lon, lat, sand, silt, clay, soil_type):
    with open(SOIL_CSV_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([get_text("Longitude"), get_text("Latitude"), get_text("Sand (%)"), get_text("Silt (%)"),
                         get_text("Clay (%)"), get_text("Soil Type")])
        writer.writerow([lon, lat, f"{sand:.1f}", f"{silt:.1f}", f"{clay:.1f}", soil_type])


def parse_pot_data(pots_text):
    pots = []
    for i, line in enumerate(pots_text.strip().splitlines()):
        parts = [p.strip() for p in line.split(',')]
        if not parts:
            continue
        shape = parts[0].lower()
        if shape == 'c':
            if len(parts) >= 3:
                try:
                    r = float(parts[1]) / 100.0
                    area_m2 = math.pi * (r ** 2)
                    plant = parts[2]
                    pots.append({"id": i + 1, "shape": "Cylinder", "area": area_m2, "plant": plant})
                except ValueError:
                    continue
            else:
                try:
                    r = float(parts[1]) / 100.0
                    area_m2 = math.pi * (r ** 2)
                    pots.append(
                        {"id": i + 1, "shape": "Cylinder", "area": area_m2, "plant": get_text("common_name_unknown")})
                except ValueError:
                    continue
        elif shape == 'r':
            if len(parts) >= 4:
                try:
                    length = float(parts[1]) / 100.0
                    width = float(parts[2]) / 100.0
                    area_m2 = length * width
                    plant = parts[3]
                    pots.append({"id": i + 1, "shape": "Rectangular", "area": area_m2, "plant": plant})
                except ValueError:
                    continue
            else:
                try:
                    length = float(parts[1]) / 100.0
                    width = float(parts[2]) / 100.0
                    area_m2 = length * width
                    pots.append({"id": i + 1, "shape": "Rectangular", "area": area_m2,
                                 "plant": get_text("common_name_unknown")})
                except ValueError:
                    continue
    return pots


def parse_yard_area_data(yard_text):
    total_area = 0
    for line in yard_text.strip().splitlines():
        parts = line.split(',')
        if len(parts) >= 2:
            try:
                length = float(parts[0].strip())
                width = float(parts[1].strip())
                total_area += length * width
            except ValueError:
                continue
    return {"area": total_area}


def parse_yard_plant_names(yard_plants_text):
    if not yard_plants_text:
        return []
    return [p.strip() for p in yard_plants_text.split(',') if p.strip()]


def parse_yard_plant_area_data(yard_plant_areas_text):
    plants = []
    for line in yard_plant_areas_text.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:
            try:
                plant = parts[0]
                area = float(parts[1])
                plants.append({"plant": plant, "area": area})
            except ValueError:
                continue
    return plants


def export_total_area_to_csv():
    with open(AREA_CSV_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([get_text("Type"), get_text("Number"), get_text("Area")])
        for pot in area_data["pots"]:
            writer.writerow([get_text("Pot"), pot["id"], f"{pot['area']:.4f}"])
        if area_data["yard"] and area_data["yard"].get("area", 0) > 0:
            writer.writerow([get_text("Yard"), "1", f"{area_data['yard']['area']:.2f}"])


def compute_reference_evapotranspiration(T_max, T_min, RH, u2, Rs):
    T_avg = (T_max + T_min) / 2.0
    es_Tmax = 0.6108 * math.exp((17.27 * T_max) / (T_max + 237.3))
    es_Tmin = 0.6108 * math.exp((17.27 * T_min) / (T_min + 237.3))
    es_avg = (es_Tmax + es_Tmin) / 2.0
    ea = (RH / 100.0) * (0.6108 * math.exp((17.27 * T_avg) / (T_avg + 237.3)))
    Delta = 4098 * (0.6108 * math.exp((17.27 * T_avg) / (T_avg + 237.3))) / ((T_avg + 237.3) ** 2)
    gamma = 0.000665
    lambda_val = 2.45
    sigma_hourly = 4.903e-9 / 24
    alpha = 0.23
    T_k = T_avg + 273.16
    Rn = (1 - alpha) * Rs - sigma_hourly * (T_k ** 4) * (0.56 - 0.08 * math.sqrt(ea))
    G = 0
    ET0 = ((Delta * (Rn - G)) + (gamma * (900 / (T_avg + 273)) * u2 * (es_avg - ea))) / (
            lambda_val * (Delta + gamma * (1 + 0.34 * u2)))
    return max(ET0, 0)


def watering_recommendation_for_pot(soil_type, daily_ET0, pot_area):
    pot_params = {
        "Sand": {"sessions": 2, "multiplier": 0.5},
        "Loamy sand": {"sessions": 2, "multiplier": 0.55},
        "Sandy loam": {"sessions": 1, "multiplier": 1.0},
        "Loam": {"sessions": 1, "multiplier": 1.0},
        "Silt loam": {"sessions": 1, "multiplier": 1.1},
        "Silt": {"sessions": 1, "multiplier": 1.2},
        "Sandy clay loam": {"sessions": 1, "multiplier": 1.2},
        "Clay loam": {"sessions": 1, "multiplier": 1.3},
        "Silty clay loam": {"sessions": 1, "multiplier": 1.4},
        "Sandy clay": {"sessions": 1, "multiplier": 1.5},
        "Silty clay": {"sessions": 1, "multiplier": 1.6},
        "Clay": {"sessions": 1, "multiplier": 1.7},
    }
    params = pot_params.get(soil_type, {"sessions": 1, "multiplier": 1.0})
    base_volume = daily_ET0 * pot_area
    water_per_session = (base_volume * params["multiplier"]) / params["sessions"]
    return {"amount": f"{water_per_session:.2f}"}


def watering_recommendation_for_yard(soil_type, daily_ET0, area):
    yard_params = {
        "Sand": {"multiplier": 0.7},
        "Loamy sand": {"multiplier": 0.75},
        "Sandy loam": {"multiplier": 1.0},
        "Loam": {"multiplier": 1.0},
        "Silt loam": {"multiplier": 1.1},
        "Silt": {"multiplier": 1.2},
        "Sandy clay loam": {"multiplier": 1.3},
        "Clay loam": {"multiplier": 1.4},
        "Silty clay loam": {"multiplier": 1.5},
        "Sandy clay": {"multiplier": 1.6},
        "Silty clay": {"multiplier": 1.7},
        "Clay": {"multiplier": 1.8},
    }
    params = yard_params.get(soil_type, {"multiplier": 1.0})
    base_volume = daily_ET0 * area
    water_volume = base_volume * params["multiplier"]
    return f"{water_volume:.2f}"


def load_plant_database():
    global plant_database_dataframe
    if plant_database_dataframe is None:
        try:
            plant_database_dataframe = pd.read_csv(PLANT_DATABASE_PATH)
        except Exception as e:
            print(f"Error loading plant database: {e}")
            plant_database_dataframe = pd.DataFrame()
    return plant_database_dataframe


def calculate_watering_frequency_for_plant(plant, multiplier):
    try:
        df = load_plant_database()
        matching = df[df['plant_name'].str.lower() == plant.lower()]
        if matching.empty:
            min_freq, max_freq = 2, 3
        else:
            freq_str = matching.iloc[0]['watering_frequency']
            m = re.search(r"(\d+)\s*-\s*(\d+)", freq_str)
            if not m:
                min_freq, max_freq = 2, 3
            else:
                min_freq = int(m.group(1))
                max_freq = int(m.group(2))
                if min_freq > max_freq:
                    min_freq, max_freq = max_freq, min_freq
    except Exception as e:
        print(f"Error calculating watering frequency: {e}")
        min_freq, max_freq = 2, 3
    new_min = math.ceil(min_freq * multiplier)
    new_max = math.ceil(max_freq * multiplier)

    # Use get_text for this dynamic string
    guideline = get_text("watering_frequency_times_per_week").format(new_min, new_max)
    return guideline, new_min, new_max


def fetch_api(url, params=None):
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return None


def get_alternative_precipitation(lat, lon, start_date, end_date):
    start_iso = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    end_iso = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
    vc_url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}"
    params = {
        "unitGroup": "metric",
        "include": "days,hours",
        "key": VISUAL_CROSSING_API_KEY,
        "contentType": "json"
    }
    data = fetch_api(vc_url, params)
    if data and "days" in data:
        total = 0.0
        for day in data["days"]:
            total += day.get("precip", 0)
        return total
    return get_text("No data")


def get_monthly_average_temperature_vc(lat, lon, month, year):
    first_day = 1
    last_day = calendar.monthrange(year, month)[1]
    start_date = f"{year:04d}-{month:02d}-{first_day:02d}"
    end_date = f"{year:04d}-{month:02d}-{last_day:02d}"
    vc_url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{start_date}/{end_date}"
    params = {
        "unitGroup": "metric",
        "key": VISUAL_CROSSING_API_KEY,
        "contentType": "json"
    }
    data = fetch_api(vc_url, params)
    if data and "days" in data:
        temps = [day.get("temp", None) for day in data["days"] if day.get("temp") is not None]
        if temps:
            return sum(temps) / len(temps)
    return get_text("No data")


def get_climate_data(lat, lon, month, year, max_years_back=1):
    import random  # Ensure random is imported
    now = datetime.now()
    if year > now.year or (year == now.year and month > now.month):
        return {"Error": get_text("Future date selected.")}  # Translate this error
    first_day = 1
    last_day = now.day if (year == now.year and month == now.month) else calendar.monthrange(year, month)[1]
    attempts, adjusted_year = 0, year
    while attempts < max_years_back:
        start = datetime(adjusted_year, month, first_day).strftime("%Y%m%d")
        end = datetime(adjusted_year, month, last_day).strftime("%Y%m%d")
        params = {
            "parameters": "T2M,PRECTOT",
            "community": "RE",
            "longitude": lon,
            "latitude": lat,
            "start": start,
            "end": end,
            "format": "JSON"
        }
        climate = fetch_api("https://power.larc.nasa.gov/api/temporal/daily/point", params)
        if climate and "properties" in climate and "parameter" in climate["properties"]:
            p = climate["properties"]["parameter"]
            valid_t2m = [v for v in p.get("T2M", {}).values() if v != -999.0]
            if valid_t2m:
                avg_t2m = sum(valid_t2m) / len(valid_t2m)
            else:
                avg_t2m = get_monthly_average_temperature_vc(lat, lon, month, adjusted_year)
                if avg_t2m == get_text("No data") or not avg_t2m:  # Use get_text
                    fb_month = month - 1 if month > 1 else 12
                    fb_year = adjusted_year if month > 1 else adjusted_year - 1
                    avg_t2m = get_monthly_average_temperature_vc(lat, lon, fb_month, fb_year)
            valid_precip = [v for v in p.get("PRECTOT", {}).values() if v != -999.0]
            total_precip = sum(valid_precip) if valid_precip else get_alternative_precipitation(lat, lon, start, end)
            full_days = calendar.monthrange(adjusted_year, month)[1]
            if total_precip != get_text("No data") and isinstance(total_precip, (int, float)):  # Use get_text
                if adjusted_year == now.year and month == now.month and now.day < full_days:
                    elapsed_days = now.day
                    if elapsed_days > 0:
                        scaled_precip = (total_precip / elapsed_days) * full_days
                        if total_precip <= 120 and scaled_precip > 120:
                            total_precip = total_precip
                        elif total_precip > 120:
                            total_precip = random.uniform(60, 100)
                        else:
                            total_precip = scaled_precip
                else:
                    if total_precip > 120:
                        total_precip = random.uniform(60, 100)
            if avg_t2m != get_text("No data") and total_precip != get_text("No data"):  # Use get_text
                return {"Average Temperature (T2M)": avg_t2m, "Total Precipitation (PRECTOT)": total_precip}
        adjusted_year -= 1
        attempts += 1
    return {"Error": get_text("No climate data available.")}  # Translate this error


def get_current_weather(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    return fetch_api(url, params)


def get_location_info(lat, lon, month, year):
    report = {}
    report["Climate Data"] = get_climate_data(lat, lon, month, year)
    weather = get_current_weather(lat, lon)
    if weather and "main" in weather:
        m = weather["main"]
        report["Weather Data"] = {
            "Temperature": m.get("temp"),
            "Humidity": m.get("humidity"),
            "Pressure": m.get("pressure")
        }
    try:
        sand, silt, clay = get_soil_values(DEFAULT_TIFF_PATH, lon, lat)
        sand_p = sand * 100.0
        silt_p = silt * 100.0
        clay_p = clay * 100.0
        stype = determine_soil_classification(sand_p, silt_p, clay_p)
    except Exception:
        stype = get_text("unknown_soil_type")  # Use get_text for "Unknown"
    report["Soil Classification"] = stype
    report["Soil pH"] = get_soil_ph(lat, lon)
    return report


def extract_sensor_data(report):
    sensor = {}
    mapping = {'T': ('Weather Data', 'Temperature'),
               'H': ('Weather Data', 'Humidity'),
               'P': ('Weather Data', 'Pressure'),
               'T_avg': ('Climate Data', 'Average Temperature (T2M)'),
               'AP': ('Climate Data', 'Total Precipitation (PRECTOT)')}
    for key, (sec, sub) in mapping.items():
        val = report.get(sec, {}).get(sub)
        try:
            sensor[key] = float(val) if val and str(val).strip() != get_text("No data") else None  # Use get_text
        except:
            sensor[key] = None
    ph_val = report.get("Soil pH")
    sensor["pH"] = float(ph_val) if ph_val is not None else None
    return sensor


def load_local_crop_datasets(folder):
    csv_files = glob.glob(os.path.join(folder, '*.csv'))
    if not csv_files:
        return None
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logging.warning(f"Could not read CSV file {file}: {e}")
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else None


def compute_optimal_conditions(df):
    col = next((c for c in ['label', 'crop', 'common_name', 'plant_name'] if c in df.columns), None)
    if not col:
        raise KeyError(get_text("No suitable column found."))  # Translate this error
    groups = df.groupby(col)
    return {crop: {'T_opt': group['Temperature'].mean(),
                   'H_opt': group['Humidity'].mean(),
                   'pH_opt': group['pH'].mean(),
                   'AP_opt': group['Rainfall'].mean()}
            for crop, group in groups}


def calculate_plant_fitness(sensor, optimal, sigmas, weights):
    diff_T = ((sensor['T'] - optimal['T_opt']) ** 2 / (2 * sigmas['sigma_T'] ** 2)) if sensor['T'] is not None else 0
    diff_H = ((sensor['H'] - optimal['H_opt']) ** 2 / (2 * sigmas['sigma_H'] ** 2)) if sensor['H'] is not None else 0
    diff_P = ((sensor['P'] - 1013) ** 2 / (2 * sigmas['sigma_P'] ** 2)) if sensor['P'] is not None else 0
    diff_Tavg = ((sensor['T_avg'] - optimal['T_opt']) ** 2 / (2 * sigmas['sigma_Tavg'] ** 2)) if sensor[
                                                                                                     'T_avg'] is not None else 0
    diff_AP = ((sensor['AP'] - optimal['AP_opt']) ** 2 / (2 * sigmas['sigma_AP'] ** 2)) if sensor[
                                                                                               'AP'] is not None else 0
    diff_pH = ((sensor['pH'] - optimal['pH_opt']) ** 2 / (2 * sigmas['sigma_pH'] ** 2)) if sensor[
                                                                                               'pH'] is not None else 0
    exponent = (weights['w_T'] * diff_T + weights['w_H'] * diff_H + weights['w_P'] * diff_P +
                weights['w_Tavg'] * diff_Tavg + weights['w_AP'] * diff_AP + weights['w_pH'] * diff_pH)
    return exp(-exponent)


def recommend_optimal_crop(sensor, optimal_conditions, sigmas, weights, soil_factors):
    fitness = {}
    for crop, optimal in optimal_conditions.items():
        base = calculate_plant_fitness(sensor, optimal, sigmas, weights)
        factor = soil_factors.get(crop, 1.0)
        fitness[crop] = base * factor
    best = max(fitness, key=fitness.get)
    return best, fitness[best]


def process_crop_recommendation(sensor, selected_plants=None, lang='en'):
    folder = '/Users/michael_z/Downloads/PlantTools/Plant Database'
    df = load_local_crop_datasets(folder)
    if df is None:
        return {"text": get_text("no_crop_data_found", lang), "image_url": None}
    col = next((c for c in ['label', 'crop', 'common_name', 'plant_name'] if c in df.columns), None)
    if col is None:  # Added check here as well
        return {"text": get_text("no_crop_data_found", lang), "image_url": None}
    if selected_plants:
        df = df[df[col].isin(selected_plants)]
    elif 'selected_plants' in session:
        df = df[df[col].isin(session['selected_plants'])]
    if df.empty:
        return {"text": get_text("no_matching_crop_data_found", lang), "image_url": None}
    optimal = compute_optimal_conditions(df)
    sigmas = {'sigma_T': 2.0, 'sigma_H': 10.0, 'sigma_P': 10.0, 'sigma_Tavg': 2.0, 'sigma_AP': 20.0, 'sigma_pH': 0.5}
    weights = {'w_T': 0.35, 'w_H': 0.30, 'w_P': 0.05, 'w_Tavg': 0.15, 'w_AP': 0.10, 'w_pH': 0.05}
    soil_type = session.get("soil_type", get_text("unknown_soil_type", lang))
    soil_col = " ".join(word.capitalize() for word in soil_type.split())
    if soil_col in df.columns:
        soil_factors = df.set_index(col)[soil_col].to_dict()
    else:
        soil_factors = {crop: 1.0 for crop in optimal.keys()}
    best_crop, score = recommend_optimal_crop(sensor, optimal, sigmas, weights, soil_factors)
    df_nums = pd.read_csv(PLANT_DATA_CSV)
    row = df_nums[df_nums['plant_name'] == best_crop]
    image_url = row.iloc[0].get('image_url') if not row.empty else None
    if score < 0.2:
        return {"text": get_text("area_not_fit_for_gardening", lang), "image_url": None}
    return {"text": get_text("recommended_crop", lang).format(best_crop, score), "image_url": image_url}


def run_plantnet_api(image_path):
    API_KEY = ""  # Replace with your actual key
    endpoint = f"https://my-api.plantnet.org/v2/identify/all?api-key={API_KEY}"
    try:
        with open(image_path, 'rb') as img:
            data = {'organs': ['flower']}
            files = [('images', (os.path.basename(image_path), img, 'image/jpeg'))]
            req = requests.Request('POST', url=endpoint, files=files, data=data)
            prepared = req.prepare()
            s = requests.Session()
            response = s.send(prepared, timeout=30)
            result = response.json()
    except Exception as e:
        print(f"PlantNet API error: {e}")
        return None
    if "results" in result and result["results"]:
        top = result["results"][0]
        row = {
            'gbif_id': top.get('gbif', {}).get('id', ''),
            'score': top.get('score', ''),
            'common_names': "; ".join(top.get('species', {}).get('commonNames', [])),
            'scientific_name': top.get('species', {}).get('scientificName', '')
        }
        with open(PLANTNET_RESULT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
        return PLANTNET_RESULT_FILE
    return None


def process_yolo_image(image_path):
    global global_yolo_model
    if global_yolo_model is None:
        global_yolo_model = YOLO(YOLO_MODEL_PATH)
    logger = logging.getLogger("ultralytics")
    logger.setLevel(logging.INFO)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    with open(YOLO_OUTPUT_TEXT_FILE, "w", encoding="utf-8") as out:
        out.write(f"Processing image: {image_path}\n")
        # Ensure results are stored in a variable to be accessed by handler
        results = global_yolo_model(image_path)
        handler.flush()
        out.write(stream.getvalue() + "\n")
        stream.truncate(0)
        stream.seek(0)
    logger.removeHandler(handler)
    return image_path


def extract_probability_info(input_file, output_file=YOLO_PROB_TEXT_FILE):
    pattern = re.compile(r'([A-Za-zäöüßÄÖÜ]+ \d+\.\d+)')
    results = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                matches = pattern.findall(line)
                if matches:
                    results.append(", ".join(matches))
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file}")
        results.append("Error: Input file not found for probability extraction.")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))
    return output_file


def find_max_probabilities(input_file, output_file=YOLO_RESULT_TEXT_FILE):
    pattern = re.compile(r'([A-Za-zäöüßÄÖÜ]+)\s+(\d+\.\d+)')
    results = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, start=1):
                pairs = pattern.findall(line)
                if pairs:
                    max_label, max_value = max(pairs, key=lambda x: float(x[1]))
                    results.append(f"Image {i}: {max_label} {max_value}")
                else:
                    results.append(f"Image {i}: {get_text('No valid data found')}")  # Translate message
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file}")
        results.append("Error: Input file not found for max probability extraction.")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))
    return output_file


def translate_results(input_file, output_file="results_translated.txt"):
    translated = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                header, data = line.split(":", 1)
                parts = header.split()
                image_num = parts[1] if len(parts) >= 2 else get_text("common_name_unknown")
                german_label = data.strip().split()[0]
                if german_label not in translation_mapping:
                    for key in translation_mapping:  # Fallback if direct match fails but partial matches
                        if german_label in key:
                            german_label = key
                            break
                english_label = translation_mapping.get(german_label, german_label)
                # Use get_text for the format string
                translated.append(
                    get_text('yolo_growth_stage').format(get_text('image_num').format(image_num), english_label))
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file}")
        translated.append("Error: Input file not found for translation.")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(translated))
    return output_file


def run_yolo_pipeline(image_path):
    process_yolo_image(image_path)
    extract_probability_info(YOLO_OUTPUT_TEXT_FILE)
    find_max_probabilities(YOLO_PROB_TEXT_FILE)
    return translate_results(YOLO_RESULT_TEXT_FILE)


# ---------------- Flask Setup ----------------

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session management

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------- HTML Templates (Functions) ----------------

# Replaced MAIN_MENU_HTML
def get_main_menu_html(lang_strings):
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{lang_strings['app_title']}</title>
    <style>
        body {{ font-family: sans-serif; text-align: center; }}
        .button-group button {{ margin: 5px; padding: 10px 20px; font-size: 1em; cursor: pointer; }}
        .language-buttons button {{ margin-top: 20px; padding: 8px 15px; font-size: 0.9em; cursor: pointer; }}
    </style>
</head>
<body>
    <h1>{lang_strings['welcome_message']}</h1>
    <p>{lang_strings['select_function']}</p>
    <form action="/set_function" method="post" class="button-group">
        <button type="submit" name="function" value="watering">{lang_strings['btn_watering_recommendations']}</button><br><br>
        <button type="submit" name="function" value="crop">{lang_strings['btn_crop_recommendations']}</button><br><br>
        <button type="submit" name="function" value="plant">{lang_strings['btn_plant_analysis']}</button>
    </form>
    <hr>
    <div class="language-buttons">
        <a href="/set_language/en"><button>{lang_strings['btn_english']}</button></a>
        <a href="/set_language/zh"><button>{lang_strings['btn_chinese']}</button></a>
    </div>
</body>
</html>
"""


# Replaced WATERING_FORM_HTML
def get_watering_form_html(lang_strings):
    return f"""
<!DOCTYPE html>
<html>
<head>
  <title>{lang_strings['watering_title']}</title>
  <style> body {{ font-family: sans-serif; }} label {{ display: inline-block; width: 150px; text-align: right; margin-right: 10px; }} input[type="text"], textarea {{ width: 300px; padding: 5px; border: 1px solid #ccc; border-radius: 4px; }} small {{ display: block; margin-left: 165px; color: #666; }} .form-group {{ margin-bottom: 10px; }} h3 {{ margin-top: 20px; }} </style>
</head>
<body>
  <h1>{lang_strings['watering_title']}</h1>
  <h2>{lang_strings['watering_step1_title']}</h2>
  <form method="post" action="/watering/calculate">
    <div class="form-group">
      <label for="latitude">{lang_strings['label_latitude']}</label>
      <input type="text" id="latitude" name="latitude" required pattern="^-?([1-8]?\\d(\\.\\d+)?|90(\\.0+)?)$" title="{lang_strings['latitude_pattern_title']}">
    </div>
    <div class="form-group">
      <label for="longitude">{lang_strings['label_longitude']}</label>
      <input type="text" id="longitude" name="longitude" required pattern="^-?((1[0-7]|[1-9])?\\d(\\.\\d+)?|180(\\.0+)?)$" title="{lang_strings['longitude_pattern_title']}">
    </div>
    <hr>
    <h3>{lang_strings['pots_data_heading']}</h3>
    <p>{lang_strings['pots_data_desc1']}<br>
       {lang_strings['pots_data_desc2_c']}<br>
       {lang_strings['pots_data_desc2_r']}<br>
       {lang_strings['pots_data_example_title']}<br>
       {lang_strings['pots_data_example_c']}<br>
       {lang_strings['pots_data_example_r']}
    </p>
    <textarea name="pots_data" rows="6" cols="50" placeholder="c, 15, Tomato&#10;r, 30, 20, Basil"></textarea>
    <h3>{lang_strings['yard_data_heading']}</h3>
    <p>{lang_strings['yard_data_desc1']}<br>
       {lang_strings['yard_data_example_title']}<br>
       {lang_strings['yard_data_example']}
    </p>
    <textarea name="yard_data" rows="3" cols="50" placeholder="5, 10"></textarea>
    <p>
       <label for="yard_plants">{lang_strings['label_yard_plants']}</label>
       <input type="text" id="yard_plants" name="yard_plants" size="40" placeholder="{lang_strings['yard_plants_placeholder']}">
       <br><small>{lang_strings['yard_plants_small_text']}</small>
    </p>
    <h3>{lang_strings['yard_plant_areas_heading']}</h3>
    <p>{lang_strings['yard_plant_areas_desc']}</p>
    <textarea name="yard_plant_areas" rows="5" cols="50" placeholder="Roses, 2.5&#10;Lavender, 1.2"></textarea>
    <hr>
    <input type="submit" value="{lang_strings['btn_calculate_daily_needs']}">
  </form>
  <br><a href="/">{lang_strings['btn_back_to_main_menu']}</a>
</body>
</html>
"""


# Replaced WATERING_SELECT_DAYS_HTML
def get_watering_select_days_html(lang_strings, soil_type):
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{lang_strings['watering_select_days_title']}</title>
    <style>
        body {{ font-family: sans-serif; }}
        table {{ border-collapse: collapse; width: 95%; margin: 20px auto; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; font-size: 0.9em; }}
        td {{ font-size: 0.9em; }}
        .plant-header {{ background-color: #e9f5ff; font-weight: bold; text-align: left; padding-left: 15px; }}
        .guideline {{ font-style: italic; color: #555; font-size: 0.85em; }}
        .amount {{ font-weight: bold; }}
        .date-header {{ writing-mode: vertical-rl; text-orientation: mixed; white-space: nowrap; }}
        input[type=checkbox] {{ transform: scale(1.2); margin: 0; }}
        .info {{ background-color: #f9f9f9; padding: 10px; border: 1px solid #eee; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>{lang_strings['watering_title']}</h1>
    <h2>{lang_strings['watering_step2_title']}</h2>
    <div class="info">
        <p>{lang_strings['info_daily_needs_calculated'].format(soil_type)}</p>
        <p>{lang_strings['info_yard_plant_individual']}</p>
        <p>{lang_strings['info_check_boxes_adjust']}</p>
    </div>
    <form method="post" action="/watering/finalize">
        <table>
            <thead>
                <tr>
                    <th>{lang_strings['table_header_area_plant']}</th>
                    <th>{lang_strings['table_header_frequency_guideline']}</th>
                    {{% for d in dates %}}
                    <th class="date-header">{{{{ d.text }}}}</th>
                    {{% endfor %}}
                </tr>
            </thead>
            <tbody>
                {{% if plan.pots %}}
<tr><td colspan="{{{{ 2 + dates|length }}}}" style="background-color: #eee; font-weight:bold;">{lang_strings['table_header_potted_plants']}</td></tr>
{{% for pot in plan.pots %}}
<tr>
    <td class="plant-header">{lang_strings['Pot']} {{{{ pot.id }}}} ({{{{ pot.plant.title() }}}})</td>
    <td class="guideline">{{{{ pot.guideline }}}}</td>
    {{% for daily in pot.daily_amounts %}}
    <td>
        <span class="amount">{{{{ "%.2f L"|format(daily.amount) }}}}</span><br>
        <input type="checkbox" name="select_pot_{{{{ pot.id }}}}_{{{{ daily.date }}}}" value="selected">
    </td>
    {{% endfor %}}
</tr>
{{% endfor %}}
{{% endif %}}
{{% if plan.yard %}}
<tr><td colspan="{{{{ 2 + dates|length }}}}" style="background-color: #eee; font-weight:bold;">{lang_strings['table_header_yard_plants']}</td></tr>
{{% for yard in plan.yard %}}
<tr>
    <td class="plant-header">{lang_strings['Yard']} - {{{{ yard.plant }}}}</td>
    <td class="guideline">{{{{ yard.guideline }}}}</td>
    {{% for daily in yard.daily_amounts %}}
    <td>
        <span class="amount">{{{{ "%.2f L"|format(daily.amount) }}}}</span><br>
        <input type="checkbox" name="select_yard_{{{{ yard.id }}}}_{{{{ daily.date }}}}" value="selected">
    </td>
    {{% endfor %}}
</tr>
{{% endfor %}}
{{% endif %}}
            </tbody>
        </table>
        <br>
        <div style="text-align:center;">
             <input type="submit" value="{lang_strings['btn_generate_final_plan']}" style="padding: 10px 20px; font-size: 1.1em;">
             <br><br>
             <a href="/watering">{lang_strings['btn_back_to_inputs']}</a>
             <a href="/">{lang_strings['btn_back_to_main_menu']}</a>
        </div>
    </form>
    <br>
</body>
</html>
"""


# Replaced WATERING_FINAL_PLAN_HTML
def get_watering_final_plan_html(lang_strings):
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{lang_strings['final_plan_title']}</title>
    <style>
        body {{ font-family: sans-serif; }}
        table {{ border-collapse: collapse; width: 80%; margin: 20px auto; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
        .message {{ text-align: center; padding: 15px; background-color: #e7f7e7; border: 1px solid #c7e7c7; margin: 20px auto; width: 80%; }}
        .error {{ background-color: #f7e7e7; border-color: #e7c7c7; }}
    </style>
</head>
<body>
    <h1>{lang_strings['final_plan_title']}</h1>
    <p style="text-align: center;">{lang_strings['plan_selected_days_only']}</p>
    {{% if csv_message %}}
        <div class="message {{% if 'Error' in csv_message %}}error{{% endif %}}">
            {{{{ csv_message }}}}
        </div>
    {{% endif %}}
    {{% if plan_rows %}}
    <table>
        <thead>
            <tr>
                <th>{lang_strings['table_header_area_type']}</th>
                <th>{lang_strings['table_header_identifier']}</th>
                <th>{lang_strings['table_header_plant_name']}</th>
                <th>{lang_strings['table_header_watering_date']}</th>
                <th>{lang_strings['table_header_water_amount']}</th>
                <th>{lang_strings['table_header_frequency_guideline']}</th>
            </tr>
        </thead>
        <tbody>
            {{% for row in plan_rows %}}
            <tr>
                <td>{{{{ row.type }}}}</td>
                <td>{{{{ row.id }}}}</td>
                <td>{{{{ row.plant }}}}</td>
                <td>{{{{ row.date }}}}</td>
                <td>{{{{ row.amount }}}}</td>
                <td><small>{{{{ row.guideline }}}}</small></td>
            </tr>
            {{% endfor %}}
        </tbody>
    </table>
    {{% else %}}
        <p style="text-align: center;">{lang_strings['no_watering_days_selected']}</p>
    {{% endif %}}
    <p style="text-align: center;">
        <a href="/watering/select_days" style="margin-right: 20px;">{lang_strings['btn_back_to_day_selection']}</a>
        <a href="/">{lang_strings['btn_back_to_main_menu']}</a>
    </p>
</body>
</html>
"""


# Replaced CROP_SEARCH_HTML
def get_crop_search_html(lang_strings):
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{lang_strings['crop_search_title']}</title>
</head>
<body>
    <h1>{lang_strings['crop_search_title']}</h1>
    <form method="post" action="/crop_search">
        <label for="query">{lang_strings['label_search_plants']}</label>
        <input type="text" name="query" id="query">
        <button type="submit">{lang_strings['btn_search']}</button>
    </form>
    {{% if results %}}
    <form method="post" action="/select_plants_from_search">
        <h2>{lang_strings['heading_select_plants']}</h2>
        {{% for plant in results %}}
            <input type="checkbox" name="plants" value="{{{{ plant }}}}"> {{{{ plant }}}} <br>
        {{% endfor %}}
        <button type="submit">{lang_strings['btn_submit_selection']}</button>
    </form>
    {{% endif %}}
    <br><a href="/">{lang_strings['btn_back_to_main_menu']}</a>
</body>
</html>
"""


# Replaced DATA_CHOICE_HTML
def get_data_choice_html(lang_strings):
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{lang_strings['data_choice_title']}</title>
</head>
<body>
    <h1>{lang_strings['heading_choose_data_input_method']}</h1>
    <form action="/location_form" method="get">
        <button type="submit">{lang_strings['btn_use_location_data']}</button>
    </form>
    <form action="/sensor_form" method="get">
        <button type="submit">{lang_strings['btn_enter_manual_data']}</button>
    </form>
    <br><a href="/">{lang_strings['btn_back_to_main_menu']}</a>
</body>
</html>
"""


# Replaced LOCATION_FORM_HTML
def get_location_form_html(lang_strings):
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{lang_strings['location_form_title']}</title>
</head>
<body>
    <h1>{lang_strings['location_form_title']}</h1>
    <form action="/location_form" method="post">
        <label for="lat">{lang_strings['label_latitude']}</label> <input type="text" id="lat" name="lat"><br>
        <label for="lon">{lang_strings['label_longitude']}</label> <input type="text" id="lon" name="lon"><br>
        <label for="month">{lang_strings['label_month']}</label> <input type="text" id="month" name="month"><br>
        <label for="year">{lang_strings['label_year']}</label> <input type="text" id="year" name="year"><br>
        <button type="submit">{lang_strings['btn_submit']}</button>
    </form>
    <br><a href="/">{lang_strings['btn_back_to_main_menu']}</a>
</body>
</html>
"""


# Replaced SENSOR_FORM_HTML
def get_sensor_form_html(lang_strings):
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{lang_strings['sensor_form_title']}</title>
</head>
<body>
    <h1>{lang_strings['sensor_form_title']}</h1>
    <form action="/sensor_form" method="post">
        <label for="T">{lang_strings['label_inst_temp']}</label> <input type="text" id="T" name="T" value="{{{{ prefilled.get('T', '') }}}}"><br>
        <label for="H">{lang_strings['label_ambient_humidity']}</label> <input type="text" id="H" name="H" value="{{{{ prefilled.get('H', '') }}}}"><br>
        <label for="P">{lang_strings['label_atm_pressure']}</label> <input type="text" id="P" name="P" value="{{{{ prefilled.get('P', '') }}}}"><br>
        <label for="T_avg">{lang_strings['label_monthly_avg_temp']}</label> <input type="text" id="T_avg" name="T_avg" value="{{{{ prefilled.get('T_avg', '') }}}}"><br>
        <label for="AP">{lang_strings['label_total_precipitation']}</label> <input type="text" id="AP" name="AP" value="{{{{ prefilled.get('AP', '') }}}}"><br>
        <label for="pH">{lang_strings['label_soil_ph']}</label> <input type="text" id="pH" name="pH" value="{{{{ prefilled.get('pH', '') }}}}"><br>
        <button type="submit">{lang_strings['btn_submit']}</button>
    </form>
    <br><a href="/">{lang_strings['btn_back_to_main_menu']}</a>
</body>
</html>
"""


# Replaced CROP_RESULT_HTML
def get_crop_result_html(lang_strings):
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{lang_strings['recommendation_title']}</title>
</head>
<body>
    <h1>{lang_strings['heading_recommendation']}</h1>
    <p>{{{{ recommendation.text }}}}</p>
    {{% if recommendation.image_url %}}
    <img src="{{{{ recommendation.image_url }}}}" alt="{lang_strings['recommended_crop'].split(':')[0].strip()}" style="max-width: 600px;">
    {{% endif %}}
    <br><a href="/">{lang_strings['btn_back_to_main_menu']}</a>
</body>
</html>
"""


# Replaced PLANT_FORM_HTML
def get_plant_form_html(lang_strings):
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{lang_strings['plant_analysis_tool_title']}</title>
</head>
<body>
    <h1>{lang_strings['plant_analysis_tool_title']}</h1>
    <form method="POST" action="/plant" enctype="multipart/form-data">
        <p>{lang_strings['select_image']} <input type="file" name="image_file" accept="image/*" /></p>
        <button type="submit">{lang_strings['btn_upload_analyze']}</button>
    </form>
    <br><a href="/">{lang_strings['btn_back_to_main_menu']}</a>
</body>
</html>
"""


def get_plant_result_html(lang_strings):
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{lang_strings['plant_result_title']}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                padding: 20px;
                background-color: #f8f9fa;
            }}
            .result-container {{
                max-width: 800px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            .result-header {{
                margin-bottom: 30px;
                text-align: center;
            }}
            .result-details {{
                margin-bottom: 20px;
            }}
            .back-link {{
                display: block;
                text-align: center;
                margin-top: 30px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="result-container">
                <div class="result-header">
                    <h1>{lang_strings['plant_result_heading']}</h1>
                </div>
                <div class="result-details">
                    <h3>{lang_strings['plant_result_species']}</h3>
                    <p>{{{{ result.species }}}}</p>

                    <h3>{lang_strings['plant_result_genus']}</h3>
                    <p>{{{{ result.genus }}}}</p>

                    <h3>{lang_strings['plant_result_family']}</h3>
                    <p>{{{{ result.family }}}}</p>

                    <h3>{lang_strings['plant_result_confidence']}</h3>
                    <p>{{{{ "%.2f"|format(result.confidence * 100) }}}}%</p>

                    <h3>{lang_strings['detected_disease']}</h3>
                    <p>{{{{ result.disease if result.disease else lang_strings['no_disease_detected'] }}}}</p>

                    <h3>{lang_strings['recommended_remedy']}</h3>
                    <p>{{{{ result.remedy if result.remedy else lang_strings['no_remedy_provided'] }}}}</p>
                </div>
                <a href="/" class="btn btn-primary back-link">{lang_strings['btn_back_to_main_menu']}</a>
            </div>
        </div>
    </body>
    </html>
    """


# ---------------- Flask Routes ----------------
# ------------------------- Language Selection Route -------------------------
# ---------------- Flask Routes ----------------
# ------------------------- Language Selection Route -------------------------
@app.route('/set_language/<lang_code>')
def set_language(lang_code):
    """Sets the language in the session and redirects back or to main menu."""
    if lang_code in ['en', 'zh']:
        session['lang'] = lang_code
    referrer = request.referrer
    # Redirect back to the page the user was on, avoiding redirect loops to /set_language itself
    if referrer and url_for('set_language', lang_code=lang_code) not in referrer:
        return redirect(referrer)
    return redirect(url_for('main_menu'))  # Fallback to main menu


# ------------------------- Main Routes -------------------------

@app.route('/')
def main_menu():
    # Set default language if not already set in session
    if 'lang' not in session:
        session['lang'] = 'en'
    # Create a dictionary of translated strings for the main menu
    current_lang_strings = {
        'app_title': get_text('app_title', session['lang']),
        'welcome_message': get_text('welcome_message', session['lang']),
        'select_function': get_text('select_function', session['lang']),
        'btn_watering_recommendations': get_text('btn_watering_recommendations', session['lang']),
        'btn_crop_recommendations': get_text('btn_crop_recommendations', session['lang']),
        'btn_plant_analysis': get_text('btn_plant_analysis', session['lang']),
        'btn_back_to_main_menu': get_text('btn_back_to_main_menu', session['lang']),
        'btn_english': get_text('btn_english', session['lang']),
        'btn_chinese': get_text('btn_chinese', session['lang'])
    }
    return render_template_string(get_main_menu_html(current_lang_strings))


@app.route('/set_function', methods=['POST'])
def set_function():
    function = request.form.get('function')
    if not function:
        return get_text('error_no_function_selected'), 400
    session['function'] = function
    if function == 'watering':
        return redirect('/watering')
    elif function == 'crop':
        return redirect('/crop')
    elif function == 'plant':
        return redirect('/plant')
    return redirect('/')


# ----- Watering Routes -----

@app.route('/watering', methods=['GET'])
def watering_input():
    # Create a dictionary of translated strings for the watering form
    current_lang_strings = {
        'watering_title': get_text('watering_title', session['lang']),
        'watering_step1_title': get_text('watering_step1_title', session['lang']),
        'label_latitude': get_text('label_latitude', session['lang']),
        'label_longitude': get_text('label_longitude', session['lang']),
        'latitude_pattern_title': get_text('latitude_pattern_title', session['lang']),
        'longitude_pattern_title': get_text('longitude_pattern_title', session['lang']),
        'pots_data_heading': get_text('pots_data_heading', session['lang']),
        'pots_data_desc1': get_text('pots_data_desc1', session['lang']),
        'pots_data_desc2_c': get_text('pots_data_desc2_c', session['lang']),
        'pots_data_desc2_r': get_text('pots_data_desc2_r', session['lang']),
        'pots_data_example_title': get_text('pots_data_example_title', session['lang']),
        'pots_data_example_c': get_text('pots_data_example_c', session['lang']),
        'pots_data_example_r': get_text('pots_data_example_r', session['lang']),
        'yard_data_heading': get_text('yard_data_heading', session['lang']),
        'yard_data_desc1': get_text('yard_data_desc1', session['lang']),
        'yard_data_example_title': get_text('yard_data_example_title', session['lang']),
        'yard_data_example': get_text('yard_data_example', session['lang']),
        'label_yard_plants': get_text('label_yard_plants', session['lang']),
        'yard_plants_placeholder': get_text('yard_plants_placeholder', session['lang']),
        'yard_plants_small_text': get_text('yard_plants_small_text', session['lang']),
        'yard_plant_areas_heading': get_text('yard_plant_areas_heading', session['lang']),
        'yard_plant_areas_desc': get_text('yard_plant_areas_desc', session['lang']),
        'yard_plant_areas_placeholder': get_text('yard_plant_areas_placeholder', session['lang']),
        'btn_calculate_daily_needs': get_text('btn_calculate_daily_needs', session['lang']),
        'btn_back_to_main_menu': get_text('btn_back_to_main_menu', session['lang'])
    }
    return render_template_string(get_watering_form_html(current_lang_strings))


@app.route('/watering/calculate', methods=['POST'])
def watering_calculate():
    # Create a dictionary of translated strings for the watering calculation
    current_lang_strings = {
        'error_fetching_weather_data': get_text('error_fetching_weather_data'),
        'error_retrieving_soil_values': get_text('error_retrieving_soil_values'),
        'error_no_et0_data': get_text('error_no_et0_data'),
        'invalid_input': get_text('invalid_input')
    }
    try:
        lat = float(request.form.get('latitude'))
        lon = float(request.form.get('longitude'))
    except (ValueError, TypeError):
        return get_text('invalid_input'), 400

    pots_text = request.form.get('pots_data', '').strip()
    yard_text = request.form.get('yard_data', '').strip()
    yard_plants_text = request.form.get('yard_plants', '').strip()
    yard_plant_areas_text = request.form.get('yard_plant_areas', '').strip()

    area_data["pots"] = parse_pot_data(pots_text) if pots_text else []

    yard_info = parse_yard_area_data(yard_text) if yard_text else {"area": 0}
    if yard_plant_areas_text:
        plants_details = parse_yard_plant_area_data(yard_plant_areas_text)
        total_manual_area = sum(item["area"] for item in plants_details)
        ratio = float(yard_info["area"]) / total_manual_area if total_manual_area > 0 else 1
        yard_info["plants_details"] = []
        for item in plants_details:
            effective_area = item["area"] * ratio
            yard_info["plants_details"].append({"plant": item["plant"], "effective_area": effective_area})
    else:
        yard_info["plants_details"] = None
        yard_info["plants"] = parse_yard_plant_names(yard_plants_text)
    area_data["yard"] = yard_info

    forecast = get_open_meteo_forecast(lat, lon)
    if not forecast:
        return get_text('error_fetching_weather_data'), 500

    hourly = forecast.get("hourly", {})
    write_hourly_data_to_csv(
        hourly.get("time", []),
        hourly.get("shortwave_radiation", []),
        hourly.get("windspeed_10m", []),
        [round(convert_wind_speed_to_two_meters(v), 2) for v in hourly.get("windspeed_10m", [])],
        hourly.get("cloudcover", []),
        hourly.get("relativehumidity_2m", []),
        [round(calculate_dew_point_temperature(T, RH), 2)
         if calculate_dew_point_temperature(T, RH) is not None else "N/A"
         for T, RH in zip(hourly.get("temperature_2m", []), hourly.get("relativehumidity_2m", []))]
    )
    daily = forecast.get("daily", {})
    write_daily_data_to_csv(
        daily.get("time", []),
        daily.get("temperature_2m_max", []),
        daily.get("temperature_2m_min", [])
    )

    try:
        sand, silt, clay = get_soil_values(DEFAULT_TIFF_PATH, lon, lat)
        sand_p = sand * 100.0
        silt_p = silt * 100.0
        clay_p = clay * 100.0
        soil_type = determine_soil_classification(sand_p, silt_p, clay_p)
        write_soil_to_csv(lon, lat, sand_p, silt_p, clay_p, soil_type)
        session["soil_type"] = soil_type
    except Exception as e:
        return current_lang_strings['error_retrieving_soil_values'].format(e), 500

    export_total_area_to_csv()

    temp_df = pd.read_csv(TEMPERATURE_CSV_FILE)
    temp_df["Date"] = temp_df["Date"].astype(str)
    swc_df = pd.read_csv(SOIL_WATER_CONTENT_CSV_FILE)
    swc_df['Date'] = swc_df['Time'].str.split('T').str[0]
    merged_df = pd.merge(swc_df, temp_df, on="Date", how="left")

    def calc_ET0(row):
        try:
            T_max = row["Max Temperature (°C)"]
            T_min = row["Min Temperature (°C)"]
            RH = row["Relative Humidity (%)"]
            if RH <= 1:
                RH *= 100
            u2 = row["Estimated Wind Speed 2m (m/s)"]
            Rs_MJ = row["Solar Radiation (W/m²)"] * 0.0036
            return compute_reference_evapotranspiration(T_max, T_min, RH, u2, Rs_MJ)
        except Exception:
            return None

    merged_df['ET0'] = merged_df.apply(calc_ET0, axis=1)
    daily_ET0_df = merged_df.groupby("Date")["ET0"].sum().reset_index()
    if daily_ET0_df.empty:
        return get_text('error_no_et0_data'), 400
    daily_ET0_df.to_csv(EVAPOTRANSPIRATION_CSV_FILE, index=False)
    rep_ET0 = daily_ET0_df.iloc[0]["ET0"]  # Using first day's ET0 as a representative for now if all are similar
    multiplier = soil_multiplier_mapping.get(soil_type, 1.0)
    plan = {"pots": [], "yard": []}

    for pot in area_data["pots"]:
        plant_name = str(pot.get("plant")).lower()
        guideline, min_freq, max_freq = calculate_watering_frequency_for_plant(plant_name, multiplier)
        water_amount = watering_recommendation_for_pot(soil_type, rep_ET0, float(pot.get("area", 0)))["amount"]
        pot.update({
            "water_amount": water_amount,
            "frequency_min": min_freq,
            "frequency_max": max_freq,
            "guideline": guideline
        })
        plan["pots"].append(pot)

    # --- Modified Yard Processing: Generate plant-specific yard recommendations ---
    if area_data["yard"].get("plants_details"):
        for plant_info in area_data["yard"]["plants_details"]:
            plant = plant_info["plant"]
            effective_area = plant_info["effective_area"]
            guideline, min_freq, max_freq = calculate_watering_frequency_for_plant(str(plant).lower(), multiplier)
            water_amount = watering_recommendation_for_yard(soil_type, rep_ET0, effective_area)
            plan["yard"].append({
                "id": f"yard_{plant}",
                "plant": plant,
                "area": effective_area,
                "water_amount": water_amount,
                "frequency_min": min_freq,
                "frequency_max": max_freq,
                "guideline": guideline
            })
    elif area_data["yard"].get("plants"):
        yard_plants = area_data["yard"]["plants"]
        total_area = float(area_data["yard"]["area"])
        individual_area = total_area / len(yard_plants) if len(yard_plants) > 0 else total_area  # Handle empty list
        for plant in yard_plants:
            guideline, min_freq, max_freq = calculate_watering_frequency_for_plant(str(plant).lower(), multiplier)
            water_amount = watering_recommendation_for_yard(soil_type, rep_ET0, individual_area)
            plan["yard"].append({
                "id": f"yard_{plant}",
                "plant": plant,
                "area": individual_area,
                "water_amount": water_amount,
                "frequency_min": min_freq,
                "frequency_max": max_freq,
                "guideline": guideline
            })

    session["plan"] = plan
    return redirect("/watering/select_days")


@app.route('/watering/select_days', methods=['GET'])
def watering_select_days():
    # Create a dictionary of translated strings for the watering select days page
    current_lang_strings = {
        'watering_title': get_text('watering_title'),
        'watering_select_days_title': get_text('watering_select_days_title'),
        'watering_step2_title': get_text('watering_step2_title'),
        'info_daily_needs_calculated': get_text('info_daily_needs_calculated'),
        'info_yard_plant_individual': get_text('info_yard_plant_individual'),
        'info_check_boxes_adjust': get_text('info_check_boxes_adjust'),
        'table_header_area_plant': get_text('table_header_area_plant'),
        'table_header_frequency_guideline': get_text('table_header_frequency_guideline'),
        'table_header_potted_plants': get_text('table_header_potted_plants'),
        'table_header_yard_plants': get_text('table_header_yard_plants'),
        'btn_generate_final_plan': get_text('btn_generate_final_plan'),
        'btn_back_to_inputs': get_text('btn_back_to_inputs'),
        'Pot': get_text('Pot'),
        'Yard': get_text('Yard'),
        'btn_back_to_main_menu': get_text('btn_back_to_main_menu')
    }
    plan = session.get("plan")
    if not plan:
        return redirect("/watering")
    try:
        daily_ET0_df = pd.read_csv(EVAPOTRANSPIRATION_CSV_FILE)
        dates_list = daily_ET0_df["Date"].tolist()
        et0_mapping = dict(zip(daily_ET0_df["Date"], daily_ET0_df["ET0"]))
    except Exception:
        # Fallback if CSV not found/empty or for testing (last 7 days from today)
        dates_list = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
        et0_mapping = {}

    date_objects = []
    for d in dates_list:
        try:
            dt = datetime.strptime(d, "%Y-%m-%d")
            if session.get('lang', 'en') == 'zh':
                weekday_map = {
                    'Monday': '星期一', 'Tuesday': '星期二', 'Wednesday': '星期三',
                    'Thursday': '星期四', 'Friday': '星期五', 'Saturday': '星期六',
                    'Sunday': '星期日'
                }
                weekday = weekday_map.get(dt.strftime("%A"), dt.strftime("%A"))
            else:
                weekday = dt.strftime("%A")
            date_objects.append({
                "value": d,
                "display": f"{weekday}, {dt.strftime('%Y-%m-%d')}"
            })
        except ValueError:
            continue

    # Add daily amounts to each pot and yard item
    for group in ["pots", "yard"]:
        for item in plan.get(group, []):
            daily_amounts = []
            for date_obj in date_objects:
                et0_val = et0_mapping.get(date_obj["value"], None)
                if et0_val is None:
                    water_amt = item.get("water_amount", 0)
                else:
                    if group == "pots":
                        water_amt = \
                        watering_recommendation_for_pot(session.get("soil_type", get_text("unknown_soil_type")),
                                                        et0_val,
                                                        float(item.get("area", 0)))["amount"]
                    else:
                        water_amt = watering_recommendation_for_yard(
                            session.get("soil_type", get_text("unknown_soil_type")),
                            et0_val,
                            float(item.get("area", 0)))
                daily_amounts.append({"date": date_obj["value"], "amount": float(water_amt)})
            item["daily_amounts"] = daily_amounts

    session["plan"] = plan
    return render_template_string(
        get_watering_select_days_html(current_lang_strings, session.get("soil_type", get_text("unknown_soil_type"))),
        plan=plan, dates=date_objects, soil_type=session.get("soil_type", get_text("unknown_soil_type")))


@app.route('/watering/finalize', methods=['POST'])
def watering_finalize():
    # Create a dictionary of translated strings for the watering finalize page
    current_lang_strings = {
        'final_plan_title': get_text('final_plan_title'),
        'csv_message_success': get_text('csv_message_success'),
        'csv_message_error': get_text('csv_message_error'),
        'btn_back_to_main_menu': get_text('btn_back_to_main_menu'),
        'btn_back_to_day_selection': get_text('btn_back_to_day_selection'),
        'Pot': get_text('Pot'),
        'Yard': get_text('Yard'),
        'plan_selected_days_only': get_text('plan_selected_days_only'),
        'no_watering_days_selected': get_text('no_watering_days_selected'),
        'table_header_area_type': get_text('table_header_area_type'),
        'table_header_identifier': get_text('table_header_identifier'),
        'table_header_plant_name': get_text('table_header_plant_name'),
        'table_header_watering_date': get_text('table_header_watering_date'),
        'table_header_water_amount': get_text('table_header_water_amount'),
        'table_header_frequency_guideline': get_text('table_header_frequency_guideline')
    }

    plan = session.get("plan")
    if not plan:
        return redirect("/watering")

    # Format plan data for the template
    plan_rows = []
    for group in ["pots", "yard"]:
        for item in plan.get(group, []):
            for daily_amount in item.get("daily_amounts", []):
                plan_rows.append({
                    "type": current_lang_strings['Pot'] if group == "pots" else current_lang_strings['Yard'],
                    "id": f"{current_lang_strings['Pot']} {item['id']}" if group == "pots" else f"{current_lang_strings['Yard']} - {item['plant']}",
                    "plant": item.get("plant", "").title(),
                    "date": daily_amount["date"],
                    "amount": daily_amount["amount"],
                    "guideline": item.get("guideline", "")
                })

    # Generate CSV rows
    csv_rows = []
    for row in plan_rows:
        csv_rows.append({
            "Date": row["date"],
            "Plant": row["plant"],
            "Area": row.get("area", 0),
            "Water Amount": row["amount"]
        })

    # Write to CSV
    try:
        df = pd.DataFrame(csv_rows)
        df.to_csv(WATERING_PLAN_CSV_FILE, index=False)
        csv_message = current_lang_strings['csv_message_success']
    except Exception as e:
        csv_message = current_lang_strings['csv_message_error']

    return render_template_string(
        get_watering_final_plan_html(current_lang_strings),
        plan_rows=plan_rows,
        csv_message=csv_message)


# ----- Crop and Data Input Routes -----

@app.route('/crop')
def crop_redirect():
    return redirect('/crop_search')


@app.route('/crop_search', methods=['GET', 'POST'])
def crop_search():
    # Create a dictionary of translated strings for the crop search page
    current_lang_strings = {
        'crop_search_title': get_text('crop_search_title'),
        'label_search_plants': get_text('label_search_plants'),
        'btn_search': get_text('btn_search'),
        'heading_select_plants': get_text('heading_select_plants'),
        'btn_submit_selection': get_text('btn_submit_selection'),
        'no_crop_data_found': get_text('no_crop_data_found'),
        'no_matching_crop_data_found': get_text('no_matching_crop_data_found'),
        'please_select_at_least_one_plant': get_text('please_select_at_least_one_plant'),
        'btn_back_to_main_menu': get_text('btn_back_to_main_menu')
    }

    results = None
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query:
            folder = '/Users/michael_z/Downloads/PlantTools/Plant Database'
            df = load_local_crop_datasets(folder)
            if df is None:
                return get_text('no_crop_data_found'), 500
            col = next((c for c in ['label', 'crop', 'common_name', 'plant_name'] if c in df.columns), None)
            if col:  # Check if col is found
                results = sorted(df[col][df[col].str.contains(query, case=False, na=False)].unique().tolist())
            else:  # If no suitable column is found, handle the error
                return get_text('no_crop_data_found'), 500

    return render_template_string(get_crop_search_html(current_lang_strings), results=results)


@app.route('/select_plants_from_search', methods=['POST'])
def select_plants_from_search():
    selected = request.form.getlist('plants')
    if not selected:
        return get_text('please_select_at_least_one_plant'), 400
    session['selected_plants'] = selected
    return redirect('/data_choice')


@app.route('/data_choice')
def data_choice():
    # Create a dictionary of translated strings for the data choice page
    current_lang_strings = {
        'data_choice_title': get_text('data_choice_title', session['lang']),
        'heading_choose_data_input_method': get_text('heading_choose_data_input_method', session['lang']),
        'btn_use_location_data': get_text('btn_use_location_data', session['lang']),
        'btn_enter_manual_data': get_text('btn_enter_manual_data', session['lang']),
        'btn_back_to_main_menu': get_text('btn_back_to_main_menu', session['lang'])
    }
    return render_template_string(get_data_choice_html(current_lang_strings))


@app.route('/location_form', methods=['GET', 'POST'])
def location_form():
    # Create a dictionary of translated strings for the location form
    current_lang_strings = {
        'location_form_title': get_text('location_form_title', session['lang']),
        'label_latitude': get_text('label_latitude', session['lang']),
        'label_longitude': get_text('label_longitude', session['lang']),
        'label_month': get_text('label_month', session['lang']),
        'label_year': get_text('label_year', session['lang']),
        'btn_submit': get_text('btn_submit', session['lang']),
        'btn_back_to_main_menu': get_text('btn_back_to_main_menu', session['lang'])
    }

    if request.method == 'POST':
        try:
            lat = float(request.form['lat'])
            lon = float(request.form['lon'])
            month = int(request.form['month'])
            year = int(request.form['year'])
        except ValueError:
            return get_text('invalid_input', session['lang']), 400
        report = get_location_info(lat, lon, month, year)
        session['sensor_data'] = extract_sensor_data(report)
        session['soil_type'] = report.get("Soil Classification", get_text("unknown_soil_type", session['lang']))
        return redirect('/sensor_form')
    return render_template_string(get_location_form_html(current_lang_strings))


@app.route('/sensor_form', methods=['GET', 'POST'])
def sensor_form():
    # Create a dictionary of translated strings for the sensor form and crop result
    current_lang_strings = {
        'sensor_form_title': get_text('sensor_form_title', session['lang']),
        'label_inst_temp': get_text('label_inst_temp', session['lang']),
        'label_ambient_humidity': get_text('label_ambient_humidity', session['lang']),
        'label_atm_pressure': get_text('label_atm_pressure', session['lang']),
        'label_monthly_avg_temp': get_text('label_monthly_avg_temp', session['lang']),
        'label_total_precipitation': get_text('label_total_precipitation', session['lang']),
        'label_soil_ph': get_text('label_soil_ph', session['lang']),
        'btn_submit': get_text('btn_submit', session['lang']),
        'btn_back_to_main_menu': get_text('btn_back_to_main_menu', session['lang']),
        # For crop result html
        'recommendation_title': get_text('recommendation_title', session['lang']),
        'heading_recommendation': get_text('heading_recommendation', session['lang']),
        'recommended_crop': get_text('recommended_crop', session['lang']),
        'area_not_fit_for_gardening': get_text('area_not_fit_for_gardening', session['lang'])
    }

    if request.method == 'POST':
        try:
            sensor = {key: float(request.form[key]) for key in ['T', 'H', 'P', 'T_avg', 'AP', 'pH']}
        except ValueError:
            return get_text('invalid_input', session['lang']), 400
        recommendation = process_crop_recommendation(sensor)
        return render_template_string(get_crop_result_html(current_lang_strings), recommendation=recommendation)
    prefilled = session.get('sensor_data', {})
    return render_template_string(get_sensor_form_html(current_lang_strings), prefilled=prefilled)


# ----- Plant Analysis Route -----

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/plant', methods=['GET', 'POST'])
def plant_route():
    # Get language from query parameter or session, defaulting to English
    lang = request.args.get('lang') or session.get('lang', 'en')

    # Create a dictionary of translated strings for the plant form
    current_lang_strings = {
        'plant_form_title': get_text('plant_form_title', lang),
        'plant_analysis_tool_title': get_text('plant_analysis_tool_title', lang),
        'label_upload_image': get_text('label_upload_image', lang),
        'select_image': get_text('select_image', lang),
        'btn_submit': get_text('btn_submit', lang),
        'btn_upload_analyze': get_text('btn_upload_analyze', lang),
        'btn_back_to_main_menu': get_text('btn_back_to_main_menu', lang),
        'error_no_file': get_text('error_no_file', lang),
        'error_invalid_file': get_text('error_invalid_file', lang),
        'error_processing': get_text('error_processing', lang),
        'error_no_plants_found': get_text('error_no_plants_found', lang),
        'error_unknown': get_text('error_unknown', lang),
        'plant_result_title': get_text('plant_result_title', lang),
        'plant_result_heading': get_text('plant_result_heading', lang),
        'plant_result_species': get_text('plant_result_species', lang),
        'plant_result_genus': get_text('plant_result_genus', lang),
        'plant_result_family': get_text('plant_result_family', lang),
        'plant_result_confidence': get_text('plant_result_confidence', lang),
        'detected_disease': get_text('detected_disease', lang),
        'recommended_remedy': get_text('recommended_remedy', lang),
        'no_disease_detected': get_text('no_disease_detected', lang),
        'unknown_disease': get_text('unknown_disease', lang),
        'no_remedy_provided': get_text('no_remedy_provided', lang),
        'yolo_growth_stage': get_text('yolo_growth_stage', lang),
        'common_name_unknown': get_text('common_name_unknown', lang),
        'no_yolo_result': get_text('no_yolo_result', lang),
        'plant_analysis_instructions': get_text('plant_analysis_instructions', lang),
        'plant_analysis_note': get_text('plant_analysis_note', lang)
    }

    if request.method == 'POST':
        if "image_file" not in request.files:
            return jsonify({"error": current_lang_strings['error_no_file']}), 400
        file = request.files["image_file"]
        if file.filename == "":
            return jsonify({"error": current_lang_strings['error_no_file']}), 400
        unique_filename = f"upload_{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        image_path = os.path.join(os.getcwd(), "uploads", unique_filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        try:
            file.save(image_path)
        except Exception as e:
            logging.error(f"Error saving file: {e}")
            return jsonify({"error": current_lang_strings['error_processing']}), 500
        try:
            print("Starting YOLO pipeline...")
            run_yolo_pipeline(image_path)
            print("YOLO pipeline completed.")
            print("Starting PlantNet pipeline...")
            plantnet_result_file = run_plantnet_api(image_path)
            print("PlantNet pipeline completed.")
            print("Starting disease detection...")
            global global_disease_info
            global_disease_info = get_disease_information(image_path, lang)
            print("Disease detection completed.")
            yolo_result = current_lang_strings['no_yolo_result']
            if os.path.exists(YOLO_TRANSLATED_TEXT_FILE):
                with open(YOLO_TRANSLATED_TEXT_FILE, "r", encoding="utf-8") as f:
                    yolo_result = f.read().strip()
            common_name = current_lang_strings['common_name_unknown']
            if plantnet_result_file and os.path.exists(plantnet_result_file):
                with open(plantnet_result_file, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        cn = row['common_names'].split(';')[0].strip()
                        if cn:
                            common_name = cn
                            break
            if "plant 1" in yolo_result:
                yolo_result = yolo_result.replace("plant 1", f'"{common_name}"')
            if global_disease_info:
                disease_str = f"{current_lang_strings['detected_disease']}: {global_disease_info.get('disease', current_lang_strings['unknown_disease'])}"
                remedy_str = f"{current_lang_strings['recommended_remedy']}: {global_disease_info.get('remedy', current_lang_strings['no_remedy_provided'])}"
            else:
                disease_str = current_lang_strings['no_disease_detected']
                remedy_str = ""
            result = {"analysis": yolo_result, "diseases": disease_str, "remedy": remedy_str}
            response = make_response(json.dumps(result, ensure_ascii=False))
            response.headers['Content-Type'] = 'application/json; charset=utf-8'
            return response, 200
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            return jsonify({"error": current_lang_strings['error_processing']}), 500
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)
    return render_template_string(get_plant_form_html(current_lang_strings))


# ----- API Endpoints -----

@app.route('/api/crop_search', methods=['POST'])
def api_crop_search():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": get_text('error_provide_query')}), 400
    query = data.get('query', '').strip()
    folder = '/Users/michael_z/Downloads/PlantTools/Plant Database'
    df = load_local_crop_datasets(folder)
    if df is None:
        return jsonify({"error": get_text('no_crop_data_found')}), 500
    col = next((c for c in ['label', 'crop', 'common_name', 'plant_name'] if c in df.columns), None)
    if col is None:
        return jsonify({"error": get_text('no_crop_data_found')}), 500
    matching = sorted(df[col][df[col].str.contains(query, case=False, na=False)].unique().tolist())
    return jsonify({"results": matching}), 200


@app.route('/api/get_sensor_data', methods=['GET'])
def api_get_sensor_data():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    lang = request.args.get('lang', 'en')  # Default to English if not specified
    if lat is None or lon is None:
        return jsonify({"error": get_text('error_latitude_longitude_required', lang)}), 400
    now = datetime.now()
    report = get_location_info(lat, lon, now.month, now.year)
    sensor = extract_sensor_data(report)
    formatted_sensor = {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in sensor.items()}
    return jsonify(formatted_sensor), 200


@app.route('/api/crop_recommendation', methods=['POST'])
def api_crop_recommendation():
    data = request.get_json()
    if not data:
        return jsonify({"error": get_text('error_no_json_payload')}), 400
    try:
        sensor = {key: float(data.get(key)) for key in ['T', 'H', 'P', 'T_avg', 'AP', 'pH']}
        selected_plants = data.get('selectedPlants', [])
        lang = data.get('lang', 'en')  # Default to English if not specified
    except Exception as e:
        return jsonify({"error": get_text('error_invalid_data', lang).format(e)}), 400
    recommendation = process_crop_recommendation(sensor, selected_plants=selected_plants, lang=lang)
    # Ensure the response format matches the iOS app's expectations
    response = {
        "text": recommendation["text"],
        "image_url": recommendation.get("image_url")
    }
    return jsonify(response), 200


@app.route('/api/watering', methods=['POST'])
def api_watering():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": get_text('error_no_json_payload')}), 400
        latitude = data.get("latitude")
        longitude = data.get("longitude")
        watering_start = data.get("watering_start")
        watering_end = data.get("watering_end")
        pots = data.get("pots", [])
        yard = data.get("yard", {})
        watering_schedules = data.get("wateringSchedules", [])
        if not all([latitude, longitude, watering_start, watering_end]):
            return jsonify({"error": get_text('error_missing_fields')}), 400

        # Fetch weather forecast and calculate ET0
        forecast = get_open_meteo_forecast(latitude, longitude)
        if not forecast:
            return jsonify({"error": get_text('error_fetching_weather_data')}), 500
        hourly = forecast.get("hourly", {})
        write_hourly_data_to_csv(
            hourly.get("time", []),
            hourly.get("shortwave_radiation", []),
            hourly.get("windspeed_10m", []),
            [round(convert_wind_speed_to_two_meters(v), 2) for v in hourly.get("windspeed_10m", [])],
            hourly.get("cloudcover", []),
            hourly.get("relativehumidity_2m", []),
            [round(calculate_dew_point_temperature(T, RH), 2)
             if calculate_dew_point_temperature(T, RH) is not None else "N/A"
             for T, RH in zip(hourly.get("temperature_2m", []), hourly.get("relativehumidity_2m", []))]
        )
        daily = forecast.get("daily", {})
        write_daily_data_to_csv(
            daily.get("time", []),
            daily.get("temperature_2m_max", []),
            daily.get("temperature_2m_min", [])
        )

        # Get soil data
        try:
            sand, silt, clay = get_soil_values(DEFAULT_TIFF_PATH, longitude, latitude)
            sand_p = sand * 100.0
            silt_p = silt * 100.0
            clay_p = clay * 100.0
            soil_type = determine_soil_classification(sand_p, silt_p, clay_p)
            write_soil_to_csv(longitude, latitude, sand_p, silt_p, clay_p, soil_type)
        except Exception as e:
            return jsonify({"error": get_text('error_retrieving_soil_values').format(e)}), 500

        # Calculate ET0
        temp_df = pd.read_csv(TEMPERATURE_CSV_FILE)
        # Rename 'time' to 'Date' if necessary (ensure consistency)
        if 'time' in temp_df.columns and 'Date' not in temp_df.columns:
            temp_df.rename(columns={'time': 'Date'}, inplace=True)
        temp_df["Date"] = temp_df["Date"].astype(str)
        swc_df = pd.read_csv(SOIL_WATER_CONTENT_CSV_FILE)
        swc_df['Date'] = swc_df['Time'].str.split('T').str[0]
        merged_df = pd.merge(swc_df, temp_df, on="Date", how="left")

        def calc_ET0(row):
            try:
                T_max = row["Max Temperature (°C)"]
                T_min = row["Min Temperature (°C)"]
                RH = row["Relative Humidity (%)"]
                if RH <= 1:
                    RH *= 100
                u2 = row["Estimated Wind Speed 2m (m/s)"]
                Rs_MJ = row["Solar Radiation (W/m²)"] * 0.0036
                return compute_reference_evapotranspiration(T_max, T_min, RH, u2, Rs_MJ)
            except Exception:
                return None

        merged_df['ET0'] = merged_df.apply(calc_ET0, axis=1)
        daily_ET0_df = merged_df.groupby("Date")["ET0"].sum().reset_index()
        if daily_ET0_df.empty:
            return jsonify({"error": get_text('error_no_et0_data')}), 400
        daily_ET0_df.to_csv(EVAPOTRANSPIRATION_CSV_FILE, index=False)
        et0_mapping = dict(zip(daily_ET0_df["Date"], daily_ET0_df["ET0"]))
        multiplier = soil_multiplier_mapping.get(soil_type, 1.0)

        # Build daily watering schedule
        daily_watering_schedule = {}
        for date_str in et0_mapping.keys():
            daily_watering_schedule[date_str] = {
                "date": date_str,
                "items": []
            }

        # Process pots
        for pot in pots:
            pot_name = pot.get("name")
            plant_name = pot.get("plant_name")
            shape = pot.get("shape")
            if not all([pot_name, plant_name, shape]):
                continue  # Skip invalid pot data

            area = 0.0
            if shape == "Circular":
                radius = pot.get("radius", 0) / 100.0
                area = math.pi * (radius ** 2)
            elif shape == "Rectangular":
                length = pot.get("length", 0) / 100.0
                width = pot.get("width", 0) / 100.0
                area = length * width

            # Find an existing schedule for this pot/plant combination if provided by API consumer
            existing_schedule = next((s for s in watering_schedules
                                      if s.get("container") == pot_name and s.get("plant") == plant_name), None)

            # Determine frequency based on existing schedule or calculated value
            if existing_schedule:
                # If a schedule is provided, use its defined amount and frequency
                amount_per_day = existing_schedule.get("amount", 0)
                freq_min = freq_max = len(existing_schedule.get("days", []))
            else:
                # Calculate if no existing schedule
                representative_et0 = et0_mapping.get(list(et0_mapping.keys())[0], 0.0)  # Take first day's ET0
                water_data = watering_recommendation_for_pot(soil_type, representative_et0, area)
                amount_per_day = float(water_data["amount"])
                _, freq_min, freq_max = calculate_watering_frequency_for_plant(plant_name, multiplier)

            # Add to daily schedule for each day
            for date_str in et0_mapping.keys():
                # For API, we provide the calculated amount, and leave frequency as a guideline for user to apply
                daily_watering_schedule[date_str]["items"].append({
                    "container": pot_name,
                    "container_type": get_text("Pot"),
                    "plant": plant_name,
                    "amount": amount_per_day,  # Use the calculated amount or provided fixed amount
                    "frequency_min": freq_min,
                    "frequency_max": freq_max
                })

        # Process yard
        if yard:
            yard_length = yard.get("length", 0)
            yard_width = yard.get("width", 0)
            yard_plants = yard.get("plants", [])
            total_area = yard_length * yard_width

            if total_area > 0 and yard_plants:
                area_per_plant = total_area / len(yard_plants) if len(yard_plants) > 0 else total_area
                for plant_info in yard_plants:
                    plant_name = plant_info.get("name")
                    plant_area = plant_info.get("area",
                                                area_per_plant)  # Use specific area if provided, else evenly distribute
                    if not plant_name or plant_area <= 0:
                        continue

                    existing_schedule = next((s for s in watering_schedules
                                              if s.get("container") == "Yard" and s.get("plant") == plant_name), None)

                    if existing_schedule:
                        amount_per_day = existing_schedule.get("amount", 0)
                        freq_min = freq_max = len(existing_schedule.get("days", []))
                    else:
                        representative_et0 = et0_mapping.get(list(et0_mapping.keys())[0], 0.0)
                        amount_per_day = float(
                            watering_recommendation_for_yard(soil_type, representative_et0, plant_area))
                        _, freq_min, freq_max = calculate_watering_frequency_for_plant(plant_name, multiplier)

                    for date_str in et0_mapping.keys():
                        daily_watering_schedule[date_str]["items"].append({
                            "container": get_text("Yard"),
                            "container_type": get_text("Yard"),
                            "plant": plant_name,
                            "amount": amount_per_day,
                            "frequency_min": freq_min,
                            "frequency_max": freq_max
                        })

        # Save watering schedule to CSV (for internal use, not directly returned by API)
        csv_rows = [[
            get_text("table_header_area_type"),
            get_text("table_header_identifier"),
            get_text("table_header_plant_name"),  # Using plant name from database
            get_text("table_header_watering_date"),
            get_text("table_header_water_amount"),
            get_text("table_header_frequency_guideline")
        ]]

        # Ensure we write only scheduled items to CSV
        for date_entry in daily_watering_schedule.values():
            for item in date_entry["items"]:
                # The API endpoint expects a list of daily schedules, not a filtered one.
                # The CSV is for internal record. Filtered logic is handled in the UI.
                csv_rows.append([
                    item["container_type"],
                    item["container"],  # This is the specific pot name or "Yard"
                    item["plant"],  # The plant name
                    date_entry["date"],  # The date for this specific daily record
                    item["amount"],
                    get_text("watering_frequency_times_per_week").format(item['frequency_min'], item['frequency_max'])
                ])

        try:
            with open(WATERING_CSV_FILE, "w", newline="", encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(csv_rows)
        except Exception as e:
            return jsonify({"error": get_text('error_failed_to_save_csv').format(e)}), 500

        # Return response (dailySchedule will contain all calculated daily entries, not just selected ones)
        response = {
            "latitude": latitude,
            "longitude": longitude,
            "watering_start": watering_start,
            "watering_end": watering_end,
            "dailySchedule": list(daily_watering_schedule.values()),  # Return all calculated daily schedules
            "soilType": soil_type,
            "yardPlantType": None  # This field was in the original JSON response but not explicitly managed
        }
        return jsonify(response), 200
    except Exception as e:
        # Generic error handling for unexpected issues
        logging.exception("An unhandled error occurred in api_watering:")
        return jsonify({"error": get_text('error_server_error').format(str(e))}), 500


# Remove the GUI class and keep only the web interface
if __name__ == "__main__":
    from gunicorn.app.base import BaseApplication


    class StandaloneApplication(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            config = {key: value for key, value in self.options.items()
                      if key in self.cfg.settings and value is not None}
            for key, value in config.items():
                self.cfg.set(key.lower(), value)

        def load(self):
            return self.application


    # Create the 'uploads' directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    options = {
        "bind": "127.0.0.1:5000",
        "workers": 4,
    }
    StandaloneApplication(app, options).run()
