# PlantTools - Smart Agriculture Assistant

A comprehensive agriculture tool that provides intelligent recommendations for plant care, watering schedules, and crop selection based on environmental conditions and soil analysis. Available as a web application and iOS companion app.

## 🌟 Features

### 1. Watering Recommendations
- Calculates optimal watering schedules based on:
  - Soil type and composition
  - Local weather conditions
  - Plant-specific requirements
  - Container size and type
- Supports both pot and yard area calculations
- Generates detailed watering plans with frequency and amount recommendations

### 2. Crop Recommendations
- Provides crop suggestions based on:
  - Local climate data
  - Soil conditions
  - Temperature and humidity
  - Available growing space
- Supports both optimal and selective modes
- Includes detailed growing requirements for each crop

### 3. Plant Analysis
- Plant identification using PlantNet API
- Disease detection using YOLO and ResNet-34 models
- Growth stage classification
- Detailed plant health analysis
- Treatment recommendations for identified issues

## 🛠 Technical Stack

### Backend
- Python 3.x
- Flask web framework
- Gunicorn WSGI server
- Key Dependencies:
  - `ultralytics` (YOLO)
  - `torch` and `torchvision`
  - `rasterio` for geospatial data
  - `pandas` for data manipulation
  - `requests` for API calls
  - `gunicorn` for production deployment


## 📋 Prerequisites

### Backend
- Python 3.x
- pip (Python package manager)
- Virtual environment (recommended)
- Required API keys:
  - OpenWeather API
  - PlantNet API
  - Visual Crossing API

## 🚀 Installation

### Backend Setup
1. Download the Repository from releases (don't clone the repo, it doesn't contain all the files), unzip, and navigate to the folder

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   export OPENWEATHER_API_KEY="your_key"
   export PLANTNET_API_KEY="your_key"
   export VISUAL_CROSSING_API_KEY="your_key"
   ```

5. Run the development server:
   ```bash
   python server.py
   ```
   
## 🌐 API Endpoints

### Watering Recommendations
- `POST /api/watering`
  - Input: Location, soil type, container details
  - Output: Watering schedule and recommendations

### Crop Recommendations
- `POST /api/crop_recommendation`
  - Input: Location, soil conditions, preferences
  - Output: Crop recommendations with details

### Plant Analysis
- `POST /api/plant_analysis`
  - Input: Plant image
  - Output: Plant identification and health analysis

## 📱 iOS App Features

### User Interface
- Clean, modern SwiftUI design
- Customizable theme colors
- Dark mode support
- Responsive layouts
- Smooth animations and transitions

### Functionality
- Real-time plant analysis
- Location-based recommendations
- Multi-language support
- Offline capabilities
- Camera integration for plant photos
- Weather data integration

### Settings
- Language selection (English/Chinese)
- Theme customization
- Location permissions
- Notification preferences
- About section with app information

## 🔒 Security

- API key management
- Secure file uploads
- Input validation
- Session management
- HTTPS support
- Location data privacy
- Secure image handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## 🙏 Acknowledgments

- PlantNet API for plant identification
- OpenWeather API for weather data
- Visual Crossing API for historical climate data
- YOLO and ResNet-34 for image analysis
