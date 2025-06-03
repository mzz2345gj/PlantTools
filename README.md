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

### iOS App
- SwiftUI framework
- Core Location for location services
- UserNotifications for alerts
- Key Features:
  - Modern SwiftUI interface
  - Dark mode support
  - Multi-language support (English and Chinese)
  - Location-based services
  - Real-time plant analysis
  - Customizable theme colors
  - Offline capability for basic features

## 📋 Prerequisites

### Backend
- Python 3.x
- pip (Python package manager)
- Virtual environment (recommended)
- Required API keys:
  - OpenWeather API
  - PlantNet API
  - Visual Crossing API

### iOS App
- Xcode 14.0 or later
- iOS 15.6 or later
- Swift 5.5 or later
- CocoaPods (for dependency management)

## 🚀 Installation

### Backend Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/mzz2345gj/PlantTools.git
   cd PlantTools
   ```

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

### iOS App Setup
1. Clone the iOS app repository:
   ```bash
   git clone https://github.com/mzz2345gj/Gardener-iOS.git
   cd Gardener-iOS
   ```

2. Install dependencies:
   ```bash
   pod install
   ```

3. Open `Gardener.xcworkspace` in Xcode
4. Configure your development team and bundle identifier
5. Build and run the application

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
- SwiftUI and Apple's development tools
