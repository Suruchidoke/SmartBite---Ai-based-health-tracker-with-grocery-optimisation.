# 🌿 SmartBite — AI-Based Health Tracker & Multi-Cuisine Grocery Optimizer

SmartBite is an intelligent, closed-loop nutrition and grocery management system designed to bridge the gap between personalized dietary planning and smart shopping. Rather than isolating meal recommendations from grocery lists, SmartBite unifies biometric calculations, multi-cuisine AI recipe generation, automated ingredient consolidation, and real-time intake tracking.

---

## 🚀 Key Features

### 1. Multi-Cuisine Dietary Personalization
- **World Cuisine Support**: Tailor meals across **Indian, Pan-Asian, Mediterranean, Mexican, and Continental** cuisines.
- **Client & Server Mifflin-St Jeor Engine**: Accurately computes Basal Metabolic Rate (BMR), Total Daily Energy Expenditure (TDEE), calorie deficit/surplus goals, and macronutrient targets (45% Carbs, 30% Protein, 25% Fat).

### 2. Intelligent Diet-to-Grocery Bridge
- **1-Click Push to Grocery Optimizer**: Automatically extracts, normalizes, and consolidates raw recipe ingredients from the weekly meal plan into an organized shopping cart.
- **Zero Food Waste via Pantry Deductions**: Items already in your pantry are automatically deducted from the shopping checklist.

### 3. Smart Grocery Optimizer & Store Catalog
- **Consolidated Shopping Checklist**: Real-time item count, cost estimation, and instant "In Pantry" transfer toggles.
- **Searchable Product Catalog**: Browse 25,000+ grocery products with real-time nutrition information and price estimates.

### 4. Interactive Health Tracker Dashboard
- **Live Calorie Ring & Linear Progress**: Tracks consumed calories against daily Mifflin-St Jeor targets.
- **Chart.js Macronutrient Doughnut**: Real-time visualization of Carbohydrates, Protein, and Healthy Fats.
- **Today's Meal Highlights**: Instant "Mark as Eaten" logging for Breakfast, Lunch, and Dinner.
- **Quick Activity Logger**: Log daily physical activities with automatic streak tracking.

### 5. Interactive Fitness & Wellness Hub
- **BiteBot AI Assistant**: On-demand recipe inspiration tailored to current pantry ingredients.
- **Fitness Mini-Games & Quizzes**: Yoga Pose Quiz, Plank Timer, Nutrition Label Reader, and Habit Streak Master.

---

## 🛠️ Architecture & Tech Stack

- **Backend**: Python / Flask, RESTful APIs, JWT Authentication
- **Machine Learning**: Scikit-learn (RandomForest, GradientBoosting, Cosine Similarity Content-Based Filtering)
- **Database**: MongoDB (with local `mongomock` fallback)
- **Frontend**: Vanilla HTML5, Modern CSS / Tailwind CSS, Chart.js, Font Awesome 6, Inter typography

---

## 📦 Quick Start Guide

### Prerequisites
- Python 3.10+
- MongoDB (optional, automatically uses in-memory mock if local MongoDB is offline)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Suruchidoke/SmartBite.git
   cd SmartBite
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Access the application**:
   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

---

## 🧪 Running Tests

To run the unit test suite:
```bash
pytest tests/
```

---

## 📄 License
This project is licensed under the MIT License.
