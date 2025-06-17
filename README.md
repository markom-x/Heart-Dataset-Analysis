# DATASET ANALYSIS

## Video Demo

### This presentation was made for training purposes. I'm using the most common libraries in data analysis (like pandas, numpy, matplotlib and scipy) to enhance skills also in python since this was created as a final project to the CS50P-Harvard online course


# 🫀 Heart Symptoms Dataset Analysis


## 🌍 Project Summary

This tool leverages clinical data from four different hospitals and regions:

- **San Francisco, California (USA)** – VA Medical Center  
- **Cleveland, Ohio (USA)** – Cleveland Clinic Foundation  
- **Zurich, Switzerland** – University Hospital  
- **Budapest, Hungary** – Hungarian Institute of Cardiology  

Each city's dataset is visually and statistically analyzed, with dedicated color-coding used consistently throughout the app for improved readability:

| City           | Color Code | Visualization Color |
|----------------|------------|----------------------|
| San Francisco  | `#3449eb`  | 🟦 Blue              |
| Cleveland      | `#eb4034`  | 🟥 Red               |
| Zurich         | `#3deb34`  | 🟩 Green             |
| Budapest       | `#dfeb34`  | 🟨 Yellow            |

---

## 🚀 Features

- 📌 **Dataset origin map** to locate each hospital on a global map  
- 📊 **Bar charts** showing data completeness by attribute and city  
- 📈 **Linear regression analysis** for blood pressure vs age  
- 🧮 **Gaussian distribution fitting** on `trestbps` (resting BP)  
- 🧾 **Cholesterol heatmap** by city and age group  
- 🔍 In-depth medical commentary and markdown-rich insights  
- 🧪 Includes math notation for understanding regression and modeling  

---

## 🧠 Methodology

This application is powered by Python and built with the following technologies:

- `pandas` for data processing  
- `matplotlib` and `seaborn` for plotting  
- `numpy` for numerical computations  
- `scipy.stats.norm` for Gaussian fitting  
- `streamlit` for interactive front-end interface  

The project starts by loading datasets using a dedicated loader class. It accounts for missing values (`?` or `0`) and harmonizes columns across files.

Each analysis module focuses on a specific question:

1. **How complete is the data per city?**  
2. **What’s the relationship between age and blood pressure?**  
3. **Are distributions of `trestbps` normal, and how do cities compare?**  
4. **How does cholesterol vary across age groups and locations?**

All these are visualized interactively using `streamlit.pyplot()` components, with commentary and code explanations provided for educational transparency.

---

## 📂 Dataset Information

Files used:

- `processed.cleveland.data`  
- `processed.va.data`  
- `processed.hungarian.data`  
- `processed.switzerland.data`  

Missing data entries are automatically converted into `NaN` using `na_values`.

---

## 📉 Key Analysis

### 📍 Age vs. Resting Blood Pressure

A linear regression is computed and visualized using `numpy.polyfit()`:

```python
slope, intercept = np.polyfit(x, y, deg=1)
ax.plot(x, slope * x + intercept)
```
---

## 🧪 Running the App
Install the required packages:

```
pip install streamlit pandas matplotlib seaborn scipy
```

Set the correct base_path to where your .data files are located in the script, then launch the app with:

```
streamlit run dataset_analysis.py
```
---
## 📁 Project Structure

```
heart-analysis/
├── heart_analysis_app.py
├── processed.cleveland.data
├── processed.va.data
├── processed.hungarian.data
└── processed.switzerland.data
```
## 🙌 Acknowledgments

UCI Machine Learning Repository – Heart Disease Dataset

Streamlit – For the data app framework

All the institutions that made this public health data available
