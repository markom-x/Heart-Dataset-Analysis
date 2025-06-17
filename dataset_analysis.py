import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import matplotlib as mpl


COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

class HeartDatasetLoader:
    CITY_COLORS = {
        'San Francisco': "#3449eb",
        'Cleveland': "#eb4034",
        'Zurich': "#3deb34",
        'Budapest': "#dfeb34",
    }

    def __init__(self, base_path):
        self.datasets = {
            "cleveland": pd.read_csv(f"{base_path}/processed.cleveland.data", names=COLUMNS, na_values="?"),
            "california": pd.read_csv(f"{base_path}/processed.va.data", names=COLUMNS, na_values="?"),
            "hungarian": pd.read_csv(f"{base_path}/processed.hungarian.data", names=COLUMNS, na_values="?"),
            "switzerland": pd.read_csv(f"{base_path}/processed.switzerland.data", names=COLUMNS, na_values=["?", 0]),
        }

    def get(self, name):
        return self.datasets.get(name)
    
    @classmethod
    def get_city_color(cls, city):
        return cls.CITY_COLORS.get(city, "#000000") 


def main():
    base_path = r"C:\Users\marco\Downloads\heart+disease (1)"
    loader = HeartDatasetLoader(base_path)
    cleveland_df = loader.get("cleveland")
    california_df = loader.get("california")
    hungarian_df = loader.get("hungarian")
    switzerland_df = loader.get("switzerland")

    st.title("HEART-SYMPTOMS DATASET ANALYSIS")
    show_map()
    dataset(cleveland_df, california_df, hungarian_df, switzerland_df, loader)
    st.header("Exploring the data")
    
    gaussian(cleveland_df, california_df, hungarian_df, switzerland_df, loader)
    data(cleveland_df, california_df, hungarian_df, switzerland_df)


def get_sample(df):
    print(df.head())


def show_map():
    st.header("Dataset map")
    st.markdown('''
     Those data come from 4 countries, each represented with a specific color for the whole presentation:  
     1. V.A. Medical Center, Long Beach, CA 🟦
     2. Cleveland Clinic Foundation 🟥
     3. University Hospital, Zurich, Switzerland 🟩
     4. Hungarian Institute of Cardiology, Budapest 🟨
     '''
)
    df = pd.DataFrame(
        {
        "city": ["San Francisco", "Zurich", "Budapest", "Cleveland"],
        "lat": [37.7749, 47.3769, 47.4979, 41.4993],
        "lon": [-122.4194, 8.5417, 19.0402, -81.6944],
        "col": ["#3449eb","#3deb34", "#dfeb34", "#eb4034"],
    }
    
    )
    st.map(df, latitude="lat", longitude="lon", color="col", zoom=1.55)
    st.divider()



def data(cleveland_df, california_df, hungarian_df, switzerland_df):
    st.subheader("Correlation between age and resting blood pressure")
    st.write("Taking just the cleveland dataset (as we have seen is the most complete) we notice that the possibility of heart failure increases with aging. This is caused by different factors:")
    st.markdown(
        '''
    - Decreased cardiovascular efficiency  
    - Higher likelihood of chronic conditions (e.g., hypertension, diabetes)  
    - Accumulated lifestyle risks over time
    '''
    )
    x = cleveland_df["age"]
    y1 = cleveland_df["trestbps"]
    y2 = cleveland_df["chol"]
    fig, ax = plt.subplots()
    ax.scatter(x, y1, c="#eb4034")
    #ax.scatter(x, y2, c="#34de0d")
    slope, intercept = np.polyfit(x, y1, deg=1)
    ax.plot(x, (slope*x + intercept), linewidth=5, color="black", label="linear regression")
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylabel('resting bp')
    ax.set_xlabel('age')
    st.pyplot(fig)
    code = '''
            x = cleveland_df["age"]
            y = cleveland_df["trestbps"]
            fig, ax = plt.subplots()
            ax.scatter(x, y, c="#0dde79")
            slope, intercept = np.polyfit(x, y, deg=1)
            ax.plot(x, (slope*x + intercept), linewidth=5, color="black", label="linear regression")
            ax.legend(fontsize=10)
    '''
    st.write("This is the code that creates the plot and compute the linear regression")
    st.code(code)
    st.write('''We easily notice that numpy.polyfit() handles all the math for the computation, but here are the concepts used to create the line we just minimaze the cost function''')
    st.latex(r"\min_{m,\,b} \; J(m, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( mx^{(i)} + b - y^{(i)} \right)^2")
    st.write("once found m and b thanks to partial derivatives (studying for what values the cost functions J(m, b) reaches his minimum) we have the line that fits all the data")
    st.latex(r"\hat{y} = mx + b")
    st.divider()

    





def dataset(cleveland_df, california_df, hungarian_df, switzerland_df, loader):
    st.write("Here is the distribution of the scores for each dataset only considering the 4 most important values")
    attributes = ("age", "trestbps", "chol", "restecg")

    city_dfs = {
        'San Francisco': california_df,
        'Cleveland': cleveland_df,
        'Zurich': switzerland_df,
        'Budapest': hungarian_df,
    }

    city_values = {
        city: tuple(df[col].count() for col in attributes)
        for city, df in city_dfs.items()
    }

    x = np.arange(len(attributes)) 
    width = 0.2 
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for city, measurement in city_values.items():
        offset = width * multiplier
        col = loader.get_city_color(city)
        rects = ax.bar(x + offset, measurement, width, label=city, color=col)
        multiplier += 1

    ax.set_ylabel('n of scores available')
    ax.set_title('Dataset score by attribute')
    ax.set_xticks(x + width, attributes)
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0, 350)
    st.pyplot(fig)
    st.write(
    "Cleveland and Budapest have the most complete data, with consistently high numbers of scores across all attributes. "
    "San Francisco has fewer records overall, especially for trestbps and chol. "
    "Zurich has notably limited data, particularly for restecg, where it has very few scores available compared to the other cities.\n\n"
    "This suggests that while Cleveland and Budapest provide strong coverage for analysis, data from Zurich and San Francisco may be less reliable "
    "due to missing values in key attributes."
)


def gaussian(cleveland_df, california_df, hungarian_df, switzerland_df, loader):
    city_dfs = {
        'San Francisco': california_df,
        'Cleveland': cleveland_df,
        'Zurich': switzerland_df,
        'Budapest': hungarian_df,
    }

    city_values = {
        city : (np.nanmean(np.array(df["trestbps"])), np.nanstd(np.array(df["trestbps"])))
        for city, df in city_dfs.items()
    }

    x = np.linspace(100, 160, 100)
    fig, ax = plt.subplots()
    
    for city, (mu, sigma) in city_values.items():
        y = norm.pdf(x, mu, sigma)
        col = loader.get_city_color(city)
        ax.plot(x, y, label=city, color=col)
    
    for city, (mu, sigma) in city_values.items():
        col = loader.get_city_color(city)
        y = norm.pdf(mu, mu, sigma) 
        ax.vlines(mu, 0.004, y, linestyle='--', alpha=0.7, color=col, label=f'{city} mean')

    
    ax.set_xlabel("trestbps")
    ax.set_ylabel("Probability Density")

    st.subheader("Gaussian Fit for trestbps of different cities")
    st.pyplot(fig)
    st.write("I genuinely expected to find higher values for american cities (it was really the only purpose of the plot) but interstingly Cleveland takes place behind Budapest. Another finding is that turistic cities like San Francisco and Zurich have broader std probably for the presence of different ethnical groups or just cause they're less reliable (like we said earlier).")

    st.divider()
    st.subheader("Cholesterol mean (mg/dl) by Age and City")

    table = create_table(cleveland_df, california_df, hungarian_df, switzerland_df)
    fig, ax = plt.subplots()
    im = ax.imshow(table.values)
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_xticklabels(table.columns)
    ax.set_yticklabels(table.index)
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            value = table.values[i, j]
            if not np.isnan(value):
                ax.text(j, i, f'{value:.0f}', ha="center", va="center", color="black", fontsize=8)
    ax.set_title("Cholesterol (mean) by Age Group and City")
    fig.tight_layout()
    st.pyplot(fig)
    st.write("Unfortunately, we don't have any data for Zurich. However, we can observe that different age groups within the same city have different dietary habits. Another interesting point is that San Francisco is the only city in the dataset where the correlation between age and cholesterol is relatively weak.")


def create_table(cleveland_df, california_df, hungarian_df, switzerland_df):
    cleveland_df['city'] = 'Cleveland'
    california_df['city'] = 'San Francisco'
    hungarian_df['city'] = 'Budapest'

    combined = pd.concat([
        cleveland_df, california_df, hungarian_df, switzerland_df
    ], ignore_index=True)

    bins = [20, 30, 40, 50, 60, 70, 80]
    labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
    combined['age_group'] = pd.cut(combined['age'], bins=bins, labels=labels, right=False)

    table = combined.pivot_table(
        index='city', columns='age_group', values='chol', aggfunc='mean'
    )
    return table


if __name__ == "__main__":
    main()

    