import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import streamlit as st

# ### Get test data from database 

# Get df from database 
conn = sqlite3.connect('covid_data.db')
cursor = conn.cursor()

# Get deaths df
query_deaths = """
SELECT 
    state, 
    population,
    date,
    deaths
FROM 
    deaths_table
WHERE date >= '2022-01-01'
"""
df_deaths = pd.read_sql_query(query_deaths, conn)

# Get vaccinations df
query_vaccinations = """
SELECT 
    state,
    date,
    series_complete_yes,
    booster_doses
FROM 
    vaccinations_table
"""
df_vaccinations = pd.read_sql_query(query_vaccinations, conn)

# Get beds df
query_beds = """
SELECT 
    hospital_name, 
    week_date,
    state,
    beds,
    inpatient_beds
FROM 
    beds_table
WHERE week_date >= '2022-01-01'
"""
df_beds = pd.read_sql_query(query_beds, conn)

conn.close()

#df_deaths.to_csv('df_deaths.csv', index=False)
#df_vaccinations.to_csv('df_vaccinations.csv', index=False)
#df_beds.to_csv('df_beds.csv', index=False)


# Data cleaning

# Harmonize the state records across three datasets
common_states = set(df_beds['state']) & set(df_deaths['state']) & set(df_vaccinations['state'])

df_deaths = df_deaths[df_deaths['state'].isin(common_states)]
df_vaccinations = df_vaccinations[df_vaccinations['state'].isin(common_states)]
df_beds = df_beds[df_beds['state'].isin(common_states)]


# df_beds -- Drop rows with empty dates
df_beds_weekly = df_beds.dropna(subset=['week_date'])
df_beds_weekly.rename(columns={'week_date': 'date'}, inplace=True)

#df_deaths.to_csv('df_deaths.csv', index=False)
#df_vaccinations.to_csv('df_vaccinations.csv', index=False)
#df_beds_weekly.to_csv('df_beds_weekly.csv', index=False)



# ### Get train data from database

# Get df from database -- train
conn = sqlite3.connect('covid_data.db')

# Get deaths df -- train
query_deaths = """
SELECT 
    state, 
    population,
    date,
    deaths
FROM 
    deaths_table
WHERE date < '2022-01-01'
"""
df_deaths_train = pd.read_sql_query(query_deaths, conn)

# Get beds df -- train
query_beds = """
SELECT 
    hospital_name, 
    week_date,
    state,
    beds,
    inpatient_beds
FROM 
    beds_table
WHERE week_date < '2022-01-01'
"""
df_beds_train = pd.read_sql_query(query_beds, conn)

conn.close()


# Data cleaning

# Harmonize the state records across datasets
df_deaths_train = df_deaths_train[df_deaths_train['state'].isin(common_states)]
df_beds_train = df_beds_train[df_beds_train['state'].isin(common_states)]

#df_deaths_train.to_csv('df_deaths_train.csv', index=False)
#df_beds_train.to_csv('df_beds_train.csv', index=False)



# ### Predict mortality

# Step1: Predict deaths between Jan 2022 to Mar 2022
# Exponential model
def predict_deaths(df, state, start_date, end_date):
    df_state = df[df['state'] == state].copy()
    df_state['date'] = pd.to_datetime(df_state['date'])
    df_state['t'] = (df_state['date'] - df_state['date'].min()).dt.days
    df_state['log_deaths'] = np.log(df_state['deaths'] + 1)
    
    # Checking for data adequacy
    if df_state.empty or len(df_state) < 2:
        print(f"Not enough data for {state}")
        return None
    
    # Linear regression fitting log deaths
    lin_reg = LinearRegression()
    lin_reg.fit(df_state[['t']], df_state['log_deaths'])

    # Create a date range for the forecast
    df_predict = pd.DataFrame({
        'date': pd.date_range(start=start_date, end=end_date)
    })
    df_predict['t'] = (df_predict['date'] - df_state['date'].min()).dt.days
    df_predict['log_deaths'] = lin_reg.predict(df_predict[['t']])
    df_predict['predicted_deaths'] = np.exp(df_predict['log_deaths']) - 1
    df_predict['state'] = state
    
    # Round the predicted deaths to the nearest integer
    df_predict['predicted_deaths'] = df_predict['predicted_deaths'].round(0).astype(int)

    return df_predict[['state', 'date', 'predicted_deaths']]


states = df_deaths_train['state'].unique()
all_forecasts = []


for state in states:
    forecast_df = predict_deaths(df_deaths_train, state, '2022-01-01', '2022-03-31')
    if forecast_df is not None:
        all_forecasts.append(forecast_df)


df_pred_death = pd.concat(all_forecasts)
#df_pred_death.to_csv('df_pred_death.csv', index=False)



# ### Trends in mortality and vaccination

# Trends in mortality and vaccination
def death_vaccination_trends(df_deaths, df_pred_death, df_vaccinations, state):
    # Clear the current figure's plot if any to prevent type conflicts
    plt.clf()
    
    df_deaths['date'] = pd.to_datetime(df_deaths['date'], errors='coerce')
    df_pred_death['date'] = pd.to_datetime(df_pred_death['date'], errors='coerce')
    df_vaccinations['date'] = pd.to_datetime(df_vaccinations['date'], errors='coerce')

    # Convert state codes to lower case for case-insensitive matching
    df_deaths = df_deaths.copy()
    df_deaths.loc[:, 'state'] = df_deaths['state'].str.lower()
    df_pred_death = df_pred_death.copy()
    df_pred_death.loc[:, 'state'] = df_pred_death['state'].str.lower()
    df_vaccinations = df_vaccinations.copy()
    df_vaccinations.loc[:, 'state'] = df_vaccinations['state'].str.lower()
    state_lower = state.lower()
    
    # Select a state
    df_death_state = df_deaths[df_deaths['state'] == state_lower]
    df_pred_death_state = df_pred_death[df_pred_death['state'] == state_lower]
    df_vaccination_state = df_vaccinations[df_vaccinations['state'] == state_lower]

    # Merge data
    df_merged = pd.merge(df_death_state, df_pred_death_state, on='date', how='outer')
    df_merged = pd.merge(df_merged, df_vaccination_state, on='date', how='outer')
    
    # Sort by date
    df_merged.sort_values('date', inplace=True)
    df_merged = df_merged.reset_index(drop=True)

    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot mortality data
    line1, = ax1.plot(df_merged['date'], df_merged['deaths'], label='Deaths', color='blue')
    line2, = ax1.plot(df_merged['date'], df_merged['predicted_deaths'], label='Predicted Deaths', color='red')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Deaths')

    # Create secondary vertical axes
    ax2 = ax1.twinx()
    # Plot completed vaccination data
    line3, = ax2.plot(df_merged['date'], df_merged['series_complete_yes'], label='Vaccination Series Complete', color='orange', linestyle='--')
    # Plot booster dose data
    line4, = ax2.plot(df_merged['date'], df_merged['booster_doses'], label='Booster Doses', color='green', linestyle='--')
    ax2.set_ylabel('Vaccinations')

    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SU, interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    fig.autofmt_xdate()
    
    start_date = pd.to_datetime('2022-01-02')
    end_date = pd.to_datetime('2022-03-31')
    ax1.set_xlim(start_date, end_date)


    ax1.set_title(f'COVID-19 Deaths and Vaccination Trends in {state}')
    ax1.legend(handles=[line1, line2, line3, line4], loc='upper left')

    st.pyplot(fig)


#death_vaccination_trends(df_deaths, df_pred_death, df_vaccinations, 'CA')


# Trends in the difference between predicted and actual deaths and vaccination
def vaccination_effect_on_death_trends(df_deaths, df_pred_death, df_vaccinations, state):
    # Clear the current figure's plot if any to prevent type conflicts
    plt.clf()
    
    df_deaths['date'] = pd.to_datetime(df_deaths['date'])
    df_pred_death['date'] = pd.to_datetime(df_pred_death['date'])
    df_vaccinations['date'] = pd.to_datetime(df_vaccinations['date'])

    # Convert state codes to lower case for case-insensitive matching
    df_deaths = df_deaths.copy()
    df_deaths.loc[:, 'state'] = df_deaths['state'].str.lower()
    df_pred_death = df_pred_death.copy()
    df_pred_death.loc[:, 'state'] = df_pred_death['state'].str.lower()
    df_vaccinations = df_vaccinations.copy()
    df_vaccinations.loc[:, 'state'] = df_vaccinations['state'].str.lower()
    state_lower = state.lower()

    # Select a state
    df_death_state = df_deaths[df_deaths['state'] == state_lower]
    df_pred_death_state = df_pred_death[df_pred_death['state'] == state_lower]
    df_vaccination_state = df_vaccinations[df_vaccinations['state'] == state_lower]

    # Merge the death and predicted death data
    df_merged = pd.merge(df_death_state, df_pred_death_state, on='date', how='inner')
    # Calculate the difference between predicted and actual deaths
    df_merged['death_difference'] = df_merged['predicted_deaths'] - df_merged['deaths']

    # Merge the vaccination data
    df_merged = pd.merge(df_merged, df_vaccination_state, on='date', how='inner')

    # Sort by date
    df_merged.sort_values('date', inplace=True)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the death difference
    line1, = ax1.plot(df_merged['date'], df_merged['death_difference'], label='Death Difference', color='purple')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Death Difference')
    
    # Create secondary vertical axes for vaccinations
    ax2 = ax1.twinx()

    # Plot completed vaccination data
    line2, = ax2.plot(df_merged['date'], df_merged['series_complete_yes'], label='Vaccination Series Complete', color='orange', linestyle='--')
    # Plot booster dose data
    line3, = ax2.plot(df_merged['date'], df_merged['booster_doses'], label='Booster Doses', color='green', linestyle='--')
    ax2.set_ylabel('Vaccinations')
    
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SU, interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    fig.autofmt_xdate()

    start_date = pd.to_datetime('2022-01-02')
    end_date = pd.to_datetime('2022-03-31')
    ax1.set_xlim(start_date, end_date)


    ax1.set_title(f'Effect of Vaccination on Death Trends in {state}')
    ax1.legend(handles=[line1, line2, line3], loc='upper left')

    st.pyplot(fig)

#vaccination_effect_on_death_trends(df_deaths, df_pred_death, df_vaccinations, 'AZ')


# Scatter plot

def vaccination_effect_on_death_scatter(df_deaths, df_pred_death, df_vaccinations, state):
    plt.clf()

    df_deaths['date'] = pd.to_datetime(df_deaths['date'])
    df_pred_death['date'] = pd.to_datetime(df_pred_death['date'])
    df_vaccinations['date'] = pd.to_datetime(df_vaccinations['date'])

    # Convert state codes to lower case for case-insensitive matching
    df_deaths = df_deaths.copy()
    df_deaths.loc[:, 'state'] = df_deaths['state'].str.lower()
    df_pred_death = df_pred_death.copy()
    df_pred_death.loc[:, 'state'] = df_pred_death['state'].str.lower()
    df_vaccinations = df_vaccinations.copy()
    df_vaccinations.loc[:, 'state'] = df_vaccinations['state'].str.lower()
    state_lower = state.lower()

    # Filter data by state
    df_death_state = df_deaths[df_deaths['state'] == state_lower]
    df_pred_death_state = df_pred_death[df_pred_death['state'] == state_lower]
    df_vaccination_state = df_vaccinations[df_vaccinations['state'] == state_lower]

    # Merge the death and predicted death data
    df_merged = pd.merge(df_death_state, df_pred_death_state, on='date', how='inner')
    df_merged['death_difference'] = df_merged['predicted_deaths'] - df_merged['deaths']

    # Merge the vaccination data
    df_merged = pd.merge(df_merged, df_vaccination_state, on='date', how='inner')

    # Sort by date
    df_merged.sort_values('date', inplace=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot for series completions
    ax.scatter(df_merged['series_complete_yes'], df_merged['death_difference'], color='orange', label='Series Completions')
    # Scatter plot for booster doses
    ax.scatter(df_merged['booster_doses'], df_merged['death_difference'], color='green', label='Booster Doses')

    # Calculate correlation coefficients
    corr_coef_series = df_merged['series_complete_yes'].corr(df_merged['death_difference'])
    corr_coef_booster = df_merged['booster_doses'].corr(df_merged['death_difference'])

    ax.set_xlabel('Vaccinations')
    ax.set_ylabel('Death Difference')
    ax.set_title(f'Scatter Plot of Vaccination vs Death Difference in {state}')
    ax.legend()
    
    # Adding text for correlation coefficients
    text_x = ax.get_xlim()[1] * 0.7 
    text_y = ax.get_ylim()[1] * 0.95
    ax.text(text_x, text_y, f'Correlation (Series Completions): {corr_coef_series:.2f}', fontsize=10, backgroundcolor='orange', color='white')
    ax.text(text_x, text_y - (ax.get_ylim()[1] * 0.05), f'Correlation (Booster Doses): {corr_coef_booster:.2f}', fontsize=10, backgroundcolor='green', color='white')


    st.pyplot(fig)

#vaccination_effect_on_death_scatter(df_deaths, df_pred_death, df_vaccinations, 'OH')



# ### Trends in hospital load and vaccination


# Data preprocessing

# Aggregate vaccination data on a weekly basis
df_vaccinations['date'] = pd.to_datetime(df_vaccinations['date'])

start_date = '2022-01-02'

df_vaccinations_weekly = df_vaccinations.groupby(
    ['state', pd.Grouper(key='date', freq='W-SUN', origin=pd.Timestamp(start_date))]
)[['series_complete_yes', 'booster_doses']].mean().reset_index()

# Aggregate beds data based on states
df_total_beds = df_beds_weekly.groupby(['date','state'])['inpatient_beds'].sum().reset_index()

#df_total_beds.to_csv('df_total_beds.csv', index=False)
#df_vaccinations_weekly.to_csv('df_vaccinations_weekly.csv', index=False)


# Trends in hospital load and vaccination -- by state
def beds_vaccination_trends_by_state(df_total_beds, df_vaccinations_weekly, state):
    plt.clf()
    
    df_total_beds['date'] = pd.to_datetime(df_total_beds['date'])
    df_vaccinations_weekly['date'] = pd.to_datetime(df_vaccinations_weekly['date'])
    
    # Convert state codes to lower case for case-insensitive matching
    df_total_beds = df_total_beds.copy()
    df_total_beds.loc[:, 'state'] = df_total_beds['state'].str.lower()
    df_vaccinations_weekly = df_vaccinations_weekly.copy()
    df_vaccinations_weekly.loc[:, 'state'] = df_vaccinations_weekly['state'].str.lower()
    state_lower = state.lower()
    
    # Filter data by state
    df_total_beds_state = df_total_beds[df_total_beds['state'] == state_lower]
    df_vaccinations_weekly_state = df_vaccinations_weekly[df_vaccinations_weekly['state'] == state_lower]
    
    # Merge beds and vaccination data
    df_merged = pd.merge(df_total_beds_state, df_vaccinations_weekly_state, on='date', how='inner')
  
    # Sort by date for plotting
    df_merged.sort_values('date', inplace=True)
    

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot inpatient beds data
    line1, = plt.plot(df_merged['date'], df_merged['inpatient_beds'], label='Inpatient Beds', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Inpatient Beds')
    
    # Create a secondary y-axis for vaccination rates
    ax2 = ax1.twinx()
    # Plot completed vaccination data
    line2, = ax2.plot(df_merged['date'], df_merged['series_complete_yes'], label='Vaccination Series Complete', color='orange', linestyle='--')
    # Plot booster dose data
    line3, = ax2.plot(df_merged['date'], df_merged['booster_doses'], label='Booster Doses', color='green', linestyle='--')
    ax2.set_ylabel('Vaccinations')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    fig.autofmt_xdate()

    ax1.set_title(f'Inpatient Beds and Vaccination Trends in {state}')
    ax1.legend(handles=[line1, line2, line3], loc='upper left')

    st.pyplot(fig)

#beds_vaccination_trends(df_total_beds, df_vaccinations_weekly, 'AK')


# Trends in hospital load and vaccination -- by hospital

# Get hospital list for each state
def list_hospitals_by_state(state):
    filtered_df = df_beds_weekly[df_beds_weekly['state'].str.upper() == state.upper()]
    return filtered_df['hospital_name'].tolist()

def beds_vaccination_trends_by_hospital(df_beds_weekly, df_vaccinations_weekly, state, name):
    plt.clf()
    
    df_beds_weekly['date'] = pd.to_datetime(df_beds_weekly['date'])
    df_vaccinations_weekly['date'] = pd.to_datetime(df_vaccinations_weekly['date'])
    
    # Convert state codes to lower case for case-insensitive matching
    df_beds_weekly = df_beds_weekly.copy()
    df_beds_weekly.loc[:, 'state'] = df_beds_weekly['state'].str.lower()
    df_vaccinations_weekly.loc[:, 'state'] = df_vaccinations_weekly['state'].str.lower()
    state_lower = state.lower()
    
    # Filter data by state
    df_beds_weekly_state = df_beds_weekly[df_beds_weekly['state'] == state_lower]
    df_vaccinations_weekly_state = df_vaccinations_weekly[df_vaccinations_weekly['state'] == state_lower]
    
    # Convert hospital names to lower case for case-insensitive matching
    df_beds_weekly_state = df_beds_weekly_state.copy()
    df_beds_weekly_state.loc[:, 'hospital_name'] = df_beds_weekly_state['hospital_name'].str.lower()
    name = name.replace("'", "").replace('"', "")
    name_lower = name.lower()

    # Further filter data by hospital name
    df_beds_specific_hospital = df_beds_weekly_state[df_beds_weekly_state['hospital_name'] == name_lower]

    if df_beds_specific_hospital.empty:
        st.write(f"No hospital found in {state} with the name {name}.")
        return
    
    # Merge beds and vaccination data
    df_merged = pd.merge(df_beds_specific_hospital, df_vaccinations_weekly_state, on='date', how='outer')
  
    # Sort by date for plotting
    df_merged.sort_values('date', inplace=True)
    

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    #line1, = plt.bar(df_merged['date'], df_merged['inpatient_beds'], label='Inpatient Beds', color='blue', width=5, align='center')
    line1, = ax1.plot(df_merged['date'], df_merged['inpatient_beds'], label='Inpatient Beds', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Inpatient Beds')
    
    # Create a secondary y-axis for vaccination rates
    ax2 = ax1.twinx()
    # Plot completed vaccination data
    line2, = ax2.plot(df_merged['date'], df_merged['series_complete_yes'], label='Vaccination Series Complete', color='orange', linestyle='--')
    # Plot booster dose data
    line3, = ax2.plot(df_merged['date'], df_merged['booster_doses'], label='Booster Doses', color='green', linestyle='--')
    ax2.set_ylabel('Vaccinations')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    fig.autofmt_xdate()

    ax1.set_title(f'Inpatient Beds and Vaccination Trends in {name}')
    ax1.legend(handles=[line1, line2, line3], loc='upper left')

    st.pyplot(fig)



# ### Build streamlit app

# Side bar
st.sidebar.title('Navigation')

# Main pages
main_pages = ['Home', 'Introduction', 'Data overview', 'Trending', 'Q&A']

page = st.sidebar.radio('Go to:', main_pages)

# sub pages
if page == 'Home':
    st.title('Welcome to My Project')
    st.write("Welcome to Zexin Gong's project! This Streamlit app provides insights into "
             "the effects of vaccines on COVID-19.")


elif page == 'Introduction':
    st.title('Q1&Q2: App Explanation')
    st.write("**Name**: Zexin Gong\n\n"
             "The side bar of this app contains five major headings, with several subheadings under each "
             "major heading. You can click on a major heading and select a subheading in the 'see more section' below it.\n\n"
             "The Data overview section contains the data used in the project with explanations of some of "
             "the data. The Trending section contains trend and relationship graphs. Overall, vaccinations "
             "can reduce the mortality rate of covid_19 and ease the hospital load rate."
             )
    st.title('Q3: Potential gotchas')
    st.write("1. Since the data comes from different data sources, there may be contradictions when making "
             "data predictions (e.g., the predicted mortality rate is so low that it is illogical and has to "
             "be removed)\n\n"
             "2. The pattern of hospital inpatient beds is difficult to capture, so there are no appropriate models for "
             "predicting hospital workload in the absence of vaccination, i.e. predicted inpatient beds. Therefore, "
             "the conclusion about the effect of vaccines on hospital workload may be biased, which means the reduction "
             "in hospital workload may also be influenced by other factors."
             )


elif page == 'Data overview':
    with st.sidebar.expander("See More"):
        data_pages = st.radio(
            "Select a dataset:",
            ['Deaths Data', 'Vaccination Data', 'Hospital Beds Data']
        )

    if data_pages == 'Deaths Data':
        st.title('Covid-19 Deaths Table')
        st.write("This table presents population and mortality data across various states in the United States "
                 "from 2022-01-01 to 2022-03-31.")
        st.dataframe(df_deaths)
    
    elif data_pages == 'Vaccination Data':
        st.title('Vaccination Table')
        st.write("This table provides the vaccination information across various states in the United States "
                 "from 2022-01-01 to 2022-03-31. Vaccination is categorized into two types:\n\n "
                 "**series_complete_yes:** Total number of people who have completed a primary series "
                 "(have second dose of a two-dose vaccine or one dose of a single-dose vaccine).\n\n "
                 "**booster_doses:** Total number of people who completed a primary series and "
                 "have received a booster (or additional) dose."
                 )
        st.dataframe(df_vaccinations)
    
    elif data_pages == 'Hospital Beds Data':
        st.title('Hospital Workload Table')
        st.write("This table provides the workload of different hospitals in the United States "
                 "from 2022-01-01 to 2022-03-31, mainly covering the number of hospital beds "
                 "and the number of beds occupied by inpatients.\n\n"
                 "**inpatient_beds:** Average of reported patients currently hospitalized in an inpatient bed "
                 "who have suspected or confirmed COVID-19 reported during the 7-day period.")
        st.dataframe(df_beds_weekly)


elif page == 'Trending':
    with st.sidebar.expander("See More"):
        intro_pages = st.radio(
            "Select a trends:",
            ['Mortality vs. Vaccinations', 'Deaths Difference vs. Vaccinations', 'Hospital Workload vs. Vaccinations by State',
             'Hospital Workload vs. Vaccinations by Hospital']
        )

    if intro_pages == 'Mortality vs. Vaccinations':
        st.title('Trends in Mortality and Vaccinations')
        st.write("This line graph illustrates the trends in predicted deaths (representing deaths in the "
                 "absence of vaccination), actual deaths, and vaccinations across various states. ")
        state_input = st.text_input("Enter a state code(case insensitive)", value="CA")
        if st.button("Generate Graph"):
            death_vaccination_trends(df_deaths, df_pred_death, df_vaccinations, state_input)
        st.write("From the graph, it's evident that predicted deaths in most states exhibit exponential growth "
                 "over time, while the increase in actual deaths appears relatively gradual. Both types of "
                 "vaccinations show a steady rise over time, indicating a growing willingness among the public "
                 "to receive vaccinations.")

    elif intro_pages == 'Deaths Difference vs. Vaccinations':
        st.title('Trends in Deaths Difference and Vaccinations')
        st.write("These graphs illustrate the trends and relationship between the difference in predicted "
                 "and actual deaths and vaccinations across various states.")
        state_input = st.text_input("Enter a state code(case insensitive)", value="CA")
        if st.button("Generate Graph"):
            vaccination_effect_on_death_trends(df_deaths, df_pred_death, df_vaccinations, state_input)
            vaccination_effect_on_death_scatter(df_deaths, df_pred_death, df_vaccinations, state_input)
        st.write("From the charts, it's apparent that in most states, death differences increase with "
                 "vaccinations. Additionally, the growth rate of booster doses exceeds that of primary "
                 "series. The correlation analysis indicates a high correlation between death differences and "
                 "vaccinations in most states. This indicates that vaccination can reduce deaths, with a greater "
                 "reduction observed in deaths among those who received booster doses.")

    elif intro_pages == 'Hospital Workload vs. Vaccinations by State':
        st.title('Trends in Hospital Workload and Vaccinations -- by State')
        st.write("This graph illustrate the trends between hospital inpatient beds and vaccinations "
                 "across various states.")
        state_input = st.text_input("Enter a state code(case insensitive)", value="CA")
        if st.button("Generate Graph"):
            beds_vaccination_trends_by_state(df_total_beds, df_vaccinations_weekly, state_input)
        st.write("From the charts, it's evident that in most states, inpatient beds decrease over time. "
                 "This suggests that vaccination can alleviate hospital burden.")
    
    elif intro_pages == 'Hospital Workload vs. Vaccinations by Hospital':
        st.title('Trends in Hospital Workload and Vaccinations -- by Hospital')
        st.write("This graph illustrate the trends between hospital inpatient beds and vaccinations "
                 "across various hospitals")
        state_input = st.text_input("Enter a state code(case insensitive)", value="CA")
        if st.button("Show Hospital List"):
            hospital_list = list_hospitals_by_state(state_input)
            if hospital_list:
                st.write("Hospitals in", state_input.upper(), ":")
                st.write(hospital_list)
            else:
                st.write("No hospitals found in the state of", state_input.upper())
        hospital_input = st.text_input("Enter a hospital name(case insensitive)", value="ADVENTIST HEALTH AND RIDEOUT")
        if st.button("Generate Graph"):
            beds_vaccination_trends_by_hospital(df_beds_weekly, df_vaccinations_weekly, state_input, hospital_input)



elif page == 'Q&A':
    st.title('Q4: Project Objective')
    st.write("These project aims to find the effectiveness of vaccination campaigns"
             "in reducing COVID-19 cases and fatalities. By correlating vaccination rates with case counts "
             "and death rates, I hope to find a clear inverse relationship, suggesting that higher vaccination "
             "rates contribute to lower infection and mortality rates. Additionally, I plan to explore the "
             "effects of rising cases on hospital capacity, suggesting that higher vaccination rates not "
             "only reduce infections but also ease the burden on healthcare facilities."
             )
    st.title('Q5: Discover & Conclusion')
    st.write("Study reveals that elevated vaccination rates can reduce COVID-19 mortality, particularly notable "
             "in individuals who have received booster doses. Vaccination also appears to alleviate hospital workload, "
             "albeit with potential margin for error in this conclusion. These findings align with original assumptions."
             )          
    st.title('Q6: Difficulties')
    st.write("1. Aligning the data on deaths, vaccinations and hospital inpatient beds at a same time slot "
             "for the study was challenging, given the difficulty in searching such data source.\n\n"
             "2. Extending from the first difficulty, predicting deaths and hospital inpatient beds were "
             "arduous, as there exists bias in the dataset and identifying a model that minimizes error and "
             "provides a robust fit presents significant challenges."
             )
    st.title('Q7: Skills')
    st.write("I wish I knew more model types and methods for predicting data. I also want to bulid a prettier streamlit app."
             )
    st.title('Q8: Future Work')
    st.write("I hope to explore the possibility of finding better data sources and additional factors to "
             "establish more suitable models, examining the impact of vaccination on COVID-19 and its various "
             "societal ramifications, such as using sentiment analysis to analyze the impact of hospital workload "
             "on customer satisfaction. Through this exploration, I aim to encourage vaccine uptake and mitigate "
             "the effects of COVID-19 on our daily lives.")

