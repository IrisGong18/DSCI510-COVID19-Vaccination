# Examing the Impact of COVID-19 Vaccination Campaigns on Case Counts and Hospital Capacities
## Streamlit URL:
[Examing the Impact of COVID-19 Vaccination Campaigns on Case Counts and Hospital Capacities](https://dsci510-finalproject-iris.streamlit.app)

## Project Overview
These project aims to find the effectiveness of vaccination campaignsin reducing COVID-19 cases and fatalities. By correlating vaccination rates with case counts and death rates, I hope to find a clear inverse relationship, suggesting that higher vaccination rates contribute to lower infection and mortality rates. Additionally, I plan to explore the effects of rising cases on hospital capacity, suggesting that higher vaccination rates not only reduce infections but also ease the burden on healthcare facilities.

## Data Sources
1. ### Covid Tracking Data
   **source:** https://covidtracking.com/data/api/version-2
   
   **description:** This data source provides daily new cases and deaths of COVID-19 across various regions of the United States. I would like to use the API under the "single day of data for a state or territory" folder.

   **Related csv after data cleaning:** df_deaths.csv, df_deaths_train.csv

2. ### COVID-19 Vaccinations in the United States,County
   **source:** https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh/data_preview
   
   **description:** This data source provides vaccination status for different counties in the United States. The data contain specific numbers and proportions of people vaccinated with different doses.

   **Related csv after data cleaning:** df_vaccinations.csv

3. ### Hospital Facilities I
   **source**: https://covidtracking.com/data/hospital-facilities#-96.93729196668039,43.588482486816645,3.5
   
   **description**: This website provides an overview of the COVID-19 burden on hospitals across various regions of the United States, including adult COVID-19 patients currently in hospital and percent of adult inpatient beds occupied by COVID-19 patients.

   **Related csv after data cleaning:** df_beds.csv, df_beds_train.csv

4. ### Hospital Facilities II
   **source**: https://www.hospitalsafetygrade.org/all-hospitals
   
   **description**: This website provides hospital lists and their scores across the United States.

   **Related csv after data cleaning:** df_beds.csv

## How to Use this Repository
Final.py contains all codes.

## Methodology:
**Data collection:** Access API, web scrape
**Data preprocessing:** Store the data in the sql database with the joint primary key of state and date. Harmonization of dates in different tables to facilitate subsequent analysis. Merge disparate databases with similar themes into a unified dataset. Refine the integration process to ensure seamless cohesion.
**Data Analysis:** Apply linear regression and ARIMA models to forecast mortality rates and hospital workload. Visualize multiple variables on one trend graph to facilitate straightforward comparison.
