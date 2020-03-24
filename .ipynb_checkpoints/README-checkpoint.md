# NOAA-Climate-Time-Series-Analysis

### Project Goals
The goal of this project was to come up with univariate SARIMA models that can model change in climate metrics across the US. The matrics we are interested in are monthly maximum, minimum, and average temperatures, cooling and heating degree days (CDD/HDD), and several indicators of drought. These metrics, as well as other interesting climate data, can be found on the National Oceanic and Atmospheric Administration (NOAA) [Climate at a Glance page](https://www.ncdc.noaa.gov/cag/). Models are constructed for each metric and for each state using the [pmdarima](https://pypi.org/project/pmdarima/) library. Runtime performance is improved by first doing stationarity testing and examining autocorrelations to get a sense of relevant model parameters.

### Directory Tree

- cag_csvs - Data obtained from the climate at a glance page
- forecast_results - Results of the forecasting
- visuals - Visuals used for project presentation
- api_test.ipynb - NOAA api tests (Under construction)
- CA_climate_presentation.pdf - Presentation focusing on California's climate
- data_fetching.ipynb - Code used to obtain the data from the NOAA website
- forecast_helper.py - Functions used to automate forecasting processes
- Forecasting.ipynb - Notebook containing code used to make forecasts


