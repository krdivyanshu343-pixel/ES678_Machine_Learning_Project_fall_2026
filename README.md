# Data access

The project uses **`city_day.csv`** from the Kaggle dataset *Air Quality Data in India (2015 - 2020)*
(<https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india>). The data were made publicly
available by the Central Pollution Control Board (CPCB), Government of India. Downloading needs a free
Kaggle account. The file is **not stored in this repository** (the Kaggle page does not state a
redistribution licence), so please download it yourself.

## Steps
1. Sign in to Kaggle and open the dataset page above, then download the dataset (a zip file).
2. Unzip it and copy **`city_day.csv`** into this folder, so the path is `data/city_day.csv`.
   (Or point to it with the environment variable `AQ_DATA=/path/to/city_day.csv`.)
3. In Google Colab, just upload `city_day.csv` when the notebook asks.

## What the file contains (verified)
29,531 rows, 26 cities, one row per city per day from 2015-01-01 to 2020-07-01, columns
`City, Date, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, AQI, AQI_Bucket`.
No personal or private information. `AQI` and `AQI_Bucket` are never used as features or targets.
