# figure8_disaster_pipeline

## Table of contents

- [Motivations](#motivations)
- [Packages used](#packages_used)
- [Instructions](#instructions)
- [Files](#files)
- [Possible improvements](#improvements)


## Motivations <a name="motivations"></a>

This project is part of the nanodegree [become a data scientist](https://eu.udacity.com/course/data-scientist-nanodegree--nd025) of [Udacity](https://eu.udacity.com/).

In this project, I am using a NLP pipeline with the NLTK package and a web app in Python.

## Packages used <a name="packages_used"></a>

- os
- numpy
- pandas
- re
- matplotlib
- pickle
- sqlalchemy
- logging
- nltk
- sklearn
- json
- plotly
- flask
- pathlib

## Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_response.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage or open your favorite browser and navigate to the URL `http://localhost:3001`.



## Files <a name="files"></a>

Here is the content of this repo:

```text

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs the app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- disaster_response.db   # database to save clean data to
|- process_data.py

- models
|- train_classifier.py
|- first_pipeline.pickle
|- gridsearch_pipeline.pickle
|- second_pipeline.pickle
|- preferred_pipeline.pickle

- notebooks
|- etl_pipeline.ipynb # Notebook to prepare cleaning function
|- ml_pipeline.ipynb # Notebook to test the ml algorithms
|- logs
| |- first_pipeline_log.txt
| |- gridsearch_pipeline_log.txt
| |- second_pipeline_log.txt

- LICENSE
- README.md
- .gitignore

```


## Possible improvements on this project: <a name="improvements"></a>


- Go into more detail about the dataset and your data cleaning and modeling process in your README file, add screenshots of your web app and model results.
- Add more visualizations to the web app.
- Based on the categories that the ML algorithm classifies text into, advise some organizations to connect to.
- Customize the design of the web app.
- Deploy the web app to a cloud service provider.
- Improve the efficiency of the code in the ETL and ML pipeline.
- This dataset is imbalanced (ie some labels like water have few examples). In your README, discuss how this imbalance, how that affects training the model, and your thoughts about emphasizing precision or recall for the various categories.