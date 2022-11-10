# figure8_disaster_pipeline
Project for DS course to create a pipeline analysing figure 8 data


## Table of contents

- [Packages used](#packages_used)
- [Instructions](#instructions)
- [Motivations](#motivations)
- [Files](#files)


## Packages used

...

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_response.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage or open your favorite browser and navigate to the URL `http://localhost:3001`.



## Motivations

This project is part of the nanodegree [become a data scientist](https://eu.udacity.com/course/data-scientist-nanodegree--nd025) of [Udacity](https://eu.udacity.com/).

In this project, I am using a NLP pipeline with the NLTK package and a web app in Python.



## Files

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


- notebooks
|- etl_pipeline.ipynb # Notebook to prepare cleaning function
|- ml_pipeline.ipynb # Notebook to test the ml algorithms

- LICENSE
- README.md
- .gitignore

```


## Notes
There are three components you'll need to complete for this project.
### ETL Pipeline
In a Python script, process_data.py, write a data cleaning pipeline that:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

### ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

### Flask Web App
We are providing much of the flask web app for you, but feel free to add extra features depending on your knowledge of flask, html, css and javascript. For this part, you'll need to:

- Modify file paths for database and model as needed
- Add data visualizations using Plotly in the web app. One example is provided for you





### Suggestions to Make Your Project Stand Out!
Go into more detail about the dataset and your data cleaning and modeling process in your README file, add screenshots of your web app and model results.
Add more visualizations to the web app.
Based on the categories that the ML algorithm classifies text into, advise some organizations to connect to.
Customize the design of the web app.
Deploy the web app to a cloud service provider.
Improve the efficiency of the code in the ETL and ML pipeline.
This dataset is imbalanced (ie some labels like water have few examples). In your README, discuss how this imbalance, how that affects training the model, and your thoughts about emphasizing precision or recall for the various categories.