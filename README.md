# figure8_disaster_pipeline

## Table of contents

- [Motivations](#motivations)
- [Project description](#description)
- [Visuals](#visuals)
- [Packages used](#packages_used)
- [Instructions](#instructions)
- [Files](#files)
- [Possible improvements](#improvements)
- [Credits](#credits)
- [License](#license)
- [Status](#status)


## Motivations and Summary <a name="motivations"></a>

This project is part of the nanodegree [become a data scientist](https://eu.udacity.com/course/data-scientist-nanodegree--nd025) of [Udacity](https://eu.udacity.com/).

Summary of the project:

## Project description <a name="description"></a>

Following a disaster (flood etc), there are usually millions of communications related to this disaster. Either directly or by social media. This data, when collected, could be analysed to know at once which organisations need to be contacted to send help.

In this project, I am using a NLP pipeline with the NLTK package to categorise the emergency messages based on the needs of the person sending the text. Then, I created a web app that takes as inpu any text message and categorizes it.

The models were trained on real life disaster data provided by Figure Eight.


## Visuals <a name="visuals"></a>

Homepage when running the app:
![image](https://user-images.githubusercontent.com/29840762/210413647-d09afa49-8f88-4744-b7d2-13bb84b4fc07.png)


Example of model's output when testing the word "flood":
![image](https://user-images.githubusercontent.com/29840762/210413724-1a016bea-16cc-4a9d-9408-fa1e4a108b40.png)



## Packages used <a name="packages_used"></a>

- os
- sys
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


## Credits <a name="credits"></a>

This project is part of the Nanodegree [become a data scientist](https://eu.udacity.com/course/data-scientist-nanodegree--nd025) of [Udacity](https://eu.udacity.com/).


## License <a name="license"></a>

MIT License

Copyright (c) [2022] [Julie Ballard]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Project status  <a name="status"></a>

This project was reviewed and accepted during my Nanodegree.
