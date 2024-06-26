# educational_recommendation_system

## Introduction
This project utilises meta-learning, deep learning and reinforcement learning to provide personalised course recommendations based on user data such as preferred exams, topics, target scores, and current proficiency levels.

## Install libraries
* Python 3.8+
* PyTorch 2.0.1+
* pandas 2.0.3
* sqlalchemy 2.0.21
* matplotlib 3.7.3
* sklearn 1.3.0

## folder & file description
**data**
- This folder gathers CSV files that have been generated by converting user profiles, user profile ratings logs, course information, and rating data used in the recommendation system.

**rs_system_DB**
- The rs_system_DB folder contains the recommender system database SQL file, so you need to import it into your database to run the web application.

**templates**
- This folder is used by web applications to store HTML template files.

**result_img**
- This folder contains collected graph images depicting the results of model training and evaluation.

**trained_model**
- This folder collects pth files containing the states of trained models and optimization states.

**merge_data.py**
- This code is responsible for retrieving user profiles, course information, and rating data from the database and performing the task of merging them into a unified dataset.

**data_pre_process.py**
- This code preprocesses the merged data.

**sequence_dataset.py**
- This code transforms the preprocessed data into sequence data.

**MAML_model.py**
- This code defines the MAML model.

**MAML_part.py**
- This code is trained and generalised using the MAML algorithm. after that, this code evaluates and saves the model.

**DRL_model.py**
- This code defines a neural network model with an LSTM-based Actor-Critic architecture.

**environment.py**
- This code defines an 'Environment' class to model a reinforcement learning environment

**train.py**
- This code performs the training of a Deep Reinforcement Learning (DRL) model.

**evaluation.py**
- This code evaluates a trained Deep Reinforcement Learning (DRL) model.

**testing.py, testing_env.py, testing_sq_dataset.py**
- These codes are created to conduct cold-start tests.

**web_app_run.py**
- This code is required when running a web application.

**web_profile.py, web_log.py, web_admin_CM.py**
- This code fetches data from the database and sends it to a web application.

**ts_testing.py**
- This code fetches trained models and creates a personalised lecture recommendation list for the user.

## How to import SQL files

1. MySQL Installation: First, you need to install the MySQL Database Management System. You can download it from the official website of MySQL (https://dev.mysql.com/downloads/). After installation, start the MySQL server.

2. Database Creation: Connect to the MySQL server and create a database. Typically, you can perform this task using GUI tools like MySQL Workbench.<br>
![MySQL connection](images/setup1.png)

3. click the 'Data import/Restore' in the Administration window.<br>
![MySQL Data import](images/setup2.png)

4. Select 'Import from Self-Contained File' option and select SQL file.<br>
![MySQL Data import](images/setup3.png)

5. Select schema from 'Default Target Schema'; if there is no existing Schema, click the 'new' button to create a new schema.<br>
![MySQL Data import](images/setup4.png)

6. Finally, click the 'start import' button.<br>
![MySQL Data import](images/setup5.png)

![MySQL Data import](images/setup6.png)

## Attention
To actually use this code for the recommendation system, you need to input your MYSQL server connection information into the "db_uri" variable in the format of <br>
**"mysql+mysqlconnector://(username):(password)@(hostname):(port number)/(Schema name)" <br>**
as shown in the diagram below. <br>
**You must modify the "db_uri" variable in the files 'merge_data.py', 'MAML_part.py', 'ts_testing.py', 'web_app_run.py', 'web_profile.py', 'web_log.py', and 'web_admin_CM.py'.<br>**

![MySQL Data import](images/attention.png)

## if you have data load problem
If you cannot access the database and retrieve the data, you can load the data from a CSV file.
If you replace the code in merge_data.py with the following, you can load the data from a CSV file.
**However, when you load a CSV file, the web application's features and some systems that use MYSQL will not work.**

```
  import pandas as pd
  
  cs_data_csv = pd.read_csv('data/course_information.csv')
  cs_data = pd.DataFrame(cs_data_csv)
  cs_data = cs_data.drop(['cs_content'], axis=1)
  cs_data.rename(columns={'cs_num': 'item'}, inplace=True)
    
  cs_rating = pd.read_csv('data/user_ratings.csv')
  cs_rating_df = pd.DataFrame(cs_rating)
  cs_rating_df.rename(columns={'UserID': 'user'}, inplace=True)
  cs_rating_df.rename(columns={'Rating': 'rating'}, inplace=True)
  cs_rating_df.rename(columns={'Timestamp': 'timestamp'}, inplace=True)
  cs_rating_df.rename(columns={'CourseID': 'item'}, inplace=True)
    
  user_profile = pd.read_csv('data/user_profiles.csv')
  user_profile_df = pd.DataFrame(user_profile)
  user_profile_df.rename(columns={'UserID': 'user'}, inplace=True)
  user_profile_df.rename(columns={'PreferredCategory': 'pf_category'}, inplace=True)
  user_profile_df.rename(columns={'PreferredTopic': 'pf_topic'}, inplace=True)
  user_profile_df.rename(columns={'Level': 'pf_level'}, inplace=True)
  user_profile_df.rename(columns={'PreferredStyle': 'pf_style'}, inplace=True)
  user_profile_df.rename(columns={'AimScore': 'aimscore'}, inplace=True)
  user_profile_df.rename(columns={'Gender': 'gender'}, inplace=True)
  user_profile_df.rename(columns={'Age': 'age'}, inplace=True)
    
  merged_df = pd.merge(cs_data, cs_rating_df, on='item', how='inner')
  merged_df = pd.merge(merged_df, user_profile_df, on='user', how='inner')
    
  desired_order = ['user','item','rating','cs_title','cs_category','cs_topic','cs_style','cs_level','pf_category','pf_topic','pf_style','pf_level','aimscore','timestamp']
  merged_df = merged_df[desired_order]
    
  merged_df = merged_df.sort_values(["user","rating"]).fillna(0)
  merged_df = merged_df.reset_index(drop=True)
  
  seq_merged_df = pd.read_csv('data/user_ch_profiles_ratings_log.csv')
  seq_merged_df = pd.concat([seq_merged_df, merged_df], ignore_index=True)
  seq_merged_df = seq_merged_df[desired_order]
  
  seq_merged_df = seq_merged_df.sort_values(by=['user', 'timestamp', 'item' ])
  seq_merged_df = seq_merged_df.reset_index(drop=True)

```



## Training and Evaluation MAML model
```
  %run MAML_part.py
```

## Training DRL model
```
  %run train.py
```

## Evaluation DRL model
```
  %run evaluation.py
```

## cold-start test
```
  %run testing.py
```

## web application running
```
  %run web_app_run.py
```
