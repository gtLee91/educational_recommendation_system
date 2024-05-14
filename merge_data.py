import pandas as pd
from sqlalchemy import create_engine

# MYSQL version
# SQLAlchemy connect object create
db_uri = "mysql+mysqlconnector://root:rhks25714334@localhost:3306/recommend_sys"
engine = create_engine(db_uri)

# SQL query course_info
query_course = "SELECT cs_num, cs_title, cs_category, cs_topic, cs_level, cs_style FROM recommend_sys.course_info"
cs_data = pd.read_sql(query_course, engine)

cs_data.rename(columns={'cs_num': 'item'}, inplace=True)

query_profile = "SELECT UserID, Gender, Age, PreferredCategory, PreferredTopic, Level, PreferredStyle, AimScore FROM recommend_sys.user_profile"
user_profile = pd.read_sql(query_profile, engine)

user_profile.rename(columns={'UserID': 'user'}, inplace=True)
user_profile.rename(columns={'PreferredCategory': 'pf_category'}, inplace=True)
user_profile.rename(columns={'PreferredTopic': 'pf_topic'}, inplace=True)
user_profile.rename(columns={'Level': 'pf_level'}, inplace=True)
user_profile.rename(columns={'PreferredStyle': 'pf_style'}, inplace=True)
user_profile.rename(columns={'AimScore': 'aimscore'}, inplace=True)
user_profile.rename(columns={'Gender': 'gender'}, inplace=True)
user_profile.rename(columns={'Age': 'age'}, inplace=True)

preferred_style_counts = user_profile['pf_category'].value_counts()
#print(preferred_style_counts)
#print(user_profile.head(10))

query_rating = "SELECT UserID, Rating, CourseID, Timestamp FROM recommend_sys.user_rating"
user_rating = pd.read_sql(query_rating, engine)
user_rating.rename(columns={'UserID': 'user'}, inplace=True)
user_rating.rename(columns={'Rating': 'rating'}, inplace=True)
user_rating.rename(columns={'Timestamp': 'timestamp'}, inplace=True)
user_rating.rename(columns={'CourseID': 'item'}, inplace=True)

#print(user_rating.head(10))

merged_df = pd.merge(cs_data, user_rating, on='item', how='inner')
merged_df = pd.merge(merged_df, user_profile, on='user', how='inner')

desired_order = ['user','item','rating','cs_title','cs_category','cs_topic','cs_style','cs_level','pf_category','pf_topic','pf_style','pf_level','aimscore','timestamp']
merged_df = merged_df[desired_order]

merged_df = merged_df.sort_values(["user","rating"]).fillna(0)
merged_df = merged_df.reset_index(drop=True)
#print(merged_df.head(20))

query_log = "SELECT * FROM recommend_sys.user_logs"
seq_merged_df = pd.read_sql(query_log, engine)

seq_merged_df = pd.concat([seq_merged_df, merged_df], ignore_index=True)
seq_merged_df = seq_merged_df[desired_order]

seq_merged_df = seq_merged_df.sort_values(by=['user', 'timestamp', 'item' ])
seq_merged_df = seq_merged_df.reset_index(drop=True)
#print(seq_merged_df.head(40))
#print(seq_merged_df.tail(40))

#--------------  
'''
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
  
#desired_order = ['user','item','rating','cs_title','cs_category','cs_topic','cs_style','cs_level','pf_category','pf_topic','pf_style','pf_level','aimscore', 'age','gender','timestamp']
desired_order = ['user','item','rating','cs_title','cs_category','cs_topic','cs_style','cs_level','pf_category','pf_topic','pf_style','pf_level','aimscore','timestamp']
merged_df = merged_df[desired_order]
  
merged_df = merged_df.sort_values(["user","rating"]).fillna(0)
merged_df = merged_df.reset_index(drop=True)
#print(merged_df.head(20))


seq_merged_df = pd.read_csv('data/user_ch_profiles_ratings_log.csv')
seq_merged_df = pd.concat([seq_merged_df, merged_df], ignore_index=True)
seq_merged_df = seq_merged_df[desired_order]

seq_merged_df = seq_merged_df.sort_values(by=['user', 'timestamp', 'item' ])
seq_merged_df = seq_merged_df.reset_index(drop=True)
#print(seq_merged_df.head(40))
#print(seq_merged_df.tail(40))
'''