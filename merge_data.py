import pandas as pd
from sqlalchemy import create_engine

# SQLAlchemy connect object create
db_uri = "mysql+mysqlconnector://root:rhks25714334@localhost:3306/recommend_sys"
engine = create_engine(db_uri)

# SQL query course_info
query_course = "SELECT cs_num, cs_title, cs_category, cs_topic, cs_level, cs_style FROM recommend_sys.course_info"
cs_data = pd.read_sql(query_course, engine)

cs_data.rename(columns={'cs_num': 'item'}, inplace=True)
#print(cs_data.head(10))

query_profile = "SELECT UserID, Gender, Age, PreferredCategory, PreferredTopic, Level, PreferredStyle, AimScore FROM recommend_sys.user_profiles"
user_profile = pd.read_sql(query_profile, engine)

user_profile.rename(columns={'UserID': 'user'}, inplace=True)
user_profile.rename(columns={'PreferredCategory': 'pf_category'}, inplace=True)
user_profile.rename(columns={'PreferredTopic': 'pf_topic'}, inplace=True)
user_profile.rename(columns={'Level': 'pf_level'}, inplace=True)
user_profile.rename(columns={'PreferredStyle': 'pf_style'}, inplace=True)
user_profile.rename(columns={'AimScore': 'aimscore'}, inplace=True)
user_profile.rename(columns={'Gender': 'gender'}, inplace=True)
user_profile.rename(columns={'Age': 'age'}, inplace=True)

#print(user_profile.head(10))

query_rating = "SELECT UserID, Rating, CourseID, Timestamp FROM recommend_sys.user_ratings"
user_rating = pd.read_sql(query_rating, engine)
user_rating.rename(columns={'UserID': 'user'}, inplace=True)
user_rating.rename(columns={'Rating': 'rating'}, inplace=True)
user_rating.rename(columns={'Timestamp': 'timestamp'}, inplace=True)
user_rating.rename(columns={'CourseID': 'item'}, inplace=True)

#print(user_rating.head(10))

merged_df = pd.merge(cs_data, user_rating, on='item', how='inner')
merged_df = pd.merge(merged_df, user_profile, on='user', how='inner')

desired_order = ['user','item','rating','cs_title','cs_category','cs_topic','cs_style','cs_level','pf_category','pf_topic','pf_style','pf_level','aimscore', 'age','gender','timestamp']
merged_df = merged_df[desired_order]

merged_df = merged_df.sort_values(["user","rating"]).fillna(0)
merged_df = merged_df.reset_index(drop=True)
#print(merged_df.head(30))
