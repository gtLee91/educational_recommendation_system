import pandas as pd
from sqlalchemy import create_engine

def profile_result(session):
    db_uri = "mysql+mysqlconnector://root:rhks25714334@localhost:3306/recommend_sys"
    engine = create_engine(db_uri)

    user_num = session.get('user_num')

    query_profile = f"SELECT UserID, Gender, Age, PreferredCategory, PreferredTopic, Level, PreferredStyle, AimScore FROM recommend_sys.user_profile WHERE UserID = {user_num}"
    user_profile = pd.read_sql(query_profile, engine)

    return user_profile