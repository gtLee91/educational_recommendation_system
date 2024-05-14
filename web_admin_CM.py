import pandas as pd
from sqlalchemy import create_engine

def CM_result():
    db_uri = "mysql+mysqlconnector://root:rhks25714334@localhost:3306/recommend_sys"
    engine = create_engine(db_uri)

    query_profile = f"SELECT cs_num, cs_title, cs_category, cs_topic, cs_level, cs_style FROM recommend_sys.course_info"
    cs_info = pd.read_sql(query_profile, engine)

    return cs_info