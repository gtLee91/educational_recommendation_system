import pandas as pd
from sqlalchemy import create_engine

def log_result(session):
    db_uri = "mysql+mysqlconnector://root:rhks25714334@localhost:3306/recommend_sys"
    engine = create_engine(db_uri)

    user_num = session.get('user_num')

    query_log = f"SELECT item, rating, cs_title, cs_category, cs_topic, cs_style, cs_level FROM recommend_sys.user_logs WHERE user = {user_num}"
    rating_log = pd.read_sql(query_log, engine)

    return rating_log