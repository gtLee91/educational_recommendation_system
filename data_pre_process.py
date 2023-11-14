import pandas as pd
from sklearn.preprocessing import LabelEncoder
from merge_data import merged_df


def pre_process(df):

    cs_category_values = ['ielts', 'pte']
    cs_topic_values = ['writing', 'speaking', 'reading', 'listening', 'vocabulary', 'grammar']
    pf_category_values = cs_category_values
    pf_topic_values = cs_topic_values

    for category in cs_category_values:
        df[f'cs_category_{category}'] = 0

    for topic in cs_topic_values:
        df[f'cs_topic_{topic}'] = 0

    for category in pf_category_values:
        df[f'pf_category_{category}'] = 0

    for topic in pf_topic_values:
        df[f'pf_topic_{topic}'] = 0

    df['cs_category_ielts'] = df['cs_category'].apply(lambda x: 1 if 'ielts' in x else 0)
    df['cs_category_pte'] = df['cs_category'].apply(lambda x: 1 if 'pte' in x else 0)

    df['pf_category_ielts'] = df['pf_category'].apply(lambda x: 1 if 'ielts' in x else 0)
    df['pf_category_pte'] = df['pf_category'].apply(lambda x: 1 if 'pte' in x else 0)

    topics = ['writing', 'speaking', 'reading', 'listening', 'vocabulary', 'grammar']
    for topic in topics:
        df[f'cs_topic_{topic}'] = df['cs_topic'].apply(lambda x: 1 if topic in x else 0)
        df[f'pf_topic_{topic}'] = df['pf_topic'].apply(lambda x: 1 if topic in x else 0)

    def generate_age(x):
        if x < 10:
            return 9
        elif x < 20:
            return 19
        elif x < 30:
            return 29
        elif x < 40:
            return 39
        elif x < 50:
            return 49
        else:
            return 59

    df['age'] = df['age'].apply(generate_age)

    label_encoder = LabelEncoder()
    # incoding
    df['cs_style'] = label_encoder.fit_transform(df['cs_style'])
    df['cs_level'] = label_encoder.fit_transform(df['cs_level'])
    if df['pf_style'].nunique() > 1:
        df['pf_style'] = label_encoder.fit_transform(df['pf_style'])
    else:
        df['pf_style'].replace({'example': 0, 'explanation': 1}, inplace=True)
    
    if df['pf_level'].nunique() > 1:
        df['pf_level'] = label_encoder.fit_transform(df['pf_level'])
    else:
        df['pf_level'].replace({'advanced': 0, 'beginner': 1, 'intermediate': 2}, inplace=True)
    df['gender'] = label_encoder.fit_transform(df['gender'])
    df['aimscore'] = label_encoder.fit_transform(df['aimscore'])
    df['age'] = label_encoder.fit_transform(df['age'])

    #Delete Unnecessary Columns
    df.drop('cs_category', axis=1, inplace=True)
    df.drop('cs_topic', axis=1, inplace=True)
    df.drop('pf_category', axis=1, inplace=True)
    df.drop('pf_topic', axis=1, inplace=True)
    if 'timestamp' in df.columns:
        df.drop('timestamp', axis=1, inplace=True)

    #print(df.head(10))
    return df

df = pre_process(merged_df)
#print(df.head(10))