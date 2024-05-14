from flask import Flask, render_template, request, redirect, url_for, session, jsonify, json
from sqlalchemy import create_engine, Table, MetaData
from ts_testing import recommend_result
from web_profile import profile_result
from web_log import log_result
from web_admin_CM import CM_result
import hashlib
import os
from datetime import datetime


app = Flask(__name__)
app.secret_key = os.urandom(24)

# SQLAlchemy connect object create
db_uri = "mysql+mysqlconnector://root:rhks25714334@localhost:3306/recommend_sys"
engine = create_engine(db_uri)

conn = None
try:
    conn = engine.connect()
except exc.SQLAlchemyError as e:
    print('MySQL connect fail:', e)
else:
    print('MySQL connect success')

metadata = MetaData()
cs_info = Table('course_info', metadata, autoload_with=engine)
user_info = Table('user_info', metadata, autoload_with=engine)
user_profile = Table('user_profile', metadata, autoload_with=engine)
user_ratings = Table('user_rating', metadata, autoload_with=engine)
user_logs = Table('user_logs', metadata, autoload_with=engine)

def check_existing_user(username):
    with engine.connect() as con:
        existing_user = con.execute(user_info.select().where(user_info.c.ID == username)).fetchone()
        return existing_user is not None  

def register_user_info(username, hashed_password, age, gender, testname, aimscore, studystyle, level, topic):
    with engine.connect() as con:
        transaction = con.begin()
        try:
            con.execute(user_info.insert().values(ID=username, Password=hashed_password ))
            con.execute(user_profile.insert().values(Gender=gender, Age=age, PreferredCategory=testname, 
                                                     PreferredTopic=topic, PreferredStyle=studystyle, 
                                                     Level=level, AimScore=aimscore))
            transaction.commit()
            print('registeration success')
        except:
            transaction.rollback()
            print('registeration fail')
            raise

def update_user_profile(user_id, age, preferred_category, preferred_topic, preferred_style, level, aim_score):
    with engine.connect() as con:
        transaction = con.begin()
        try:
            con.execute(user_profile.update().where(user_profile.c.UserID == user_id).values(
                Age=age,
                PreferredCategory=preferred_category,
                PreferredTopic=preferred_topic,
                PreferredStyle=preferred_style,
                Level=level,
                AimScore=aim_score
            ))
            transaction.commit()
            print('Profile updated successfully')
        except:
            transaction.rollback()
            print('Failed to update profile')
            raise

def input_rs_result_rating(user_num, item_num, cs_title, cs_category, cs_topic, cs_style, cs_level, user_rating):
    with engine.connect() as con:
        transaction = con.begin()
        try:
            current_time = datetime.now().timestamp()
            con.execute(user_ratings.insert().values(UserID=user_num, CourseID=item_num, Rating=user_rating, Timestamp=current_time))

            user_pf = profile_result(session)
            row = user_pf.iloc[0]

            pf_category = row['PreferredCategory']
            pf_topic = row['PreferredTopic']
            pf_level = row['Level']
            pf_style = row['PreferredStyle']
            aimscore = row['AimScore']
            
            con.execute(user_logs.insert().values(user=user_num, item=item_num, rating=user_rating, cs_title=cs_title, 
                                                  cs_category=cs_category, cs_topic=cs_topic, cs_style=cs_style, cs_level=cs_level,
                                                  pf_category=pf_category, pf_topic=pf_topic, pf_style=pf_style, pf_level=pf_level,
                                                  aimscore=aimscore, timestamp=current_time))
            transaction.commit()
            print('recommend result rating success')
        except:
            transaction.rollback()
            print('recommend result rating fail')
            raise

def update_user_log(user_id, item, cs_title, preferred_category, preferred_topic, preferred_style, level, user_rating):
    with engine.connect() as con:
        transaction = con.begin()
        try:
            current_time = datetime.now().timestamp()
            con.execute(user_logs.update().where((user_logs.c.user == user_id) & (user_logs.c.item == item)).values(
                rating=user_rating,
                timestamp=current_time
            ))

            con.execute(user_ratings.update().where((user_ratings.c.UserID == user_id) & (user_ratings.c.CourseID == item)).values(
                Rating=user_rating,
                Timestamp=current_time
            ))

            transaction.commit()
            print('Rating updated successfully')
        except:
            transaction.rollback()
            print('Failed to update Rating')
            raise

def update_cs(item, cs_title, category, topic, style, level):
    with engine.connect() as con:
        transaction = con.begin()
        try:
            con.execute(cs_info.update().where(cs_info.c.cs_num == item).values(
                cs_category=category,
                cs_topic=topic,
                cs_level=level,
                cs_style=style
            ))

            transaction.commit()
            print('Course infomation updated successfully')
        except:
            transaction.rollback()
            print('Failed to update Course infomation')
            raise

def add_cs(title, category, topic, style, level):
    with engine.connect() as con:
        transaction = con.begin()
        try:
            con.execute(cs_info.insert().values(cs_title=title, cs_category=category, cs_topic=topic, cs_level=level, cs_style=style ))

            transaction.commit()
            print('Course infomation add successfully')
        except:
            transaction.rollback()
            print('Failed to add Course infomation')
            raise

def del_cs(item):
    with engine.connect() as con:
        transaction = con.begin()
        try:
            con.execute(cs_info.delete().where(cs_info.c.cs_num == item))

            transaction.commit()
            print('Course infomation delete successfully')
        except:
            transaction.rollback()
            print('Failed to delete Course infomation')
            raise        

@app.route('/')
def init_page():
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = hashlib.sha256(password.encode()).hexdigest()  #test123
        age= request.form['age']
        gender= request.form['gender']
        testname= request.form['testname']
        aimscore= request.form['aimscore']
        studystyle = request.form['style']
        level = request.form['level']
        selected_topics = request.form.getlist('topic')
        topic = ','.join(selected_topics)

        if check_existing_user(username):
            return render_template('register_fail.html')
        elif username == 0:
            return 'Please put in your ID'
        else:
            register_user_info(username, hashed_password, age, gender, testname, aimscore, studystyle, level, topic)
            return render_template('register_success.html')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        with engine.connect() as con:
            user = con.execute(user_info.select().where(user_info.c.ID == username, user_info.c.Password == hashed_password)).fetchone()
        if user:
                       
            session['user_num'] = user[0]
            session['user_id'] = user[1]
            print(session['user_id'])

            if(session['user_id']=="admin"):
                return redirect(url_for('admin_main'))
            else:
                return redirect(url_for('main'))        
        else:
            return 'Invalid User ID or Password'

    return render_template('login.html')

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop('user_num', None)  # user_num 세션 삭제
    session.pop('user_id', None)  # user_id 세션 삭제
    return redirect(url_for('login'))

@app.route('/admin_main')
def admin_main():
    if 'user_num' in session:
        user_num = session.get('user_num')
        user_id = session.get('user_id')
        return render_template('admin_main.html', user_num=user_num, user_id=user_id, server_status=json.dumps({'status': 0}))
    else:
        return redirect(url_for('login'))  # 로그인되지 않은 경우 로그인 페이지로 리디렉션
    
@app.route('/admin_home')
def admin_home():
    return render_template('admin_home.html')

@app.route('/admin_CM')
def admin_CM():
    cm_results = CM_result()
    return render_template('admin_CM.html',cm_result=cm_results)

@app.route('/admin_CM_md', methods=['GET', 'POST'])
def admin_CM_md():
    if request.method == 'POST':
        cs_num = request.form['modal_item_id']
        cs_title = request.form['modal_cs_title_input']
        cs_category = request.form['modal_cs_category']
        cs_level = request.form['modal_cs_level']
        cs_style = request.form['modal_cs_style']
        selected_topics = request.form.getlist('topic')
        topic = ','.join(selected_topics)

        update_cs(cs_num, cs_title, cs_category, topic, cs_style, cs_level)
        return render_template('admin_main.html', server_status=json.dumps({'status': 1}))
    return 'This route is only for POST requests.', 405

@app.route('/admin_CM_add', methods=['GET', 'POST'])
def admin_CM_add():
    if request.method == 'POST':
        cs_title = request.form['modal_cs_title']
        cs_category = request.form['modal_cs_category']
        cs_level = request.form['modal_cs_level']
        cs_style = request.form['modal_cs_style']
        selected_topics = request.form.getlist('topic')
        topic = ','.join(selected_topics)

        add_cs(cs_title, cs_category, topic, cs_style, cs_level)
        return render_template('admin_main.html', server_status=json.dumps({'status': 2}))
    return 'This route is only for POST requests.', 405

@app.route('/admin_CM_del', methods=['GET', 'POST'])
def admin_CM_del():
    if request.method == 'POST':
        cs_num = request.form['del_modal_item_id']
        print(cs_num)
        del_cs(cs_num)
        return render_template('admin_main.html', server_status=json.dumps({'status': 3}))
    return 'This route is only for POST requests.', 405


@app.route('/main')
def main():
    if 'user_num' in session:
        user_num = session.get('user_num')
        user_id = session.get('user_id')
        return render_template('main.html', user_num=user_num, user_id=user_id, server_status=json.dumps({'status': 0}))
    else:
        return redirect(url_for('login'))  # 로그인되지 않은 경우 로그인 페이지로 리디렉션
    
@app.route('/home')
def home():
    if 'user_num' in session:
        user_num = session.get('user_num')
        user_id = session.get('user_id')
        
        return render_template('home.html', user_num=user_num, user_id=user_id)
    
@app.route('/profile')
def profile():
    if 'user_num' in session:
        pf_result = profile_result(session)
        return render_template('user_profile.html', pf_result=pf_result)
    
@app.route('/updata_profile', methods=['GET', 'POST'])
def updata_profile():
    if request.method == 'POST':
        user_id = request.form['user_id']
        age = request.form['age']
        testname= request.form['testname']
        aimscore= request.form['aimscore']
        studystyle = request.form['style']
        level = request.form['level']
        selected_topics = request.form.getlist('topic')
        topic = ','.join(selected_topics)

        update_user_profile(user_id, age, testname, topic, studystyle, level, aimscore)
        
        return render_template('main.html', server_status=json.dumps({'status': 1}))
    return 'This route is only for POST requests.', 405

@app.route('/update_success')
def update_success():
    return render_template('update_success.html')

@app.route('/recommend')
def recommend():
    if 'user_num' in session:
        user_num = session.get('user_num')
        user_id = session.get('user_id')
        rs_result = recommend_result(session)
        return render_template('recommend_course.html', rs_result=rs_result, user_num=user_num, user_id=user_id)
    
@app.route('/submit_rating', methods=['GET', 'POST'])
def submit_rating():
    try:
        user_num = request.form['user_id']
        item_num = request.form['modal_item_id']
        cs_title = request.form['modal_cs_title_input']
        cs_category = request.form['modal_cs_category_input']
        cs_topic = request.form['modal_cs_topic_input']
        cs_style = request.form['modal_cs_style_input']
        cs_level = request.form['modal_cs_level_input']
        user_rating = int(request.form['rating'])
        
        input_rs_result_rating(user_num, item_num, cs_title, cs_category, cs_topic, cs_style, cs_level, user_rating)
        return render_template('main.html', server_status=json.dumps({'status': 2}))
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/log')
def log():
    if 'user_num' in session:
        user_num = session.get('user_num')
        user_id = session.get('user_id')
        log_results = log_result(session)
        return render_template('user_log.html', log_results=log_results, user_num=user_num, user_id=user_id)
        
@app.route('/log_update', methods=['GET', 'POST'])
def log_update():
    try:
        user_num = request.form['user_id']
        item_num = request.form['rc_modal_item_id']
        cs_title = request.form['rc_modal_cs_title_input']
        cs_category = request.form['rc_modal_cs_category_input']
        cs_topic = request.form['rc_modal_cs_topic_input']
        cs_style = request.form['rc_modal_cs_style_input']
        cs_level = request.form['rc_modal_cs_level_input']
        user_rating = int(request.form['rating'])
        
        update_user_log(user_num, item_num, cs_title, cs_category, cs_topic, cs_style, cs_level, user_rating)
        return render_template('main.html', server_status=json.dumps({'status': 3}))
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    

if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port="5000")