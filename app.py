from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, json
from chat import get_response
from flask_cors import CORS
import cv2
import os
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from flask_mysqldb import MySQL
import MySQLdb.cursors

#### Defining Flask App

app = Flask(__name__)
CORS(app)
  
app.secret_key = 'pakistan'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'online_voting_system'
  
mysql = MySQL(app)
  


# HOME SECTION
@app.route("/")
def index_get():
    return render_template("base.html")

# VOTER SECTION
@app.route("/voter", methods=['GET', 'POST'])
def voter():
    return render_template("voter/voterDashboard.html")

@app.route("/register_voter", methods=['GET', 'POST'])
def register_voter():
    mesage = ''
    if request.method == "POST":
        name = request.form.get('name')
        cnic = request.form.get('cnic')
        dob = datetime.strptime(request.form.get('dob'),'%Y-%m-%d')
        password = request.form.get('password')
        confirmPassword = request.form.get('confirm_password')
        today = date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        if age < 18:
            flash("Warning! AGE IS LESS THAN 18")
    
        elif age >= 18:
            if password == confirmPassword:
                cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                cursor.execute('SELECT * FROM voters WHERE cnic = %s', (cnic, ))
                account = cursor.fetchone()
                if account:
                    flash('Account already exists !')
                elif not name or not password or not cnic:
                    flash('Please fill out the form !')
                else:
                  

                    userimagefolder = 'static/faces/'+cnic
                    if not os.path.isdir(userimagefolder):
                        os.makedirs(userimagefolder)
                    cap = cv2.VideoCapture(0)
                    i,j = 0,0
                    while 1:
                        _,frame = cap.read()
                        faces = extract_faces(frame)
                        for (x,y,w,h) in faces:
                            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
                            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                            if j%10==0:
                                newCnic = cnic+'_'+str(i)+'.jpg'
                                cv2.imwrite(userimagefolder+'/'+newCnic,frame[y:y+h,x:x+w])
                                i+=1
                            j+=1
                        if j==500:
                            break
                       
                        cv2.imshow('Adding new User',frame)
                        if cv2.waitKey(1)==27:
                            break

                    locationPic = 'static/faces/'+cnic+'/'+cnic+'_3.jpg'
                    cursor.execute('INSERT INTO voters(name,pic,cnic,age,password, voted) VALUES (%s,%s, %s, %s, %s, %s)', (name,locationPic, cnic,age, password, 0))
                    mysql.connection.commit()
                    cursor.close()
                    cap.release()
                    cv2.destroyAllWindows()
                    print('Training Model')
                    train_model()
                    names,rolls,times,l = extract_attendance()    
                    flash('You have successfully registered !')
                    return render_template('voter/registerVoter.html', mesage = mesage)
       
    
    return render_template('voter/registerVoter.html', mesage = mesage)
           

@app.route("/login_voter", methods=['GET', 'POST'])
def login_voter():
    message = ''
    if request.method == "POST":
        cnic = request.form.get('cnic')
        password = request.form.get('password')
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM voters WHERE cnic = %s AND password = %s', (cnic, password))
        user = cursor.fetchone()
        
        if user:
            session['loggedin'] = True
            session['ID'] = user['ID']
            session['name'] = user['name']
            session['cnic'] = user['cnic']
            session['age'] = user['age']
            session['pic'] = user['pic']
            if 'face_recognition_model.pkl' not in os.listdir('static'):
                return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2(),mess='There is no trained model in the static folder. Please add a new face to continue.') 

            cap = cv2.VideoCapture(0)
            ret = True
            while ret:
                ret,frame = cap.read()
                if extract_faces(frame)!=():
                    (x,y,w,h) = extract_faces(frame)[0]
                    cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
                    face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
                    identified_person = identify_face(face.reshape(1,-1))[0]
                    add_attendance(identified_person)
                    cv2.putText(frame,f'{identified_person}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                cv2.imshow('LOGIN',frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
            names,rolls,times,l = extract_attendance()    
            return redirect(url_for("voter"))

     
        else:
            flash('Please enter correct email / password !')

       
    return render_template('voter/loginVoter.html', message = message)

@app.route("/voteCast")
def voteCast():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM candidates')
    data = cursor.fetchall()
    cursor.close()
   

    return render_template("voter/voteCast.html", candidates=data)

@app.route("/voted", methods=["GET","POST"])
def voted():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    
    if request.method == "POST":
        cursor.execute('SELECT * FROM candidates')
        mysql.connection.commit()
        data = cursor.fetchall()
        candidateMame = request.form.getlist("vote")[0]
        voterName = session['name']
        cursor.execute('SELECT * FROM voters WHERE name = %s', (voterName,))
        user = cursor.fetchone()

        if user["voted"] == 0:
            cursor.execute('UPDATE votes SET number_of_votes = number_of_votes+1 WHERE candidate_name = %s', (candidateMame,))
            cursor.execute('UPDATE voters SET voted = %s WHERE name = %s', (1, voterName,))
            cursor.execute('UPDATE candidates SET voted = %s WHERE name = %s', (1, voterName,))
            mysql.connection.commit()
            cursor.close()
            flash("VOTED SUCCESSFULLY!")
           
        else:
            if len(request.form.getlist("vote")) > 1:
                flash("PLEASE SELECT ONLY ONE CANDIDATE")

            if len(request.form.getlist("vote")) == 0:
                flash("PLEASE SELECT A CANDIDATE")

        
             
    flash("ERROR! CAN NOT VOTE MORE THAN ONES")
         
    return render_template("voter/voteCast.html", candidates=data)

   
  

@app.route("/logout")
def logout():
    session.pop("voter", None)
    return redirect(url_for("index_get"))  

@app.route("/stats")
def stats():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM votes')
    mysql.connection.commit()
    data = cursor.fetchall()

    return render_template("stats.html", candidates=data)  



# CANDIDATE SECTION
@app.route("/candidate", methods=['GET', 'POST'])
def candidate():
    return render_template("candidate/candidateDashboard.html")

@app.route("/register_candidate", methods=['GET', 'POST'])
def register_candidate():
    mesage = ''
    if request.method == "POST":
        name = request.form.get('name')
        cnic = request.form.get('cnic')
        password = request.form.get('password')
        confirmPassword = request.form.get('confirm_password')
        partySymbol = request.form.get('party_symbol')
        dob = datetime.strptime(request.form.get('dob'),'%Y-%m-%d')
        
        today = date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

        if age < 18:
            flash("Warning! AGE IS LESS THAN 18")
    
        elif age >= 18:

            if password == confirmPassword:
                cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                cursor.execute('SELECT * FROM candidates WHERE cnic = (%s)', (cnic,))
                account = cursor.fetchone()
                if account:
                    flash('Account already exists !')
                elif not name or not password or not cnic:
                   flash('Please fill out the form !')
                else:
                    userimagefolder = 'static/faces/'+cnic
                    if not os.path.isdir(userimagefolder):
                        os.makedirs(userimagefolder)
                    cap = cv2.VideoCapture(0)
                    i,j = 0,0
                    while 1:
                        _,frame = cap.read()
                        faces = extract_faces(frame)
                        for (x,y,w,h) in faces:
                            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
                            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                            if j%10==0:
                                newCnic = cnic+'_'+str(i)+'.jpg'
                                cv2.imwrite(userimagefolder+'/'+newCnic,frame[y:y+h,x:x+w])
                                i+=1
                            j+=1
                        if j==500:
                            break
                        cv2.imshow('REGISTRATION',frame)
                        if cv2.waitKey(1)==27:
                            break

                    locationPic = 'static/faces/'+cnic+'/'+cnic+'_3.jpg'
                    cursor.execute('INSERT INTO candidates(name,pic,cnic,age,party_symbol,password,voted) VALUES (%s,%s, %s,%s,%s, %s, %s)', (name,locationPic,cnic,age,partySymbol,password, 0))
                    cursor.execute('INSERT INTO votes(pic, candidate_name,party_symbol,number_of_votes) VALUES (%s,%s, %s, %s)', (locationPic, name,partySymbol,0))
                    mysql.connection.commit()
                    cursor.close()
                    cap.release()
                    cv2.destroyAllWindows()
                    print('Training Model')
                    train_model()
                    names,rolls,times,l = extract_attendance()    
                    flash('You have successfully registered !')
                    return render_template('candidate/registerCandidate.html', mesage = mesage)
                    

    return render_template('candidate/registerCandidate.html', mesage = mesage)
           
           

@app.route("/login_candidate", methods=['GET', 'POST'])
def login_candidate():
    message = ''
    if request.method == "POST":
        cnic = request.form.get('cnic')
        password = request.form.get('password')
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM candidates WHERE cnic = %s AND password = %s', (cnic, password))
        user = cursor.fetchone()
        if user:
            session['loggedin'] = True
            session['ID'] = user['ID']
            session['name'] = user['name']
            session['cnic'] = user['cnic']
            session['pic'] = user['pic']
            if 'face_recognition_model.pkl' not in os.listdir('static'):
                return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2(),mess='There is no trained model in the static folder. Please add a new face to continue.') 

            cap = cv2.VideoCapture(0)
            ret = True
            while ret:
                ret,frame = cap.read()
                if extract_faces(frame)!=():
                    (x,y,w,h) = extract_faces(frame)[0]
                    cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
                    face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
                    identified_person = identify_face(face.reshape(1,-1))[0]
                    add_attendance(identified_person)
                    cv2.putText(frame,f'{identified_person}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                cv2.imshow('LOGIN',frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
            names,rolls,times,l = extract_attendance()    
            return redirect(url_for("candidate"))

        else:
            return 'Please enter correct email / password !'
        
    return render_template('candidate/loginCandidate.html', message = message)



#FACE RECOGNITION

#### Saving Date today in 2 different formats
def datetoday():
    return date.today().strftime("%m_%d_%y")
def datetoday2():
    return date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday()}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday()}.csv','w') as f:
        f.write('Name,Roll,Time')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points


#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l


#### Add Attendance of a specific user
def add_attendance(cnic):
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
    with open(f'Attendance/Attendance-{datetoday()}.csv','a') as f:
        f.write(f'\n{cnic},{current_time}')


################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/face_recognition')
def home():
    names,rolls,times,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2()) 




#### This function will run when we add a new user

#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)

