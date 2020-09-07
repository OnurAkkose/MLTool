from flask import Flask,render_template,request,redirect,flash,Response
import pandas as pd
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import exp, cos, linspace
from matplotlib.figure import Figure
import os, re
import numpy as np
from matplotlib.figure import Figure
import secrets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
app = Flask(__name__)
secret = secrets.token_urlsafe(32)

app.secret_key = secret
csv_file = None
df = None
columns_all = None
x = None
datas = None
bagimli_degisken = None
test = None
algorithm = None



@app.route("/")
def index():

    return render_template("index.html")

@app.route("/data", methods =["GET","POST"])
def data():
    if 'form1' in request.form:
        global datas
        global csv_file
        global df
        global columns_all
        global x
        

        df = request.files["myfile"]
        csv_file = request.files["myfile"]
        
        
        
        if datas:
            print(datas.head())
        if csv_file:
            file_name = csv_file.filename

            df = pd.read_csv(csv_file)           
            describe = df.describe()
            columns_all = df.columns
            missing = df.isnull().sum()
            missing = missing.to_frame()
            missing =  missing.rename(columns={0: 'Eksik Veri'})

            corr = df.corr()
            
            
            
            df_numeric = df.select_dtypes(include =['float64','int64'])
            columns = df_numeric.columns
            columns_all = df.columns
            
            
            fig = plt.figure()
            io = BytesIO()
            for i in columns:
                df_numeric[i] = df_numeric[i].fillna(df_numeric[i].mean())
            flierprops = dict(marker='o', markerfacecolor='r', markersize=12,
                  linestyle='none', markeredgecolor='g')

            df_numeric = df_numeric.astype(int)
            columns = df_numeric.columns
            plt.boxplot(df_numeric[columns].values ,flierprops=flierprops)
            
            
            fig.savefig(io, format='png')

           

                       
            
            
            data = base64.encodestring(io.getvalue()).decode('utf-8')
            
            html = 'data:image/png;base64,{}'


    
            
            

            number_of_rows = len(df)
            number_of_columns = len(df.columns)
            
            return render_template("data.html" ,html = html.format(data), len = len(columns),  columns = columns,tables=[describe.to_html(classes='data')] ,tablestwo=[missing.to_html(classes='data')] , tablesthree=[corr.to_html(classes='data')] ,satir = number_of_rows, sutun = number_of_columns , file = file_name)
            
    if 'form2' in request.form:
       
        
        return render_template("dataSteptwo.html")
    
    if 'datasteptwo' in request.form:
              
        if request.form.getlist('deletemissing'):
            df = df.dropna()
            flash('Eksik veriler silindi!')
            

        if request.form.getlist("meanmissing"):
            df = df.fillna(df.mean())
            df = df.fillna(method = "bfill") 
            flash('Eksik veriler ortalama değer ile dolduruldu!')
                  
        
        if request.form.getlist("deleteoutlier"):
            df_n = df.select_dtypes(["int64","float64"])
            Q1 = df_n.quantile(0.25)
            Q3 = df_n.quantile(0.75)
            IQR = Q3-Q1
            alt_sinir = Q1 - 1.5*IQR
            ust_sinir = Q3 + 1.5*IQR
            df_n = df_n[~((df_n < (alt_sinir )) | (df_n > (ust_sinir))).any(axis = 1)]
            df_n = pd.DataFrame(df_n)
            df[df_n.columns] = df_n
            flash('Aykırı veriler silindi!')
            print(df)
            
        
        x = df["yas"]
       
           
        x= x.values.reshape(-1,1)
        
                     
        return render_template("dataSteptwo.html")

    if 'datastepthree' in request.form:
        global algorithm
        global test
        global bagimli_degisken
    
        test = request.form.get('testt')
        
        bagimli_degisken =  request.form.get('features')
        algorithm = request.form.get('algorithms')

        
        return render_template("dataStepthree.html" ,option_list = columns_all)
    if 'datastepfour' in request.form:
        test = float(test)
        print(algorithm)                       
        bagimli_degisken = df[bagimli_degisken]
        X_train, X_test, Y_train, Y_test = train_test_split(x,bagimli_degisken, test_size = test, random_state = 42) 
        if algorithm == "knn":
            knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
            knn.fit(X_train,Y_train)
            y_pred = knn.predict(X_test)

            rs = recall_score(Y_test, y_pred, average=None)
            ps = precision_score(Y_test, y_pred, average=None)
            acc = accuracy_score(Y_test, y_pred)

          





        if algorithm == "linear":
            lin_reg = LinearRegression()
            lin_reg.fit(X_train,Y_train)
           
            print('Linear R2 degeri')
            print(r2_score(Y_test, lin_reg.predict(X_test)))
        
        if algorithm == "svr":
            svr_reg = SVR(kernel='rbf')
            svr_reg.fit(X_test,Y_test)

            print('SVR R2 degeri')
            print(r2_score(Y_test, svr_reg.predict(X_test)))
        if algorithm == "dt":
            dtc = DecisionTreeClassifier(criterion = 'entropy')

            dtc.fit(X_train,Y_train)
            y_pred = dtc.predict(X_test)

            cm = confusion_matrix(Y_test,y_pred)
            
            print(cm)
        if algorithm == "rf":
            print("burada")  
        if algorithm == "logistic":
            logr = LogisticRegression(random_state=0)
            logr.fit(X_train,Y_train)
            y_pred = logr.predict(X_test)

            cm = confusion_matrix(Y_test,y_pred)
            print(cm)
        if algorithm == "naive":
            svc = SVC(kernel='rbf')
            svc.fit(X_train,Y_train)

            y_pred = svc.predict(X_test)

            cm = confusion_matrix(Y_test,y_pred)
            print(cm)
        if algorithm == "k-means":
            print("burada")
        if algorithm == "svm":
            svc = SVC(kernel='rbf')
            svc.fit(X_train,Y_train)

            y_pred = svc.predict(X_test)

            cm = confusion_matrix(Y_test,y_pred)
            print(cm)
    
    
    
   
    return render_template("dataStepfour.html" , acc = acc, rs=rs,ps=ps )
    if 'datastepfive' in request.form:
            

        return render_template("dataStepfive.html")





    




if __name__ == "__main__":
    app.run(debug = True)
