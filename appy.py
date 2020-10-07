from flask import Flask,render_template,request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
dataset=pd.read_csv(r'https://raw.githubusercontent.com/sairamsnv/salaryapp/master/Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression


app=Flask(__name__)


@app.route('/',methods=['GET','POST'])


def fun():
    return render_template('my.html')

@app.route('/sai/',methods=['POST'])

def fun1():
    myfetures=[int (x) for x in request.form.values()]
    #final_feature=[np.array(myfetures)]
    #x =final_feature
    #x = np.asarray(myfetures, dtype='float64')
    slr=LinearRegression()
    slr.fit(x_train,y_train)
    #myfetures=np.reshape(0,1)
    #print(myfetures)
   # X = X.reshape(X.shape[0], -1)
    y_predict=slr.predict([myfetures])
    #print(y_predict)
    return render_template('mydisplay.html',value='prediction salary ${}'.format(y_predict))

#@app.route('/ram/',methods=['POST'])
#def fun2(request):
   
    #plt.scatter(x_train,y_train,color='red')
    #slr=LinearRegression()
    #slr.fit(x_train,y_train)
    #global a
    
    #x = np.asarray(a, dtype='float64')
    #a=np.reshape(-1,1)
    
    ##plt.show()
    #return render(request,'pixil.html')

if __name__ == "__main__":
    app.run(debug=True)


