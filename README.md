# Madhawa_Illangasinghe
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 16:00:54 2021

@author: madhawa
"""

import socket 
import sys
import time
import Data_Manage_2 as DM


# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

server_address = (HOST, PORT)
print (sys.stderr, 'starting up on %s port %s' % server_address)
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

while True:
    # Wait for a connection
#    print (sys.stderr, 'waiting for a connection')
    connection, client_address = sock.accept()
    try:
#        print (sys.stderr, 'connection from', client_address)

        # Receive the data in small chunks and retransmit it
        while True:
            data = connection.recv(1024)
#            print (sys.stderr, 'received "%s"' % data)
            if data:
#                print (sys.stderr, 'sending data back to the client')
                txt= data.decode("utf-8")                
                D1 = txt.split(", ")
                print("01   txt  ==============" , txt)
                print("02 Split Data List  ==============" ,D1)
                if D1[0]=="CLIENT_SEND_CONFIG_DATA":                   
                    Lower = float(D1[1])
                    Upper = float(D1[2])
                    Ratio = float(D1[3])
                    No_of_Variable =int(D1[4])
                    
                    Customer= ["NULL"]
                    Out_put_data,Out_Fea_List,Load_Parametres = DM.main(Lower,Upper,Ratio,No_of_Variable,Customer) 
                    
#                    Send_data = "SERVER_SEND_CONFIG_OUT_DATA" + "," +"68" + "," +"71000"  + "," + "67"  + "," +"25000"  + "," + "66" + "," + "20000" + "," + "65"  + "," + "5000"  + "," + "64"  + "," + "15000" + "," +"63" + "," +"11000" + "," +"62" + "," +"4000"  + "," +"1" + "," +"2" + "," +"3" + "," +"4" + "," +"5" + "," +"6"
                    Send_data1 = "SERVER_SEND_CONFIG_OUT_DATA"
                    for i in range (26):
                        A= Out_put_data[i+1]
                        if isinstance(A, str):
                            Send_data1=Send_data1+","+A
                        else:
                            Send_data1=Send_data1+","+str(A)
                    
                    print("\n\n ==== Data Send to Server  ================================", Send_data1 )
                    connection.sendall(bytes(Send_data1, 'utf-8'))
                    
                    time.sleep(1)
                    
                    Numer_FL , Bool_FL = Out_Fea_List
                    Nume_ls = Numer_FL["Specs"].tolist()
                    Bool_ls = Bool_FL["Specs"].tolist()
                        

                    Send_data2 = "SERVER_SEND_FEA_LIST_DATA"
                    for i in range (len(Nume_ls)):
                        A= Nume_ls[i]
                        if isinstance(A, str):
                            Send_data2=Send_data2+","+A
                        else:
                            Send_data2=Send_data2+","+str(A)
                            
                    for i in range (len(Bool_ls)):
                        A= Bool_ls[i]
                        if isinstance(A, str):
                            Send_data2=Send_data2+","+A
                        else:
                            Send_data2=Send_data2+","+str(A)
                    
                    connection.sendall(bytes(Send_data2, 'utf-8'))

#'''============ Customer Data LOAD============================== '''

                if D1[0]=="CUSTMER_DATA_LOAD":   
                    print("\n\nCUSTMER_DATA_VALI",)
                    Lower = float(D1[1])
                    Upper = float(D1[2])
                    Ratio = float(D1[3])
                    No_of_Variable =int(D1[4])
                    
                    
                    Customer= ["LOAD"]
                    
                    Out_put_data,Out_Fea_List,OUT_Parametres = DM.main(Lower,Upper,Ratio,No_of_Variable,Customer) 
                    
                    Send_data3 = "CUSTMER_DATA_LOAD_OUT"
                    for i in range (len(OUT_Parametres)):
                        Load= OUT_Parametres[i]
                        print('==== ', i ,'=======',Load)
                        if isinstance(Load, str):
                            Send_data3=Send_data3+"," + Load
                        else:
                            Send_data3=Send_data3+","+str(round(Load,3))
                            
                    print("\n\n ==== Data Send to Server  ================================", Send_data3 )
                    time.sleep(1)
                    connection.sendall(bytes(Send_data3, 'utf-8'))
                    time.sleep(1)

#'''============ Customer Data validation============================== '''

                if D1[0]=="CUSTMER_DATA_VALI":   
                    print("\n\nCUSTMER_DATA_VALI  :",txt)
                    Lower = float(D1[1])
                    Upper = float(D1[2])
                    Ratio = float(D1[3])
                    No_of_Variable =int(D1[4])
                    
                    var_n=[]
                    print("lenDi-------",len(D1), '====',((len(D1)-5)/2))
                    for i in range (int((len(D1)-5)/2)) :
                        var_n= var_n+ [str(D1[i+5])]

                    var_V=[]
                    for i in range (int((len(D1)-5)/2)) :
                        var_V=var_V +[ float(D1[i+19])]
                                            
                    
                    Customer= ["VALID"] + var_n + var_V
                    
                    Out_put_data,Out_Fea_List,OUT_Parametres = DM.main(Lower,Upper,Ratio,No_of_Variable,Customer) 
                    
                    Send_data4 = "CUSTMER_DATA_VALI_OUT"
                    print("\n\n\nOUT_Parametres==== : ", OUT_Parametres)
                    for i in range (len(OUT_Parametres)):
                        Load= OUT_Parametres[i]
                        print('==== ', i ,'=======',Load)
                        if isinstance(Load, str):
                            Send_data4=Send_data4+"," + Load
                        else:
                            Send_data4=Send_data4+","+str(round(Load,3))
                            
#                    Send_ACC=""
#                    for i in range (5):
#                        A= Out_put_data[i+19]
#                        if isinstance(A, str):
#                            Send_ACC=Send_ACC+","+A
#                        else:
#                            Send_ACC=Send_ACC+","+str(A)
#                    
#                    Send_data4=Send_data4+Send_ACC
#                            
                    
                    print("\n\n ==== Data Send to Server  ================================", Send_data4 )
                    time.sleep(1)
                    connection.sendall(bytes(Send_data4, 'utf-8'))
                    time.sleep(1)

            
    finally:
        # Clean up the connection
        connection.close()


# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 16:01:25 2021

@author: madhawa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.api as sm
import os

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn import linear_model


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import ComplementNB
#from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn.decomposition import PCA


def NB_Classifires(X_train, X_test, y_train, y_test):
    
    clf_GNB = GaussianNB()
    clf_GNB.fit(X_train, y_train)
    y_pred = clf_GNB.predict(X_test) 
    M = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = M.ravel()  
    print ("Accuracy GNB ==1================= :",metrics.accuracy_score(y_test, y_pred))
    NB_Acc =metrics.accuracy_score(y_test, y_pred)
    
    clf_DT = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    clf_DT.fit(X_train, y_train)
    y_pred= clf_DT.predict(X_test)
    M = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = M.ravel() 
    print ("Accuracy DT  ==================== :", metrics.accuracy_score(y_test, y_pred))
    DT_Acc =metrics.accuracy_score(y_test, y_pred)
    
    
    clf_SVM = svm.SVC(kernel='linear') # Linear Kernel
    clf_SVM.fit(X_train, y_train)
    y_pred = clf_SVM.predict(X_test)
    M = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = M.ravel() 
    print ("Accuracy SVM ==================== :", metrics.accuracy_score(y_test, y_pred))
    SVM_Acc =metrics.accuracy_score(y_test, y_pred)
    
    
    clf_NB_DT = VotingClassifier(estimators=[('GNB', clf_GNB), ('DT', clf_DT)],voting='soft', weights=[2, 1])
    clf_NB_DT.fit(X_train, y_train)
    y_pred= clf_NB_DT.predict(X_test)
    M = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = M.ravel() 
    print ("Accuracy Voting (GNB + DT)======= :", metrics.accuracy_score(y_test, y_pred))
    NBDT_Acc   =metrics.accuracy_score(y_test, y_pred)


    clf_SVM_DT = VotingClassifier(estimators=[ ('SVM', clf_SVM), ('DT', clf_DT)], voting='hard')
    clf_SVM_DT.fit(X_train, y_train)
    y_pred= clf_SVM_DT.predict(X_test)
    M = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = M.ravel()   
    print ("Accuracy Voting (SVM + DT )====== :", metrics.accuracy_score(y_test, y_pred))
    DTSVM_Acc    =metrics.accuracy_score(y_test, y_pred) 

    clf_NB_SVM = VotingClassifier(estimators=[ ('SVM', clf_SVM), ('GNB', clf_GNB)], voting='hard')
    clf_NB_SVM.fit(X_train, y_train)
    y_pred= clf_NB_SVM.predict(X_test)
    M = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = M.ravel()   
    print ("Accuracy Voting (SVM + GNB)====== :", metrics.accuracy_score(y_test, y_pred))
    SVMNB_Acc     =metrics.accuracy_score(y_test, y_pred)
 
#    SVM_Acc   = 1.000
#    DTSVM_Acc = 2.000 
#    SVMNB_Acc = 3.000 
    
    return ([round(NB_Acc,4),round(DT_Acc,4),round(SVM_Acc,4),round(NBDT_Acc,4),round(DTSVM_Acc,4) ,round(SVMNB_Acc,4) ])
    # plotting the confusion matrix

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 23:41:04 2021

@author: madhawa
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

fig = plt.figure(figsize = (8,8))
ax = plt.axes(projection = '3d')
# Make data.
X = np.arange(1,11, 1)
Y = np.arange(0.2, 1.6, 0.1)
X, Y = np.meshgrid(X, Y)
Z1 = np.array([0.8845,0.8825,0.8922,0.8863,0.8863,0.8886,0.8816,0.8888,0.8825,0.8898,0.8254,0.8165,0.8266,0.8258,0.8266,0.8261,0.8246,0.8252,0.8244,0.8219,0.7677,0.7629,0.7665,0.7655,0.7669,0.7643,0.7610,0.7627,0.7623,0.7687,0.7330,0.7319,0.7350,0.7393,0.7320,0.7356,0.7407,0.7381,0.7390,0.7362,0.7305,0.7362,0.7337,0.7367,0.7294,0.7299,0.7277,0.7326,0.7323,0.7330,0.7233,0.7238,0.7248,0.7241,0.7243,0.7211,0.7244,0.7264,0.7252,0.7245,0.7276,0.7254,0.7223,0.7255,0.7258,0.7231,0.7217,0.7227,0.7270,0.7242,0.7215,0.7210,0.7238,0.7227,0.7221,0.7210,0.7188,0.7205,0.7215,0.7197,0.7157,0.7173,0.7183,0.7194,0.7158,0.7174,0.7166,0.7159,0.7155,0.7154,0.7154,0.7135,0.7136,0.7154,0.7136,0.7176,0.7137,0.7135,0.7166,0.7153,0.7140,0.7162,0.7125,0.7127,0.7134,0.7125,0.7135,0.7138,0.7152,0.7164,0.7140,0.7142,0.7139,0.7139,0.7139,0.7135,0.7164,0.7159,0.7132,0.7140,0.7141,0.7124,0.7148,0.7125,0.7120,0.7120,0.7147,0.7127,0.7133,0.7141,0.7113,0.7149,0.7111,0.7121,0.7114,0.7129,0.7111,0.7156,0.7133,0.7106 ])
Z=Z1.reshape(14,10)



# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.set_zlim3d(np.min(Z1,0),np.max(Z1,0))
ax.set_zlabel("Accuracy")


ax.view_init(40,30)
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 23:41:04 2021

@author: madhawa
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

fig = plt.figure(figsize = (8,8))
ax = plt.axes(projection = '3d')
# Make data.
X = np.arange(1,11, 1)
Y = np.arange(0.2, 1.6, 0.1)
X, Y = np.meshgrid(X, Y)
Z1 = np.array([0.8845,0.8825,0.8922,0.8863,0.8863,0.8886,0.8816,0.8888,0.8825,0.8898,0.8254,0.8165,0.8266,0.8258,0.8266,0.8261,0.8246,0.8252,0.8244,0.8219,0.7677,0.7629,0.7665,0.7655,0.7669,0.7643,0.7610,0.7627,0.7623,0.7687,0.7330,0.7319,0.7350,0.7393,0.7320,0.7356,0.7407,0.7381,0.7390,0.7362,0.7305,0.7362,0.7337,0.7367,0.7294,0.7299,0.7277,0.7326,0.7323,0.7330,0.7233,0.7238,0.7248,0.7241,0.7243,0.7211,0.7244,0.7264,0.7252,0.7245,0.7276,0.7254,0.7223,0.7255,0.7258,0.7231,0.7217,0.7227,0.7270,0.7242,0.7215,0.7210,0.7238,0.7227,0.7221,0.7210,0.7188,0.7205,0.7215,0.7197,0.7157,0.7173,0.7183,0.7194,0.7158,0.7174,0.7166,0.7159,0.7155,0.7154,0.7154,0.7135,0.7136,0.7154,0.7136,0.7176,0.7137,0.7135,0.7166,0.7153,0.7140,0.7162,0.7125,0.7127,0.7134,0.7125,0.7135,0.7138,0.7152,0.7164,0.7140,0.7142,0.7139,0.7139,0.7139,0.7135,0.7164,0.7159,0.7132,0.7140,0.7141,0.7124,0.7148,0.7125,0.7120,0.7120,0.7147,0.7127,0.7133,0.7141,0.7113,0.7149,0.7111,0.7121,0.7114,0.7129,0.7111,0.7156,0.7133,0.7106 ])
Z=Z1.reshape(14,10)



# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.set_zlim3d(np.min(Z1,0),np.max(Z1,0))
ax.set_zlabel("Accuracy")


ax.view_init(40,30)
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 08:07:30 2021

@author: madhawa
"""

# =============================================================================================
""" This module is toload the Data Set and 

 """ 
# ===============================================================================================


import numpy as np
import pandas as pd


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn import linear_model


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

def Anova_Feature_selction (df,Ratio,No_of_Variable):
    X = df.iloc[:,1:-1]  #independent columns
    y = df.iloc[:,-1]    #target column i.e price range
#    print("x----\n\n",X.head())
#    print("y----\n\n",y.head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Ratio, random_state=0)

    bestfeatures = SelectKBest(score_func=f_classif, k=No_of_Variable)
    fit = bestfeatures.fit(X_train,y_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_train.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    return (featureScores.nlargest(No_of_Variable,'Score'))  #print 10 best features


def Chi_Feature_selction (df,Ratio,No_of_Variable):
    X = df.iloc[:,1:-1]  #independent columns
    y = df.iloc[:,-1]    #target column i.e price range
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Ratio, random_state=0)

    bestfeatures = SelectKBest(score_func=chi2, k=No_of_Variable)
    fit = bestfeatures.fit(X_train,y_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_train.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    df=featureScores.nlargest(No_of_Variable,'Score')
    df=df[df.Specs != "churn"]
    print ("*****************************************\n",df)
    return (df)  #print 10 best features

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 08:07:30 2021

@author: madhawa
"""

# =============================================================================================
""" This module is toload the Data Set and 

 """ 
# ===============================================================================================


import numpy as np
import pandas as pd
import xlsxwriter


#import Visulization as vis
import Feature_selction as fea_sel
#import Gaussian_Class_1 as Gas_Cla

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn.decomposition import PCA

import sys
import warnings


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def main(Lower,Upper,Ratio,No_of_Variable,Customer):
    
    Lower= Lower
    Upper= Upper
    
    Ratio=Ratio
    No_of_Variable=No_of_Variable


    missing_values = ["n/a", "na", "--"]
    df = pd.read_csv("data_set_21K.csv" ,delimiter=',',na_values = missing_values)
    DF_Shape = df.shape
    (DF_No_Row ,DF_No_Col)=df.shape
    df = df.apply(pd.to_numeric,errors='coerce')
    Column_List=df.columns.tolist()
    print("Original Data Set Size",df.shape)
    np_index=Column_List[1]
    np_body=np.append(Column_List[6:],Column_List[4])
    df= df[np.append(np_index,np_body)]
    
    
    print("Remove inrelevant coumns",df.shape)
    
    
    
    Bool_Column = ["children","credita","creditaa","prizmrur","prizmub","prizmtwn","refurb","webcap","truck","rv","occprof","occcler","occcrft","occstud","occhmkr","occret","occself","ownrent","marryun","marryyes","mailord","mailres","mailflag","travel","pcown","creditcd","newcelly","newcelln","incmiss","mcycle","setprcm","retcall","churn"]
    Num_Column = ["revenue","mou","recchrge","directas","overage","roam","changem","changer","dropvce","blckvce","unansvce","custcare","threeway","mourec","outcalls","incalls","peakvce","opeakvce","dropblk","callfwdv","callwait","months","uniqsubs","actvsubs","phones","models","eqpdays","age1","age2","retcalls","retaccpt","refer","income","setprc"]
    
    
    row = 0
    col = 0
    
    workbook = xlsxwriter.Workbook("Report_06_18"+'.xlsx')
    worksheet = workbook.add_worksheet("INI_DET")
    
    cell_format0 = workbook.add_format({'bold': True, 'font_color': 'red','border': True})
    cell_format1 = workbook.add_format({'bold': True, 'font_color': 'blue','border': True})
    
    worksheet.write(row, col, "Numerical ini",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    col = 4
    worksheet.write(row, col, "Normalized-Numerical ini",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    col = 8
    worksheet.write(row, col, "Normalized-Num ini -LWD Outlier",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    col = 0
    
    #####  x = x[x.between(x.quantile(.15), x.quantile(.85))]
    
    row += 2
    
    
    
    df_n =df.copy()
    df_n[Num_Column] = df_n[Num_Column].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    df_n_o= df_n.copy()
    df_n_o= df_n_o.loc[ : , ["X"]+Num_Column]
    
    Q1 = df_n_o.quantile(0.25)
    Q3 = df_n_o.quantile(0.75)
    IQR = Q3 - Q1
    
    df_n_o = df_n_o[~((df_n_o < (Q1 - Lower * IQR)) |(df_n_o > (Q3 + Upper * IQR))).any(axis=1)]
    df_Numeric_Out_rem =df_n_o
    df_Bool =df_n.copy()
    df_Bool =df_Bool.loc[ : , ["X"]+Bool_Column]
    
    df_n_o =pd.merge(df_Numeric_Out_rem, df_Bool, on='X')
    
    """df_n_o is row data set -->> Normalized (0,1) --->> remove Outliers based on Q+ IQR*Factor    """
    
    for i in Num_Column:
    #    vis.Plot_His_and_Box(df,i)
        df_temp = df[i]
        df_sub = df_temp.describe() 
        Values = df_sub.values.tolist()+[ df_temp.isnull().sum()]
        Data_lable = ["count","mean","std","min","Q1-25%","Q2-50%","Q3-75%","max","null"]
        
        df_temp_n = df_n[i]
        df_sub_n = df_temp_n.describe() 
        Values_n = df_sub_n.values.tolist()+[ df_temp_n.isnull().sum()]
    
        df_temp_n_o = df_n_o[i]
        df_sub_n_o = df_temp_n_o.describe() 
        Values_n_o = df_sub_n_o.values.tolist()+[ df_temp_n_o.isnull().sum()]
    
       
        worksheet.write(row, col, i,cell_format0)
        col = 4
        worksheet.write(row, col, i,cell_format0)
        col = 8
        worksheet.write(row, col, i,cell_format0)
        col = 0
    
        row += 1
        for j in range(len(Data_lable)):
            worksheet.write(row, col, Data_lable[j],cell_format1)
            worksheet.write(row, col + 1, Values[j],cell_format1)
            col = 4
            worksheet.write(row, col, Data_lable[j],cell_format1)
            worksheet.write(row, col + 1, Values_n[j],cell_format1)
            col = 8
            worksheet.write(row, col, Data_lable[j],cell_format1)
            worksheet.write(row, col + 1, Values_n_o[j],cell_format1)
    
            col = 0
            
    #        print (i," ===  ",Data_lable[j],"  ,  ",Values[j])
            row += 1
        
        row += 2
    
    row = 0
    col = 12
    worksheet.write(row, col, "Bool before process",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    
    row += 2
    
    
    for i in Bool_Column:
    #    vis.Plot_His_and_Box(df,i)
        df_temp = df[i]
        df_sub = df_temp.describe() 
        Values_temp = df_sub.values.tolist()
        Count_1 = df_temp.sum()
        Count_0 = Values_temp[0]-df_temp.sum()
        mode = df_temp.mode()
        Null=  df_temp.isnull().sum() 
        Data_lable = ["count","mean","std","1-count","0-Count","mode",""," ","Null "]
        Values = Values_temp[0:3]+ [Count_1,Count_0,mode," "," ",Null]
        worksheet.write(row, col, i,cell_format0)
    
        row += 1
        for j in range(len(Data_lable)):
            worksheet.write(row, col, Data_lable[j],cell_format1)
            worksheet.write(row, col + 1, Values[j],cell_format1)
    #        print (i," ===  ",Data_lable[j],"  ,  ",Values[j])
            row += 1
        
        row += 2
    
        
     
    #for i in Bool_Column:
    #    df_temp = df[i]
    #    print (df_temp.describe())
        
    #for i in Num_Column:
    #    vis.Plot_His_and_Box(df,i)
    #
    #for i in Bool_Column:
    #    vis.Plot_His_and_Box(df,i)
    #    
    #    
    """==================================================================================
    
        List Wise Deltion
        
    ====================================================================================="""
        
    df_LD = df.copy() 
    df_LD_n = df.copy()
    df_LD.dropna(inplace=True)
    df_LD_n.dropna(inplace=True)
    
    df_LD_n[Num_Column] = df_LD_n[Num_Column].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    df_LD_o= df_LD_n.copy()
    df_LD_o= df_LD_o.loc[ : , ["X"]+Num_Column]
    
    Q1 = df_LD_o.quantile(0.25)
    Q3 = df_LD_o.quantile(0.75)
    IQR = Q3 - Q1
    
    df_LD_o = df_LD_o[~((df_LD_o < (Q1 - Lower * IQR)) |(df_LD_o > (Q3 + Upper * IQR))).any(axis=1)]
    df_Numeric_Out_rem_LD =df_LD_o
    df_Bool_LD =df_LD.copy()
    df_Bool_LD =df_Bool_LD.loc[ : , ["X"]+Bool_Column]
    
    df_LD_o =pd.merge(df_Numeric_Out_rem_LD, df_Bool_LD, on='X')
    
     
    """df_LD_o is row data set---->> Remove missing value List wise deltion 
     -->> Normalized (0,1) --->> remove Outliers based on Q+ IQR*Factor    """
    
    
    #df_LD_n= df_LD_n.div(df_LD_n.max(),1)
     
    worksheet1 = workbook.add_worksheet("List_DEL")
    
    
    row = 0
    col = 0
    
    
    worksheet1.write(row, col, "Numerical list wise deletion",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    col = 4
    worksheet1.write(row, col, "Numerical list wise deletion->Normlaized-",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    col = 8
    worksheet1.write(row, col, "Numerical list wise deletion->Normlaized-> outlier",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    col = 0
    row += 2
    
    for i in Num_Column:
    #    vis.Plot_His_and_Box(df,i)
        df_temp = df_LD[i]
        df_sub = df_temp.describe() 
        Values = df_sub.values.tolist()+[ df_temp.isnull().sum()]
        Data_lable = ["count","mean","std","min","Q1-25%","Q2-50%","Q3-75%","max","null"]
    
        df_temp_n = df_LD_n[i]
        df_sub_n = df_temp_n.describe() 
        Values_n = df_sub_n.values.tolist()+[ df_temp_n.isnull().sum()]
        
        df_temp_n_o = df_LD_o[i]
        df_sub_n_o = df_temp_n_o.describe() 
        Values_n_o = df_sub_n_o.values.tolist()+[ df_temp_n_o.isnull().sum()]
        
        
            
        worksheet1.write(row, col, i,cell_format0)
        col = 4
        worksheet1.write(row, col, i,cell_format0)
        col = 8
        worksheet1.write(row, col, i,cell_format0)
        col = 0
    
        row += 1
        for j in range(len(Data_lable)):
            worksheet1.write(row, col, Data_lable[j],cell_format1)
            worksheet1.write(row, col + 1, Values[j],cell_format1)
    
            col = 4
            worksheet1.write(row, col, Data_lable[j],cell_format1)
            worksheet1.write(row, col + 1, Values_n[j],cell_format1)
            col = 8
            worksheet1.write(row, col, Data_lable[j],cell_format1)
            worksheet1.write(row, col + 1, Values_n_o[j],cell_format1)
    
    
            col = 0
    
    
            row += 1
        
        row += 2
    
    row = 0
    col = 12
    worksheet1.write(row, col, "Bool list wise deletion",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    row += 2
    
    
    for i in Bool_Column:
    #    vis.Plot_His_and_Box(df,i)
        df_temp = df_LD[i]
        df_sub = df_temp.describe() 
        Values_temp = df_sub.values.tolist()
        Count_1 = df_temp.sum()
        Count_0 = Values_temp[0]-df_temp.sum()
        mode = df_temp.mode()
        Null=  df_temp.isnull().sum() 
        Data_lable = ["count","mean","std","1-count","0-Count","mode",""," ","Null "]
        Values = Values_temp[0:3]+ [Count_1,Count_0,mode," "," ",Null]
        worksheet1.write(row, col, i,cell_format0)
    
        row += 1
        for j in range(len(Data_lable)):
            worksheet1.write(row, col, Data_lable[j],cell_format1)
            worksheet1.write(row, col + 1, Values[j],cell_format1)
    #        print (i," ===  ",Data_lable[j],"  ,  ",Values[j])
            row += 1
        
        row += 2
    
    
    """==================================================================================
    
       Mean , Mode imputation
        
    ====================================================================================="""
    
    df_MMI = df.copy() 
    df_MMI_n = df.copy() 
    df_MMI_n[Num_Column] = df_MMI_n[Num_Column].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    
    df_MMI_o= df_MMI_n.copy()
    df_MMI_o= df_MMI_o.loc[ : , ["X"]+Num_Column]
    
    Q1 = df_MMI_o.quantile(0.25)
    Q3 = df_MMI_o.quantile(0.75)
    IQR = Q3 - Q1
    
    df_MMI_o = df_MMI_o[~((df_MMI_o < (Q1 - Lower * IQR)) |(df_MMI_o > (Q3 + Upper * IQR))).any(axis=1)]
    df_Numeric_Out_rem_MMI =df_MMI_o
    df_Bool_MMI =df_MMI.copy()
    df_Bool_MMI =df_Bool_MMI.loc[ : , ["X"]+Bool_Column]
    
    df_MMI_o =pd.merge(df_Numeric_Out_rem_MMI, df_Bool_MMI, on='X')
    
    """df_MMI_o is row data set---->> Mean mode impute 
     -->> Normalized (0,1) --->> remove Outliers based on Q+ IQR*Factor    """
     
    
    
    #df_MMI_n= df_MMI_n.div(df_MMI_n.max(),1)
    
    worksheet1 = workbook.add_worksheet("M_M Impute")
    
    
    row = 0
    col = 0
    
    
    worksheet1.write(row, col, "Numerical Mean mode Imputation",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    col = 4
    worksheet1.write(row, col, "Numerical Mean mode Imputation-Normalized",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    col = 8
    worksheet1.write(row, col, "Numerical Mean mode Imputation-Normalized-Outlier",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    
    col = 0
    row += 2
    
    for i in Num_Column:
    #    vis.Plot_His_and_Box(df,i)
        df_MMI[i].fillna(df_MMI[i].mean(), inplace=True)
        df_temp=df_MMI[i]
        df_sub = df_temp.describe() 
        Values = df_sub.values.tolist()+[ df_temp.isnull().sum()]
        Data_lable = ["count","mean","std","min","Q1-25%","Q2-50%","Q3-75%","max","null"]
        
        df_MMI_n[i].fillna(df_MMI_n[i].mean(), inplace=True)
        df_temp_n=df_MMI_n[i]
        df_sub_n = df_temp_n.describe() 
        Values_n = df_sub_n.values.tolist()+[ df_temp_n.isnull().sum()]
        
        df_MMI_o[i].fillna(df_MMI_o[i].mean(), inplace=True)
        df_temp_o=df_MMI_o[i]
        df_sub_o = df_temp_o.describe() 
        Values_o = df_sub_o.values.tolist()+[ df_temp_o.isnull().sum()]
      
        
        
        worksheet1.write(row, col, i,cell_format0)
        col = 4
        worksheet1.write(row, col, i,cell_format0)
        col = 8
        worksheet1.write(row, col, i,cell_format0)
        
        col = 0
    
        row += 1
        for j in range(len(Data_lable)):
            worksheet1.write(row, col, Data_lable[j],cell_format1)
            worksheet1.write(row, col + 1, Values[j],cell_format1)
          
            col = 4
            worksheet1.write(row, col, Data_lable[j],cell_format1)
            worksheet1.write(row, col + 1, Values_n[j],cell_format1)
            
            col = 8
            worksheet1.write(row, col, Data_lable[j],cell_format1)
            worksheet1.write(row, col + 1, Values_o[j],cell_format1)
            
            col = 0
            
            row += 1
        
        row += 2
    
    row = 0
    col = 12
    worksheet1.write(row, col, "Bool list wise deletion",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    row += 2
    
    
    for i in Bool_Column:
    #    vis.Plot_His_and_Box(df,i)
        df_MMI[i].fillna(df_MMI[i].mode(), inplace=True)
        df_temp=df_MMI[i]
    
        df_sub = df_temp.describe() 
        Values_temp = df_sub.values.tolist()
        Count_1 = df_temp.sum()
        Count_0 = Values_temp[0]-df_temp.sum()
        mode = df_temp.mode()
        Null=  df_temp.isnull().sum() 
        Data_lable = ["count","mean","std","1-count","0-Count","mode",""," ","Null "]
        Values = Values_temp[0:3]+ [Count_1,Count_0,mode," "," ",Null]
        worksheet1.write(row, col, i,cell_format0)
    
        row += 1
        for j in range(len(Data_lable)):
            worksheet1.write(row, col, Data_lable[j],cell_format1)
            worksheet1.write(row, col + 1, Values[j],cell_format1)
    #        print (i," ===  ",Data_lable[j],"  ,  ",Values[j])
            row += 1
        
        row += 2
    
    
    print("Before impution")
    print(df.shape)
    print("After impution")
    print(df_MMI.shape)
    
    Raw_Data = df.copy()
    
    Raw_Norm = df_n.copy()
    Raw_Norm_OutLier=df_n_o.copy()
    
    Lst_Del_Norm= df_LD_n.copy()
    Lst_Del_Norm_OutLier =df_LD_o.copy()
    
    MeMo_Imp_Norm= df_MMI_n.copy()
    MeMo_Imp_Norm_OutLier= df_MMI_o.copy()
    
    print("\n\n")
    print ("Raw_Data========================= : ",Raw_Data.shape )
    print ("Raw_Norm========================= : ",Raw_Norm.shape )
    print ("Raw_Norm_OutLier================= : ",Raw_Norm_OutLier.shape )
    
    print ("Lst_Del_Norm===================== : ",Lst_Del_Norm.shape )
    print ("Lst_Del_Norm_OutLier============= : ",Lst_Del_Norm_OutLier.shape )
    print ("MeMo_Imp_Norm==================== : ",MeMo_Imp_Norm.shape )
    print ("MeMo_Imp_Norm_OutLier============ : ",MeMo_Imp_Norm_OutLier.shape )
    
    
    
    
    worksheet1 = workbook.add_worksheet("Fea_List")
    
    col = 0
    row = 0
    
    worksheet1.write(row, col, "List_Delele_Normlized Numerical ",cell_format0)
    col = 4
    worksheet1.write(row, col, "List_Delele_Normlized Bool ",cell_format0)
    col = 0
    
    row += 1
    FS_Lst_Del_Norm = fea_sel.Anova_Feature_selction (          Lst_Del_Norm.loc[ : , Num_Column+["churn"]],Ratio,No_of_Variable)
    FS_Lst_Del_Bool = fea_sel.Chi_Feature_selction (          Lst_Del_Norm.loc[ : , Bool_Column+["churn"]],Ratio,No_of_Variable+1)
    for i in range(len(FS_Lst_Del_Norm["Specs"])):
        A=FS_Lst_Del_Norm["Specs"]
        B=FS_Lst_Del_Norm["Score"]    
        worksheet1.write(row, col    , A.iloc[i] ,cell_format1)
        worksheet1.write(row, col + 1, B.iloc[i] ,cell_format1)
        
        col = 4
        A=FS_Lst_Del_Bool["Specs"]
        B=FS_Lst_Del_Bool["Score"]    
        if (B.iloc[i]==np.inf):
            B="inf"
        else:
            B=B.iloc[i]   
        worksheet1.write(row, col    , A.iloc[i] ,cell_format1)
        worksheet1.write(row, col + 1, B         ,cell_format1)
        col = 0
        row += 1
    
    row+=2
    worksheet1.write(row, col, "List_Delele_Normlized Outlier Numerical ",cell_format0)
    col = 4
    worksheet1.write(row, col, "List_Delele_Normlized Outlier Bool ",cell_format0)
    col = 0
    row += 1
    FS_Lst_Del_Norm_out = fea_sel.Anova_Feature_selction (  Lst_Del_Norm_OutLier.loc[ : , Num_Column+["churn"]],Ratio,No_of_Variable)
    FS_Lst_Del_Bool_out = fea_sel.Chi_Feature_selction (  Lst_Del_Norm_OutLier.loc[ : , Bool_Column+["churn"]],Ratio,No_of_Variable+1)
    for i in range(len(FS_Lst_Del_Norm_out["Specs"])):
        A=FS_Lst_Del_Norm_out["Specs"]
        B=FS_Lst_Del_Norm_out["Score"]
        worksheet1.write(row, col    , A.iloc[i] ,cell_format1)
        worksheet1.write(row, col + 1, B.iloc[i] ,cell_format1)
        
        col = 4
        A=FS_Lst_Del_Bool_out["Specs"]
        B=FS_Lst_Del_Bool_out["Score"]  
        if (B.iloc[i]==np.inf):
            B="inf"
        else:
            B=B.iloc[i]   
        worksheet1.write(row, col    , A.iloc[i] ,cell_format1)
        worksheet1.write(row, col + 1, B         ,cell_format1)
        col = 0
    
        row += 1
        
    row+=2    
    worksheet1.write(row, col, "Mean ,Mode impute Normlized Numerical ",cell_format0)
    col = 4
    worksheet1.write(row, col, "Mean ,Mode impute Normlized Bool ",cell_format0)
    col = 0
    
    row += 1
    FS_MMI_Norm = fea_sel.Anova_Feature_selction (         MeMo_Imp_Norm.loc[ : , Num_Column+["churn"]],Ratio,No_of_Variable)
    FS_MMI_Bool = fea_sel.Chi_Feature_selction   (         MeMo_Imp_Norm.loc[ : , Bool_Column+["churn"]],Ratio,No_of_Variable+1)
    for i in range(len(FS_MMI_Norm["Specs"])):
        A=FS_MMI_Norm["Specs"]
        B=FS_MMI_Norm["Score"]
        worksheet1.write(row, col    , A.iloc[i] ,cell_format1)
        worksheet1.write(row, col + 1, B.iloc[i] ,cell_format1)
        
        col = 4
        A=FS_MMI_Bool["Specs"]
        B=FS_MMI_Bool["Score"]  
        if (B.iloc[i]==np.inf):
            B="inf"
        else:
            B=B.iloc[i]   
        worksheet1.write(row, col    , A.iloc[i] ,cell_format1)
        worksheet1.write(row, col + 1, B         ,cell_format1)
        col = 0
        row += 1
        
        
    row+=2
    worksheet1.write(row, col, "Mean ,Mode impute Normlized Numerical Outl ",cell_format0)
    col = 4
    worksheet1.write(row, col, "Mean ,Mode impute Normlized Bool Outl ",cell_format0)
    col = 0
    row += 1
    FS_MMI_Norm_out = fea_sel.Anova_Feature_selction ( MeMo_Imp_Norm_OutLier.loc[ : , Num_Column+["churn"]],Ratio,No_of_Variable)
    FS_MMI_Bool_out = fea_sel.Chi_Feature_selction   ( MeMo_Imp_Norm_OutLier.loc[ : , Bool_Column+["churn"]],Ratio,No_of_Variable+1)
    
    for i in range(len(FS_MMI_Norm_out["Specs"])):
        A=FS_MMI_Norm_out["Specs"]
        B=FS_MMI_Norm_out["Score"]
        worksheet1.write(row, col    , A.iloc[i] ,cell_format1)
        worksheet1.write(row, col + 1, B.iloc[i] ,cell_format1)
        
        col = 4
        A=FS_MMI_Bool_out["Specs"]
        B=FS_MMI_Bool_out["Score"]
        if (B.iloc[i]==np.inf):
            B="inf"
        else:
            B=B.iloc[i]       
        worksheet1.write(row, col    , A.iloc[i] ,cell_format1)
        worksheet1.write(row, col + 1, B         ,cell_format1)
        col = 0
        row += 1
    
    def check_availability(element, collection: iter):
        return element 
        
    ##==================================================================================================
    #print ("\n\nLst_Del_Norm===================== : ",Lst_Del_Norm.shape )
    #
    #List_Del_Nor_FL= FS_Lst_Del_Norm["Specs"].tolist()+FS_Lst_Del_Bool["Specs"].tolist()
    #if check_availability('churn', List_Del_Nor_FL)=='churn':
    #    List_Del_Nor_FL.remove('churn')
    #
    #Temp_LD_N_DF =Lst_Del_Norm[List_Del_Nor_FL+["churn"]].copy()
    #print("No of Featues  ==== : ",len(List_Del_Nor_FL))
    #
    
    #X = Temp_LD_N_DF.iloc[:,:-1]   #independent columns
    #y = Temp_LD_N_DF.iloc[:,-1]    #target column i.e price range
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Ratio, random_state=0) 
    #A=Gas_Cla.NB_Classifires(X_train, X_test, y_train, y_test)
    
    
    ##==================================================================================================
    
    print ("\n\nLst_Del_Norm_OutLier============= : ",Lst_Del_Norm_OutLier.shape)
    (DF_LD_O_No_Row,DF_LD_O_No_Col )=Lst_Del_Norm_OutLier.shape
    

    List_Del_Nor_OUT_FL= FS_Lst_Del_Norm_out["Specs"].tolist()+FS_Lst_Del_Bool_out["Specs"].tolist()
#    if check_availability('churn', List_Del_Nor_OUT_FL)=='churn':
#        List_Del_Nor_OUT_FL.remove('churn')
    
    Temp_LD_N_O_DF=Lst_Del_Norm_OutLier[List_Del_Nor_OUT_FL+["churn"]].copy()
    print("No of Featues  ==== : ",len(List_Del_Nor_OUT_FL))

    X = Temp_LD_N_O_DF.iloc[:,:-1]   #independent columns
    y = Temp_LD_N_O_DF.iloc[:,-1]    #target column i.e price range
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Ratio, random_state=0) 
    
    (DF_LD_O_No_Train_Row ,DF_LD_O_No_Train_Col )=X_train.shape
    (DF_LD_O_No_Test_Row  ,DF_LD_O_No_Test_Col  )=X_test.shape

#    LD_N_O_Acc=Gas_Cla.NB_Classifires(X_train, X_test, y_train, y_test)
    
    '''=============LD_N_O_Acc ================================================================'''
    
    clf_GNB = GaussianNB()
    clf_GNB.fit(X_train, y_train)
    y_pred = clf_GNB.predict(X_test) 
    M = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = M.ravel()  
    print ("Accuracy GNB ==1================= :",metrics.accuracy_score(y_test, y_pred))
    NB_Acc =metrics.accuracy_score(y_test, y_pred)
    
    clf_DT = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    clf_DT.fit(X_train, y_train)
    y_pred= clf_DT.predict(X_test)
    M = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = M.ravel() 
    print ("Accuracy DT  ==================== :", metrics.accuracy_score(y_test, y_pred))
    DT_Acc =metrics.accuracy_score(y_test, y_pred)
    
    
    clf_SVM = svm.SVC(kernel='linear') # Linear Kernel
    clf_SVM.fit(X_train, y_train)
    y_pred = clf_SVM.predict(X_test)
    M = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = M.ravel() 
    print ("Accuracy SVM ==================== :", metrics.accuracy_score(y_test, y_pred))
    SVM_Acc =metrics.accuracy_score(y_test, y_pred)
    
    
    clf_NB_DT = VotingClassifier(estimators=[('GNB', clf_GNB), ('DT', clf_DT)],voting='soft', weights=[2, 1])
    clf_NB_DT.fit(X_train, y_train)
    y_pred= clf_NB_DT.predict(X_test)
    M = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = M.ravel() 
    print ("Accuracy Voting (GNB + DT)======= :", metrics.accuracy_score(y_test, y_pred))
    NBDT_Acc   =metrics.accuracy_score(y_test, y_pred)



    clf_SVM_DT = VotingClassifier(estimators=[ ('SVM', clf_SVM), ('DT', clf_DT)], voting='hard')
    clf_SVM_DT.fit(X_train, y_train)
    y_pred= clf_SVM_DT.predict(X_test)
    M = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = M.ravel()   
    print ("Accuracy Voting (SVM + DT )====== :", metrics.accuracy_score(y_test, y_pred))
    DTSVM_Acc    =metrics.accuracy_score(y_test, y_pred) 

    clf_NB_SVM = VotingClassifier(estimators=[ ('SVM', clf_SVM), ('GNB', clf_GNB)], voting='hard')
    clf_NB_SVM.fit(X_train, y_train)
    y_pred= clf_NB_SVM.predict(X_test)
    M = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = M.ravel()   
    print ("Accuracy Voting (SVM + GNB)====== :", metrics.accuracy_score(y_test, y_pred))
    SVMNB_Acc     =metrics.accuracy_score(y_test, y_pred)
 
#    SVM_Acc   = 1.000
#    DTSVM_Acc = 2.000 
#    SVMNB_Acc = 3.000 
    
    LD_N_O_Acc = [round(NB_Acc,4),round(DT_Acc,4),round(SVM_Acc,4),round(NBDT_Acc,4),round(DTSVM_Acc,4) ,round(SVMNB_Acc,4) ]

    '''=============LD_N_O_Acc ================================================================'''
    
    
    
    
    print("LD_N_O_Acc",LD_N_O_Acc)
    
    [NB_LD_Acc,DT_LD_Acc,SVM_LD_Acc,NBDT_LD_Acc,DTSVM_LD_Acc ,SVMNB_LD_Acc]=LD_N_O_Acc
    ##==================================================================================================
    
    #print ("\n\nMean Mode impute Norm_============= : ",MeMo_Imp_Norm.shape)
    #MMI_Nor_FL= FS_MMI_Norm["Specs"].tolist()+FS_MMI_Bool["Specs"].tolist()
    #if check_availability('churn', MMI_Nor_FL)=='churn':
    #    MMI_Nor_FL.remove('churn')
    #
    #Temp_MMI_N_DF=MeMo_Imp_Norm[MMI_Nor_FL+["churn"]].copy()
    #print("No of Featues  ==== : ",len(MMI_Nor_FL))
    
    #X = Temp_MMI_N_DF.iloc[:,:-1]   #independent columns
    #y = Temp_MMI_N_DF.iloc[:,-1]    #target column i.e price range
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Ratio, random_state=0) 
    #A=Gas_Cla.NB_Classifires(X_train, X_test, y_train, y_test)
    
    
    
    ##==================================================================================================
    
    print ("\n\nMeMo_Imp_Norm_OutLier============ : ",MeMo_Imp_Norm_OutLier.shape )
    MMI_Nor_OUT_FL= FS_MMI_Norm_out["Specs"].tolist()+FS_MMI_Bool_out["Specs"].tolist()
#    if check_availability('churn', MMI_Nor_OUT_FL)=='churn':
#        MMI_Nor_OUT_FL.remove('churn')

    (DF_MMI_O_No_Row,DF_MMI_O_No_Col)=MeMo_Imp_Norm_OutLier.shape
    
    Temp_MMI_N_O_DF=MeMo_Imp_Norm_OutLier[MMI_Nor_OUT_FL+["churn"]].copy()
    print("No of Featues  ==== : ",len(MMI_Nor_OUT_FL))
    
    X = Temp_MMI_N_O_DF.iloc[:,:-1]   #independent columns
    y = Temp_MMI_N_O_DF.iloc[:,-1]    #target column i.e price range

    print("No of Shape  ==== : ",(X.shape))
    print("No of Head  ==== : ",(X.head()))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Ratio, random_state=0) 
    
    (DF_MMI_O_No_Train_Row, DF_MMI_O_No_Train_Col)=X_train.shape
    (DF_MMI_O_No_Test_Row, DF_MMI_O_No_Test_Col)=X_test.shape
    
   
#    MMI_N_O_Acc =Gas_Cla.NB_Classifires(X_train, X_test, y_train, y_test)
    
    '''=============MMI_N_O_Acc ================================================================'''
    
    clf_GNB = GaussianNB()
    clf_GNB.fit(X_train, y_train)
    y_pred = clf_GNB.predict(X_test) 
    M = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = M.ravel()  
    print ("Accuracy GNB ==1================= :",metrics.accuracy_score(y_test, y_pred))
    NB_Acc = metrics.accuracy_score(y_test, y_pred)
    NB_Acc = (tn+ fp+ tp)/(tn+ fp+ fn+ tp)

    clf_DT = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    clf_DT.fit(X_train, y_train)
    y_pred= clf_DT.predict(X_test)
    M = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = M.ravel() 
    print ("Accuracy DT  ==================== :", metrics.accuracy_score(y_test, y_pred))
    DT_Acc =metrics.accuracy_score(y_test, y_pred)
    
    clf_SVM = svm.SVC(kernel='linear') # Linear Kernel
    clf_SVM.fit(X_train, y_train)
    y_pred = clf_SVM.predict(X_test)
    M = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = M.ravel() 
    print ("Accuracy SVM ==================== :", metrics.accuracy_score(y_test, y_pred))
    SVM_Acc =metrics.accuracy_score(y_test, y_pred)
    
    
    clf_NB_DT = VotingClassifier(estimators=[('GNB', clf_GNB), ('DT', clf_DT)],voting='soft', weights=[2, 1])
    clf_NB_DT.fit(X_train, y_train)
    y_pred= clf_NB_DT.predict(X_test)
    M = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = M.ravel() 
    print ("Accuracy Voting (GNB + DT)======= :", metrics.accuracy_score(y_test, y_pred))
 


    clf_SVM_DT = VotingClassifier(estimators=[ ('SVM', clf_SVM), ('DT', clf_DT)], voting='hard')
    clf_SVM_DT.fit(X_train, y_train)
    y_pred= clf_SVM_DT.predict(X_test)
    M = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = M.ravel()   
    print ("Accuracy Voting (SVM + DT )====== :", metrics.accuracy_score(y_test, y_pred))
    DTSVM_Acc    =metrics.accuracy_score(y_test, y_pred) 

    clf_NB_SVM = VotingClassifier(estimators=[ ('SVM', clf_SVM), ('GNB', clf_GNB)], voting='hard')
    clf_NB_SVM.fit(X_train, y_train)
    y_pred= clf_NB_SVM.predict(X_test)
    M = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = M.ravel()   
    print ("Accuracy Voting (SVM + GNB)====== :", metrics.accuracy_score(y_test, y_pred))
    SVMNB_Acc     =metrics.accuracy_score(y_test, y_pred)
 
#    SVM_Acc   = 1.000
#    DTSVM_Acc = 2.000 
#    SVMNB_Acc = 3.000 
    
    MMI_N_O_Acc = [round(NB_Acc,4),round(DT_Acc,4),round(SVM_Acc,4),round(NBDT_Acc,4),round(DTSVM_Acc,4) ,round(SVMNB_Acc,4) ]

    '''=============MMI_N_O_Acc ================================================================'''
    
    
    [NB_MMI_Acc,DT_MMI_Acc,SVM_MMI_Acc,NBDT_MMI_Acc,DTSVM_MMI_Acc ,SVMNB_MMI_Acc] = MMI_N_O_Acc
    workbook.close()
    
    if Customer[0]=="VALID" :
#        X1= X_test.iloc[50:51,:]

        X1_Name = Customer[1:15]
        X1_Val=[]
        for i in range(14) :
            X1_Val =X1_Val +[ Customer[i+15]]   
        X1_Val=[X1_Val]        
                
        X1=pd.DataFrame(X1_Val,  columns =X1_Name)
        
    #    Temp_MMI_N_O_DF.iloc[:,:-1]
        print ("\n\nX1 shape    : = " , X1)
    
        
        [y_pred_GNB] = clf_GNB.predict(X1) 
        [y_pred_DT] = clf_DT.predict(X1) 
        [y_pred_SVM] = clf_SVM.predict(X1) 
        [y_pred_NB_DT] = clf_NB_DT.predict(X1) 
        [y_pred_DT_SVM] = clf_SVM_DT.predict(X1) 
        [y_pred_SVM_NB] = clf_NB_SVM.predict(X1) 
        
        print ("y_pred_GNB     : = " , y_pred_GNB)
        print ("y_pred_DT      : = " , y_pred_DT)
        print ("y_pred_SVM     : = " , y_pred_SVM)
        print ("y_pred_NB_DT   : = " , y_pred_NB_DT)
        print ("y_pred_DT_SVM  : = " , y_pred_DT_SVM)
        print ("y_pred_SVM_NB  : = " , y_pred_SVM_NB)
        
        ACC_LD=[NB_MMI_Acc,DT_MMI_Acc,SVM_MMI_Acc,NBDT_MMI_Acc,DTSVM_MMI_Acc ,SVMNB_MMI_Acc]
        
        Classifire_Pred = [y_pred_GNB,y_pred_DT,y_pred_SVM,y_pred_NB_DT,y_pred_DT_SVM,y_pred_SVM_NB]+ACC_LD
        
        df_Lst_Del_min= df.copy()
    
        FS_Selceted =FS_Lst_Del_Norm_out["Specs" ].tolist()+ FS_Lst_Del_Bool_out["Specs" ].tolist()
    #    print ("FS_Selceted",FS_Selceted)
        df_Lst_Del_min=df_Lst_Del_min[FS_Selceted]
        Lst_Del_min =df_Lst_Del_min.min()
        Lst_Del_max =df_Lst_Del_min.max()        
        print ("\n\nX1 Lst_    : =\n" ,FS_Selceted, Lst_Del_min.tolist(),Lst_Del_max.tolist())
        Load_Parametres =FS_Selceted+ Lst_Del_min.tolist()+Lst_Del_max.tolist()+Classifire_Pred
    else:
        df_Lst_Del_min= df.copy()
    
        FS_Selceted =FS_Lst_Del_Norm_out["Specs" ].tolist()+ FS_Lst_Del_Bool_out["Specs" ].tolist()
    #    print ("FS_Selceted",FS_Selceted)
        df_Lst_Del_min=df_Lst_Del_min[FS_Selceted]
        Lst_Del_min =df_Lst_Del_min.min()
        Lst_Del_max =df_Lst_Del_min.max()
        
        print ("\n\nX1 Lst_    : =\n" ,FS_Selceted, Lst_Del_min.tolist(),Lst_Del_max.tolist())        
        Load_Parametres =FS_Selceted+ Lst_Del_min.tolist()+Lst_Del_max.tolist()




        
#    return ([DF_Shape,DF_No_Col ,DF_No_Row ,DF_LD_O_No_Row,DF_LD_O_No_Col, DF_LD_O_No_Train_Row ,DF_LD_O_No_Train_Col,DF_LD_O_No_Test_Row  ,DF_LD_O_No_Test_Col,DF_MMI_O_No_Row,DF_MMI_O_No_Col,DF_MMI_O_No_Train_Row, DF_MMI_O_No_Train_Col,DF_MMI_O_No_Test_Row, DF_MMI_O_No_Test_Col, NB_LD_Acc,DT_LD_Acc,SVM_LD_Acc,NBDT_LD_Acc  ,DTSVM_LD_Acc ,SVMNB_LD_Acc])
    return ([DF_Shape,DF_No_Col ,DF_No_Row ,DF_LD_O_No_Col,DF_LD_O_No_Row, DF_LD_O_No_Train_Col ,DF_LD_O_No_Train_Row,DF_LD_O_No_Test_Col  ,DF_LD_O_No_Test_Row,DF_MMI_O_No_Col,DF_MMI_O_No_Row,DF_MMI_O_No_Train_Col, DF_MMI_O_No_Train_Row,DF_MMI_O_No_Test_Col, DF_MMI_O_No_Test_Row, NB_LD_Acc,DT_LD_Acc,SVM_LD_Acc,NBDT_LD_Acc  ,DTSVM_LD_Acc ,SVMNB_LD_Acc,NB_MMI_Acc,DT_MMI_Acc,SVM_MMI_Acc,NBDT_MMI_Acc,DTSVM_MMI_Acc ,SVMNB_MMI_Acc],[FS_Lst_Del_Norm_out,FS_Lst_Del_Bool_out],Load_Parametres)

#
#C_DATA = ["VALID"] + ['eqpdays', 'changem', 'mou', 'age1', 'recchrge', 'months', 'age2', 'ownrent', 'incmiss', 'occprof', 'mailres', 'mailord', 'marryun', 'travel']+[812.0, 2192.25, 2336.25, 89.0, 159.92999269999996, 20.0, 49.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
##C_DATA = ["NULL"]
#
#A,B,C,y_pred_GNB =main(1.5,1.5,0.2,7,C_DATA)

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 08:07:30 2021

@author: madhawa
"""

# =============================================================================================
""" This module is toload the Data Set and 

 """ 
# ===============================================================================================


import numpy as np
import pandas as pd
import xlsxwriter


#import Visulization as vis
import Feature_selction as fea_sel
import Gaussian_Class_1 as Gas_Cla

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA

import sys
import warnings


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def main(Lower,Upper,Ratio,No_of_Variable):
    
    Lower= Lower
    Upper= Upper
    
    Ratio=Ratio
    No_of_Variable=No_of_Variable


    missing_values = ["n/a", "na", "--"]
    df = pd.read_csv("data_set_21K.csv" ,delimiter=',',na_values = missing_values)
    DF_Shape = df.shape
    (DF_No_Row ,DF_No_Col)=df.shape
    df = df.apply(pd.to_numeric,errors='coerce')
    Column_List=df.columns.tolist()
    print("Original Data Set Size",df.shape)
    np_index=Column_List[1]
    np_body=np.append(Column_List[6:],Column_List[4])
    df= df[np.append(np_index,np_body)]
    
    
    print("Remove inrelevant coumns",df.shape)
    
    
    
    Bool_Column = ["children","credita","creditaa","prizmrur","prizmub","prizmtwn","refurb","webcap","truck","rv","occprof","occcler","occcrft","occstud","occhmkr","occret","occself","ownrent","marryun","marryyes","mailord","mailres","mailflag","travel","pcown","creditcd","newcelly","newcelln","incmiss","mcycle","setprcm","retcall","churn"]
    Num_Column = ["revenue","mou","recchrge","directas","overage","roam","changem","changer","dropvce","blckvce","unansvce","custcare","threeway","mourec","outcalls","incalls","peakvce","opeakvce","dropblk","callfwdv","callwait","months","uniqsubs","actvsubs","phones","models","eqpdays","age1","age2","retcalls","retaccpt","refer","income","setprc"]
    
    
    row = 0
    col = 0
    
    workbook = xlsxwriter.Workbook("Report_06_18"+'.xlsx')
    worksheet = workbook.add_worksheet("INI_DET")
    
    cell_format0 = workbook.add_format({'bold': True, 'font_color': 'red','border': True})
    cell_format1 = workbook.add_format({'bold': True, 'font_color': 'blue','border': True})
    
    worksheet.write(row, col, "Numerical ini",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    col = 4
    worksheet.write(row, col, "Normalized-Numerical ini",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    col = 8
    worksheet.write(row, col, "Normalized-Num ini -LWD Outlier",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    col = 0
    
    #####  x = x[x.between(x.quantile(.15), x.quantile(.85))]
    
    row += 2
    
    
    
    df_n =df.copy()
    df_n[Num_Column] = df_n[Num_Column].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    df_n_o= df_n.copy()
    df_n_o= df_n_o.loc[ : , ["X"]+Num_Column]
    
    Q1 = df_n_o.quantile(0.25)
    Q3 = df_n_o.quantile(0.75)
    IQR = Q3 - Q1
    
    df_n_o = df_n_o[~((df_n_o < (Q1 - Lower * IQR)) |(df_n_o > (Q3 + Upper * IQR))).any(axis=1)]
    df_Numeric_Out_rem =df_n_o
    df_Bool =df_n.copy()
    df_Bool =df_Bool.loc[ : , ["X"]+Bool_Column]
    
    df_n_o =pd.merge(df_Numeric_Out_rem, df_Bool, on='X')
    
    """df_n_o is row data set -->> Normalized (0,1) --->> remove Outliers based on Q+ IQR*Factor    """
    
    for i in Num_Column:
    #    vis.Plot_His_and_Box(df,i)
        df_temp = df[i]
        df_sub = df_temp.describe() 
        Values = df_sub.values.tolist()+[ df_temp.isnull().sum()]
        Data_lable = ["count","mean","std","min","Q1-25%","Q2-50%","Q3-75%","max","null"]
        
        df_temp_n = df_n[i]
        df_sub_n = df_temp_n.describe() 
        Values_n = df_sub_n.values.tolist()+[ df_temp_n.isnull().sum()]
    
        df_temp_n_o = df_n_o[i]
        df_sub_n_o = df_temp_n_o.describe() 
        Values_n_o = df_sub_n_o.values.tolist()+[ df_temp_n_o.isnull().sum()]
    
       
        worksheet.write(row, col, i,cell_format0)
        col = 4
        worksheet.write(row, col, i,cell_format0)
        col = 8
        worksheet.write(row, col, i,cell_format0)
        col = 0
    
        row += 1
        for j in range(len(Data_lable)):
            worksheet.write(row, col, Data_lable[j],cell_format1)
            worksheet.write(row, col + 1, Values[j],cell_format1)
            col = 4
            worksheet.write(row, col, Data_lable[j],cell_format1)
            worksheet.write(row, col + 1, Values_n[j],cell_format1)
            col = 8
            worksheet.write(row, col, Data_lable[j],cell_format1)
            worksheet.write(row, col + 1, Values_n_o[j],cell_format1)
    
            col = 0
            
    #        print (i," ===  ",Data_lable[j],"  ,  ",Values[j])
            row += 1
        
        row += 2
    
    row = 0
    col = 12
    worksheet.write(row, col, "Bool before process",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    
    row += 2
    
    
    for i in Bool_Column:
    #    vis.Plot_His_and_Box(df,i)
        df_temp = df[i]
        df_sub = df_temp.describe() 
        Values_temp = df_sub.values.tolist()
        Count_1 = df_temp.sum()
        Count_0 = Values_temp[0]-df_temp.sum()
        mode = df_temp.mode()
        Null=  df_temp.isnull().sum() 
        Data_lable = ["count","mean","std","1-count","0-Count","mode",""," ","Null "]
        Values = Values_temp[0:3]+ [Count_1,Count_0,mode," "," ",Null]
        worksheet.write(row, col, i,cell_format0)
    
        row += 1
        for j in range(len(Data_lable)):
            worksheet.write(row, col, Data_lable[j],cell_format1)
            worksheet.write(row, col + 1, Values[j],cell_format1)
    #        print (i," ===  ",Data_lable[j],"  ,  ",Values[j])
            row += 1
        
        row += 2
    
        
     
    #for i in Bool_Column:
    #    df_temp = df[i]
    #    print (df_temp.describe())
        
    #for i in Num_Column:
    #    vis.Plot_His_and_Box(df,i)
    #
    #for i in Bool_Column:
    #    vis.Plot_His_and_Box(df,i)
    #    
    #    
    """==================================================================================
    
        List Wise Deltion
        
    ====================================================================================="""
        
    df_LD = df.copy() 
    df_LD_n = df.copy()
    df_LD.dropna(inplace=True)
    df_LD_n.dropna(inplace=True)
    
    df_LD_n[Num_Column] = df_LD_n[Num_Column].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    df_LD_o= df_LD_n.copy()
    df_LD_o= df_LD_o.loc[ : , ["X"]+Num_Column]
    
    Q1 = df_LD_o.quantile(0.25)
    Q3 = df_LD_o.quantile(0.75)
    IQR = Q3 - Q1
    
    df_LD_o = df_LD_o[~((df_LD_o < (Q1 - Lower * IQR)) |(df_LD_o > (Q3 + Upper * IQR))).any(axis=1)]
    df_Numeric_Out_rem_LD =df_LD_o
    df_Bool_LD =df_LD.copy()
    df_Bool_LD =df_Bool_LD.loc[ : , ["X"]+Bool_Column]
    
    df_LD_o =pd.merge(df_Numeric_Out_rem_LD, df_Bool_LD, on='X')
    
     
    """df_LD_o is row data set---->> Remove missing value List wise deltion 
     -->> Normalized (0,1) --->> remove Outliers based on Q+ IQR*Factor    """
    
    
    #df_LD_n= df_LD_n.div(df_LD_n.max(),1)
     
    worksheet1 = workbook.add_worksheet("List_DEL")
    
    
    row = 0
    col = 0
    
    
    worksheet1.write(row, col, "Numerical list wise deletion",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    col = 4
    worksheet1.write(row, col, "Numerical list wise deletion->Normlaized-",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    col = 8
    worksheet1.write(row, col, "Numerical list wise deletion->Normlaized-> outlier",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    col = 0
    row += 2
    
    for i in Num_Column:
    #    vis.Plot_His_and_Box(df,i)
        df_temp = df_LD[i]
        df_sub = df_temp.describe() 
        Values = df_sub.values.tolist()+[ df_temp.isnull().sum()]
        Data_lable = ["count","mean","std","min","Q1-25%","Q2-50%","Q3-75%","max","null"]
    
        df_temp_n = df_LD_n[i]
        df_sub_n = df_temp_n.describe() 
        Values_n = df_sub_n.values.tolist()+[ df_temp_n.isnull().sum()]
        
        df_temp_n_o = df_LD_o[i]
        df_sub_n_o = df_temp_n_o.describe() 
        Values_n_o = df_sub_n_o.values.tolist()+[ df_temp_n_o.isnull().sum()]
        
        
            
        worksheet1.write(row, col, i,cell_format0)
        col = 4
        worksheet1.write(row, col, i,cell_format0)
        col = 8
        worksheet1.write(row, col, i,cell_format0)
        col = 0
    
        row += 1
        for j in range(len(Data_lable)):
            worksheet1.write(row, col, Data_lable[j],cell_format1)
            worksheet1.write(row, col + 1, Values[j],cell_format1)
    
            col = 4
            worksheet1.write(row, col, Data_lable[j],cell_format1)
            worksheet1.write(row, col + 1, Values_n[j],cell_format1)
            col = 8
            worksheet1.write(row, col, Data_lable[j],cell_format1)
            worksheet1.write(row, col + 1, Values_n_o[j],cell_format1)
    
    
            col = 0
    
    
            row += 1
        
        row += 2
    
    row = 0
    col = 12
    worksheet1.write(row, col, "Bool list wise deletion",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    row += 2
    
    
    for i in Bool_Column:
    #    vis.Plot_His_and_Box(df,i)
        df_temp = df_LD[i]
        df_sub = df_temp.describe() 
        Values_temp = df_sub.values.tolist()
        Count_1 = df_temp.sum()
        Count_0 = Values_temp[0]-df_temp.sum()
        mode = df_temp.mode()
        Null=  df_temp.isnull().sum() 
        Data_lable = ["count","mean","std","1-count","0-Count","mode",""," ","Null "]
        Values = Values_temp[0:3]+ [Count_1,Count_0,mode," "," ",Null]
        worksheet1.write(row, col, i,cell_format0)
    
        row += 1
        for j in range(len(Data_lable)):
            worksheet1.write(row, col, Data_lable[j],cell_format1)
            worksheet1.write(row, col + 1, Values[j],cell_format1)
    #        print (i," ===  ",Data_lable[j],"  ,  ",Values[j])
            row += 1
        
        row += 2
    
    
    """==================================================================================
    
       Mean , Mode imputation
        
    ====================================================================================="""
    
    df_MMI = df.copy() 
    df_MMI_n = df.copy() 
    df_MMI_n[Num_Column] = df_MMI_n[Num_Column].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    
    df_MMI_o= df_MMI_n.copy()
    df_MMI_o= df_MMI_o.loc[ : , ["X"]+Num_Column]
    
    Q1 = df_MMI_o.quantile(0.25)
    Q3 = df_MMI_o.quantile(0.75)
    IQR = Q3 - Q1
    
    df_MMI_o = df_MMI_o[~((df_MMI_o < (Q1 - Lower * IQR)) |(df_MMI_o > (Q3 + Upper * IQR))).any(axis=1)]
    df_Numeric_Out_rem_MMI =df_MMI_o
    df_Bool_MMI =df_MMI.copy()
    df_Bool_MMI =df_Bool_MMI.loc[ : , ["X"]+Bool_Column]
    
    df_MMI_o =pd.merge(df_Numeric_Out_rem_MMI, df_Bool_MMI, on='X')
    
    """df_MMI_o is row data set---->> Mean mode impute 
     -->> Normalized (0,1) --->> remove Outliers based on Q+ IQR*Factor    """
     
    
    
    #df_MMI_n= df_MMI_n.div(df_MMI_n.max(),1)
    
    worksheet1 = workbook.add_worksheet("M_M Impute")
    
    
    row = 0
    col = 0
    
    
    worksheet1.write(row, col, "Numerical Mean mode Imputation",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    col = 4
    worksheet1.write(row, col, "Numerical Mean mode Imputation-Normalized",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    col = 8
    worksheet1.write(row, col, "Numerical Mean mode Imputation-Normalized-Outlier",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    
    col = 0
    row += 2
    
    for i in Num_Column:
    #    vis.Plot_His_and_Box(df,i)
        df_MMI[i].fillna(df_MMI[i].mean(), inplace=True)
        df_temp=df_MMI[i]
        df_sub = df_temp.describe() 
        Values = df_sub.values.tolist()+[ df_temp.isnull().sum()]
        Data_lable = ["count","mean","std","min","Q1-25%","Q2-50%","Q3-75%","max","null"]
        
        df_MMI_n[i].fillna(df_MMI_n[i].mean(), inplace=True)
        df_temp_n=df_MMI_n[i]
        df_sub_n = df_temp_n.describe() 
        Values_n = df_sub_n.values.tolist()+[ df_temp_n.isnull().sum()]
        
        df_MMI_o[i].fillna(df_MMI_o[i].mean(), inplace=True)
        df_temp_o=df_MMI_o[i]
        df_sub_o = df_temp_o.describe() 
        Values_o = df_sub_o.values.tolist()+[ df_temp_o.isnull().sum()]
      
        
        
        worksheet1.write(row, col, i,cell_format0)
        col = 4
        worksheet1.write(row, col, i,cell_format0)
        col = 8
        worksheet1.write(row, col, i,cell_format0)
        
        col = 0
    
        row += 1
        for j in range(len(Data_lable)):
            worksheet1.write(row, col, Data_lable[j],cell_format1)
            worksheet1.write(row, col + 1, Values[j],cell_format1)
          
            col = 4
            worksheet1.write(row, col, Data_lable[j],cell_format1)
            worksheet1.write(row, col + 1, Values_n[j],cell_format1)
            
            col = 8
            worksheet1.write(row, col, Data_lable[j],cell_format1)
            worksheet1.write(row, col + 1, Values_o[j],cell_format1)
            
            col = 0
            
            row += 1
        
        row += 2
    
    row = 0
    col = 12
    worksheet1.write(row, col, "Bool list wise deletion",workbook.add_format({'bold': True, 'font_color': 'black','border': True}))
    row += 2
    
    
    for i in Bool_Column:
    #    vis.Plot_His_and_Box(df,i)
        df_MMI[i].fillna(df_MMI[i].mode(), inplace=True)
        df_temp=df_MMI[i]
    
        df_sub = df_temp.describe() 
        Values_temp = df_sub.values.tolist()
        Count_1 = df_temp.sum()
        Count_0 = Values_temp[0]-df_temp.sum()
        mode = df_temp.mode()
        Null=  df_temp.isnull().sum() 
        Data_lable = ["count","mean","std","1-count","0-Count","mode",""," ","Null "]
        Values = Values_temp[0:3]+ [Count_1,Count_0,mode," "," ",Null]
        worksheet1.write(row, col, i,cell_format0)
    
        row += 1
        for j in range(len(Data_lable)):
            worksheet1.write(row, col, Data_lable[j],cell_format1)
            worksheet1.write(row, col + 1, Values[j],cell_format1)
    #        print (i," ===  ",Data_lable[j],"  ,  ",Values[j])
            row += 1
        
        row += 2
    
    
    print("Before impution")
    print(df.shape)
    print("After impution")
    print(df_MMI.shape)
    
    Raw_Data = df.copy()
    
    Raw_Norm = df_n.copy()
    Raw_Norm_OutLier=df_n_o.copy()
    
    Lst_Del_Norm= df_LD_n.copy()
    Lst_Del_Norm_OutLier =df_LD_o.copy()
    
    MeMo_Imp_Norm= df_MMI_n.copy()
    MeMo_Imp_Norm_OutLier= df_MMI_o.copy()
    
    print("\n\n")
    print ("Raw_Data========================= : ",Raw_Data.shape )
    print ("Raw_Norm========================= : ",Raw_Norm.shape )
    print ("Raw_Norm_OutLier================= : ",Raw_Norm_OutLier.shape )
    
    print ("Lst_Del_Norm===================== : ",Lst_Del_Norm.shape )
    print ("Lst_Del_Norm_OutLier============= : ",Lst_Del_Norm_OutLier.shape )
    print ("MeMo_Imp_Norm==================== : ",MeMo_Imp_Norm.shape )
    print ("MeMo_Imp_Norm_OutLier============ : ",MeMo_Imp_Norm_OutLier.shape )
    
    
    
    
    worksheet1 = workbook.add_worksheet("Fea_List")
    
    col = 0
    row = 0
    
    worksheet1.write(row, col, "List_Delele_Normlized Numerical ",cell_format0)
    col = 4
    worksheet1.write(row, col, "List_Delele_Normlized Bool ",cell_format0)
    col = 0
    
    row += 1
    FS_Lst_Del_Norm = fea_sel.Anova_Feature_selction (          Lst_Del_Norm.loc[ : , Num_Column+["churn"]],Ratio,No_of_Variable)
    FS_Lst_Del_Bool = fea_sel.Chi_Feature_selction (          Lst_Del_Norm.loc[ : , Bool_Column+["churn"]],Ratio,No_of_Variable+1)
    for i in range(len(FS_Lst_Del_Norm["Specs"])):
        A=FS_Lst_Del_Norm["Specs"]
        B=FS_Lst_Del_Norm["Score"]    
        worksheet1.write(row, col    , A.iloc[i] ,cell_format1)
        worksheet1.write(row, col + 1, B.iloc[i] ,cell_format1)
        
        col = 4
        A=FS_Lst_Del_Bool["Specs"]
        B=FS_Lst_Del_Bool["Score"]    
        if (B.iloc[i]==np.inf):
            B="inf"
        else:
            B=B.iloc[i]   
        worksheet1.write(row, col    , A.iloc[i] ,cell_format1)
        worksheet1.write(row, col + 1, B         ,cell_format1)
        col = 0
        row += 1
    
    row+=2
    worksheet1.write(row, col, "List_Delele_Normlized Outlier Numerical ",cell_format0)
    col = 4
    worksheet1.write(row, col, "List_Delele_Normlized Outlier Bool ",cell_format0)
    col = 0
    row += 1
    FS_Lst_Del_Norm_out = fea_sel.Anova_Feature_selction (  Lst_Del_Norm_OutLier.loc[ : , Num_Column+["churn"]],Ratio,No_of_Variable)
    FS_Lst_Del_Bool_out = fea_sel.Chi_Feature_selction (  Lst_Del_Norm_OutLier.loc[ : , Bool_Column+["churn"]],Ratio,No_of_Variable+1)
    for i in range(len(FS_Lst_Del_Norm_out["Specs"])):
        A=FS_Lst_Del_Norm_out["Specs"]
        B=FS_Lst_Del_Norm_out["Score"]
        worksheet1.write(row, col    , A.iloc[i] ,cell_format1)
        worksheet1.write(row, col + 1, B.iloc[i] ,cell_format1)
        
        col = 4
        A=FS_Lst_Del_Bool_out["Specs"]
        B=FS_Lst_Del_Bool_out["Score"]  
        if (B.iloc[i]==np.inf):
            B="inf"
        else:
            B=B.iloc[i]   
        worksheet1.write(row, col    , A.iloc[i] ,cell_format1)
        worksheet1.write(row, col + 1, B         ,cell_format1)
        col = 0
    
        row += 1
        
    row+=2    
    worksheet1.write(row, col, "Mean ,Mode impute Normlized Numerical ",cell_format0)
    col = 4
    worksheet1.write(row, col, "Mean ,Mode impute Normlized Bool ",cell_format0)
    col = 0
    
    row += 1
    FS_MMI_Norm = fea_sel.Anova_Feature_selction (         MeMo_Imp_Norm.loc[ : , Num_Column+["churn"]],Ratio,No_of_Variable)
    FS_MMI_Bool = fea_sel.Chi_Feature_selction   (         MeMo_Imp_Norm.loc[ : , Bool_Column+["churn"]],Ratio,No_of_Variable+1)
    for i in range(len(FS_MMI_Norm["Specs"])):
        A=FS_MMI_Norm["Specs"]
        B=FS_MMI_Norm["Score"]
        worksheet1.write(row, col    , A.iloc[i] ,cell_format1)
        worksheet1.write(row, col + 1, B.iloc[i] ,cell_format1)
        
        col = 4
        A=FS_MMI_Bool["Specs"]
        B=FS_MMI_Bool["Score"]  
        if (B.iloc[i]==np.inf):
            B="inf"
        else:
            B=B.iloc[i]   
        worksheet1.write(row, col    , A.iloc[i] ,cell_format1)
        worksheet1.write(row, col + 1, B         ,cell_format1)
        col = 0
        row += 1
        
        
    row+=2
    worksheet1.write(row, col, "Mean ,Mode impute Normlized Numerical Outl ",cell_format0)
    col = 4
    worksheet1.write(row, col, "Mean ,Mode impute Normlized Bool Outl ",cell_format0)
    col = 0
    row += 1
    FS_MMI_Norm_out = fea_sel.Anova_Feature_selction ( MeMo_Imp_Norm_OutLier.loc[ : , Num_Column+["churn"]],Ratio,No_of_Variable)
    FS_MMI_Bool_out = fea_sel.Chi_Feature_selction   ( MeMo_Imp_Norm_OutLier.loc[ : , Bool_Column+["churn"]],Ratio,No_of_Variable+1)
    
    for i in range(len(FS_MMI_Norm_out["Specs"])):
        A=FS_MMI_Norm_out["Specs"]
        B=FS_MMI_Norm_out["Score"]
        worksheet1.write(row, col    , A.iloc[i] ,cell_format1)
        worksheet1.write(row, col + 1, B.iloc[i] ,cell_format1)
        
        col = 4
        A=FS_MMI_Bool_out["Specs"]
        B=FS_MMI_Bool_out["Score"]
        if (B.iloc[i]==np.inf):
            B="inf"
        else:
            B=B.iloc[i]       
        worksheet1.write(row, col    , A.iloc[i] ,cell_format1)
        worksheet1.write(row, col + 1, B         ,cell_format1)
        col = 0
        row += 1
    
    def check_availability(element, collection: iter):
        return element 
        
    ##==================================================================================================
    #print ("\n\nLst_Del_Norm===================== : ",Lst_Del_Norm.shape )
    #
    #List_Del_Nor_FL= FS_Lst_Del_Norm["Specs"].tolist()+FS_Lst_Del_Bool["Specs"].tolist()
    #if check_availability('churn', List_Del_Nor_FL)=='churn':
    #    List_Del_Nor_FL.remove('churn')
    #
    #Temp_LD_N_DF =Lst_Del_Norm[List_Del_Nor_FL+["churn"]].copy()
    #print("No of Featues  ==== : ",len(List_Del_Nor_FL))
    #
    
    #X = Temp_LD_N_DF.iloc[:,:-1]   #independent columns
    #y = Temp_LD_N_DF.iloc[:,-1]    #target column i.e price range
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Ratio, random_state=0) 
    #A=Gas_Cla.NB_Classifires(X_train, X_test, y_train, y_test)
    
    
    ##==================================================================================================
    
    print ("\n\nLst_Del_Norm_OutLier============= : ",Lst_Del_Norm_OutLier.shape)
    (DF_LD_O_No_Row,DF_LD_O_No_Col )=Lst_Del_Norm_OutLier.shape
    

    List_Del_Nor_OUT_FL= FS_Lst_Del_Norm_out["Specs"].tolist()+FS_Lst_Del_Bool_out["Specs"].tolist()
#    if check_availability('churn', List_Del_Nor_OUT_FL)=='churn':
#        List_Del_Nor_OUT_FL.remove('churn')
    
    Temp_LD_N_O_DF=Lst_Del_Norm_OutLier[List_Del_Nor_OUT_FL+["churn"]].copy()
    print("No of Featues  ==== : ",len(List_Del_Nor_OUT_FL))

    X = Temp_LD_N_O_DF.iloc[:,:-1]   #independent columns
    y = Temp_LD_N_O_DF.iloc[:,-1]    #target column i.e price range
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Ratio, random_state=0) 
    
    (DF_LD_O_No_Train_Row ,DF_LD_O_No_Train_Col )=X_train.shape
    (DF_LD_O_No_Test_Row  ,DF_LD_O_No_Test_Col  )=X_test.shape

    LD_N_O_Acc=Gas_Cla.NB_Classifires(X_train, X_test, y_train, y_test)
    print("LD_N_O_Acc",LD_N_O_Acc)
    
    [NB_LD_Acc,DT_LD_Acc,SVM_LD_Acc,NBDT_LD_Acc,DTSVM_LD_Acc ,SVMNB_LD_Acc]=LD_N_O_Acc
    ##==================================================================================================
    
    #print ("\n\nMean Mode impute Norm_============= : ",MeMo_Imp_Norm.shape)
    #MMI_Nor_FL= FS_MMI_Norm["Specs"].tolist()+FS_MMI_Bool["Specs"].tolist()
    #if check_availability('churn', MMI_Nor_FL)=='churn':
    #    MMI_Nor_FL.remove('churn')
    #
    #Temp_MMI_N_DF=MeMo_Imp_Norm[MMI_Nor_FL+["churn"]].copy()
    #print("No of Featues  ==== : ",len(MMI_Nor_FL))
    
    #X = Temp_MMI_N_DF.iloc[:,:-1]   #independent columns
    #y = Temp_MMI_N_DF.iloc[:,-1]    #target column i.e price range
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Ratio, random_state=0) 
    #A=Gas_Cla.NB_Classifires(X_train, X_test, y_train, y_test)
    
    
    
    ##==================================================================================================
    
    print ("\n\nMeMo_Imp_Norm_OutLier============ : ",MeMo_Imp_Norm_OutLier.shape )
    MMI_Nor_OUT_FL= FS_MMI_Norm_out["Specs"].tolist()+FS_MMI_Bool_out["Specs"].tolist()
#    if check_availability('churn', MMI_Nor_OUT_FL)=='churn':
#        MMI_Nor_OUT_FL.remove('churn')

    (DF_MMI_O_No_Row,DF_MMI_O_No_Col)=MeMo_Imp_Norm_OutLier.shape
    
    Temp_MMI_N_O_DF=MeMo_Imp_Norm_OutLier[MMI_Nor_OUT_FL+["churn"]].copy()
    print("No of Featues  ==== : ",len(MMI_Nor_OUT_FL))
    
    X = Temp_MMI_N_O_DF.iloc[:,:-1]   #independent columns
    y = Temp_MMI_N_O_DF.iloc[:,-1]    #target column i.e price range

    print("No of Shape  ==== : ",(X.shape))
    print("No of Head  ==== : ",(X.head()))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Ratio, random_state=0) 
    
    (DF_MMI_O_No_Train_Row, DF_MMI_O_No_Train_Col)=X_train.shape
    (DF_MMI_O_No_Test_Row, DF_MMI_O_No_Test_Col)=X_test.shape
    
   
    MMI_N_O_Acc =Gas_Cla.NB_Classifires(X_train, X_test, y_train, y_test)
    [NB_MMI_Acc,DT_MMI_Acc,SVM_MMI_Acc,NBDT_MMI_Acc,DTSVM_MMI_Acc ,SVMNB_MMI_Acc] = MMI_N_O_Acc
    workbook.close()
    
    
    y_pred_GNB = clf_GNB.predict(X1) 
    y_pred_DT = clf_DT.predict(X1) 
    y_pred_SVM = clf_SVM.predict(X1) 
    y_pred_NB_DT = clf_NB_DT.predict(X1) 
    y_pred_DT_SVM = clf_SVM_DT.predict(X1) 
    y_pred_SVM_NB = clf_NB_SVM.predict(X1) 
    
    Print ("y_pred_GNB     : = " , y_pred_GNB)
    Print ("y_pred_DT      : = " , y_pred_DT)
    Print ("y_pred_SVM     : = " , y_pred_SVM)
    Print ("y_pred_NB_DT   : = " , y_pred_NB_DT)
    Print ("y_pred_DT_SVM  : = " , y_pred_DT_SVM)
    Print ("y_pred_SVM_NB  : = " , y_pred_SVM_NB)
   
    
#    return ([DF_Shape,DF_No_Col ,DF_No_Row ,DF_LD_O_No_Row,DF_LD_O_No_Col, DF_LD_O_No_Train_Row ,DF_LD_O_No_Train_Col,DF_LD_O_No_Test_Row  ,DF_LD_O_No_Test_Col,DF_MMI_O_No_Row,DF_MMI_O_No_Col,DF_MMI_O_No_Train_Row, DF_MMI_O_No_Train_Col,DF_MMI_O_No_Test_Row, DF_MMI_O_No_Test_Col, NB_LD_Acc,DT_LD_Acc,SVM_LD_Acc,NBDT_LD_Acc  ,DTSVM_LD_Acc ,SVMNB_LD_Acc])
    return ([DF_Shape,DF_No_Col ,DF_No_Row ,DF_LD_O_No_Col,DF_LD_O_No_Row, DF_LD_O_No_Train_Col ,DF_LD_O_No_Train_Row,DF_LD_O_No_Test_Col  ,DF_LD_O_No_Test_Row,DF_MMI_O_No_Col,DF_MMI_O_No_Row,DF_MMI_O_No_Train_Col, DF_MMI_O_No_Train_Row,DF_MMI_O_No_Test_Col, DF_MMI_O_No_Test_Row, NB_LD_Acc,DT_LD_Acc,SVM_LD_Acc,NBDT_LD_Acc  ,DTSVM_LD_Acc ,SVMNB_LD_Acc,NB_MMI_Acc,DT_MMI_Acc,SVM_MMI_Acc,NBDT_MMI_Acc,DTSVM_MMI_Acc ,SVMNB_MMI_Acc],[FS_Lst_Del_Norm_out,FS_Lst_Del_Bool_out])

A,B =main(1.5,1.5,0.2,10)
