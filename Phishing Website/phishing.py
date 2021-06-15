import pandas as pd
import numpy as np
import timeit 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 

#   Veri setinin tanımlanması ve niteliklerin ayrılması
#   Veri seti: https://data.mendeley.com/datasets/72ptz43s9v/1
dataset=pd.read_csv('dataset.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

#   Eğitim ve Test verilerilinin tanımlanması
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

#   her testin sonunda dogruluk degerini ekrana yazan fonksiyon
def test_Sonuc(Y_pred,model,cm,time):
    tp = cm[0,0]
    tn = cm[1,1]
    fp = cm[0,1]
    fn = cm[1,0]
    toplam =tp+tn+fp+fn
    acc =(tp+tn)/toplam*100
    sens=tp/(tp+fn)*100
    prec=tp/(tp+fp)*100
    spec=tn/(tn+fp)*100
    fpr =fp/(tn+fp)
    print("\n",model)
    print("\t",acc,"% Başarılı")
    print("\t",sens,"% Duyarlı")
    print("\t",prec,"% Hassas")
    print("\t",spec,"% Öznel")
    print("\t",fpr,"False Positive Rate")
    print("\t",time,"Saniye Eğitim ve Test Süresi")
    
    # Grafiklerin olusturulmasi
    '''graf=('Basarı','Duyarlılık','Hassasiyet','Öznellik')
    y_pos=np.arange(len(graf))
    performance=[acc,sens,prec,spec] 
    plt.bar(y_pos, performance, align='center', alpha=1)
    plt.xticks(y_pos, graf)
    plt.ylabel('Oranı (%)')
    plt.title(model)
    plt.show()'''
    
#   Normalizasyon
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#   Lojistik Regresyon modeli
from sklearn.linear_model import LogisticRegression
classifierLR=LogisticRegression(random_state=(0),max_iter=500)
classifierLR.fit(X_train,Y_train) # Eğitim bloğu
start = timeit.default_timer() # Çalışma zaman ölçümü için timer baslangıç 
Y_pred=classifierLR.predict(X_test) # Eğitim sonrasi test bloğu
stop = timeit.default_timer() # Timer bitiş
time=stop-start #  Süreyi hesaplama
cmLR=confusion_matrix(Y_test, Y_pred)  # Confusion Matrix
# C.matrix üzerinden doğruluk hesaplayan ve diğer bilgileri ekrana yazdıran fonsksiyon çağrımı
test_Sonuc(Y_pred,"Lojistik Regresyon",cmLR,time) 

#   Karar agaci Modeli
from sklearn.tree import DecisionTreeClassifier

classifierDT=DecisionTreeClassifier(random_state=(5))
classifierDT.fit(X_train,Y_train)
start = timeit.default_timer()
Y_pred=classifierDT.predict(X_test)
stop = timeit.default_timer()
time=stop-start
cmDT=confusion_matrix(Y_test, Y_pred)
test_Sonuc(Y_pred,"Karar Ağacı",cmDT,time)

#   Random Forest Modeli
from sklearn.ensemble import RandomForestClassifier

classifierRF=RandomForestClassifier(n_estimators=60)
classifierRF.fit(X_train,Y_train)
start = timeit.default_timer()
Y_pred=classifierRF.predict(X_test)
stop = timeit.default_timer()
time=stop-start
cmRF=confusion_matrix(Y_test, Y_pred)
test_Sonuc(Y_pred,"Random Forest",cmRF,time)

#   K-Nearest Neighbors Modeli
from sklearn.neighbors import KNeighborsClassifier

classifierKN=KNeighborsClassifier()
classifierKN.fit(X_train,Y_train)
start = timeit.default_timer()
Y_pred=classifierKN.predict(X_test)
stop = timeit.default_timer()
time=stop-start
cmKN=confusion_matrix(Y_test, Y_pred)
test_Sonuc(Y_pred,"K-Nearest Neighbors",cmKN,time)

#   Yardimci Vektor Makinesi Modeli
from sklearn import svm

classifierSVM=svm.SVC()
classifierSVM.fit(X_train,Y_train)
start = timeit.default_timer()
Y_pred=classifierSVM.predict(X_test)
stop = timeit.default_timer()
time=stop-start
cmSVM=confusion_matrix(Y_test, Y_pred)
test_Sonuc(Y_pred,"Support Vector Machine",cmSVM,time)

#   Cok Katmanli Algilayici Modeli
from sklearn.neural_network import MLPClassifier

classifierMLP=MLPClassifier(max_iter=500)
classifierMLP.fit(X_train,Y_train)
start = timeit.default_timer()
Y_pred=classifierMLP.predict(X_test)
stop = timeit.default_timer()
time=stop-start
cmMLP=confusion_matrix(Y_test, Y_pred)
test_Sonuc(Y_pred,"Multilayer Perceptron",cmMLP,time)


