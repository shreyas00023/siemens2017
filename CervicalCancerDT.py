import csv
import sys
from sklearn import tree
from sklearn.metrics import accuracy_score
import random
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


def getData(file):
    header = []
    data = []
    labels = []
    with open(file) as infile:
        reader = csv.reader(infile)
        for row in reader:
            header = row
            break
        header = header[0:]
        rowscount = 0
        for row in reader:
            rowscount+=1
            tempdata=[]
            for s in row[1:31]:
                if s=="?": tempdata.append(0)
                else: tempdata.append(float(s))
            data.append(tempdata)
            temp = row[32:]
            templabel = [int(s) for s in temp]
            templabel[3] = templabel[3]*2
            labels.append(sum(templabel))
    return header, data,labels
def makeTraningTest(trainingsize, data,labels):
    dataset = [[j for j in i] for i in data]
    labelsset = labels.copy()
    trainingdata = []
    traininglabels = []
    for i in range(trainingsize):
        num = random.randrange(0,len(dataset)-1)
        trainingdata.append(dataset.pop(num))
        traininglabels.append(labelsset.pop(num))
    return trainingdata,traininglabels,dataset,labelsset
def decisionTree(trainingdata,traininglabels, testdata, testlabels):
    clf = tree.DecisionTreeClassifier(criterion="entropy",min_samples_leaf=7,splitter="random")
    clf = clf.fit(trainingdata, traininglabels)
    pred = clf.predict(testdata)
    return accuracy_score(testlabels,pred)
def neuralNetwork(trainingdata,traininglabels, testdata, testlabels):
    clf = MLPClassifier(solver='adam',hidden_layer_sizes=(9,7,5),learning_rate="adaptive")
    clf = clf.fit(trainingdata, traininglabels)
    pred = clf.predict(testdata)
    return accuracy_score(testlabels, pred), clf

one = getData("risk_factors_cervical_cancer.csv")
# accmax = 0
# sizemax = 0
#average_acc = []
#training_size = []
# for i in range(700, 715, 2):
#     accuaracies = []
#     for n in range(5):
#         data = makeTraningTest(i,one[1],one[2])
#         acc = neuralNetwork(data[0],data[1],data[2],data[3])
#         accuaracies.append(acc)
#         # if acc>accmax:
#         #     accmax=acc
#         #     sizemax=i
#         #     #print(i,acc)
#         # print(sizemax,accmax)
#     print(i)
#     average_acc.append(100*sum(accuaracies)/len(accuaracies))
#     training_size.append(i)
# plt.plot(training_size, average_acc, 'ro')
# plt.axis([700, 750, 80, 100])
# plt.show()
#for i in range(50):
#    data = makeTraningTest(704, one[1],one[2])
#    acc = neuralNetwork(data[0],data[1],data[2],data[3])
#    average_acc.append(acc)
# print(sum(average_acc)/50)
data = makeTraningTest(704, one[1],one[2])
clf = neuralNetwork(data[0],data[1],data[2],data[3])[1]



a = float(sys.argv[1])
b = float(sys.argv[2])
c = float(sys.argv[3])
d = float(sys.argv[4])
e = float(sys.argv[5])
f = float(sys.argv[6])
g = float(sys.argv[7])
h = float(sys.argv[8])
i = float(sys.argv[9])
j = float(sys.argv[10])
k = float(sys.argv[11])
l = float(sys.argv[12])
m = float(sys.argv[13])
n = float(sys.argv[14])
o = float(sys.argv[15])
p = float(sys.argv[16])
q = float(sys.argv[17])
r = float(sys.argv[18])
s = float(sys.argv[19])
t = float(sys.argv[20])
u = float(sys.argv[21])
v = float(sys.argv[22])
w = float(sys.argv[23])
x = float(sys.argv[24])
y = float(sys.argv[25])
z = float(sys.argv[26])
a1 = float(sys.argv[27])
a2 = float(sys.argv[28])
a3 = float(sys.argv[29])
a4 = float(sys.argv[30])
a5 = float(sys.argv[31])

pred = clf.predict([[b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,a1,a2,a3,a4,a5]])[0]

if pred==0:
    print("Very Low Risk of Cervical Cancer")
if pred==1:
    print("Low Risk of Cervical Cancer")
if pred==2:
    print("Slight Risk of Cervical Cancer")
if pred==3:
    print("Medium Risk of Cervical Cancer")
if pred==4:
    print("High Risk of Cervical Cancer")
if pred==5:
    print("Very High Risk of Cervical Cancer, See a Doctor")
