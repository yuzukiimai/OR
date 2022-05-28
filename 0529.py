import numpy as np
from sklearn.metrics import confusion_matrix


#重み初期値
w = np.random.rand(3)*2-1

#w = np.array([1.0, 2.0, 3.0]) 自分で値決めるときはこっち使う


#入力データ
X = np.array([[0,0,0],
              [0,1,0],
              [0,0,1],
              [0,1,1],
              [1,0,0],
              [1,0,1],
              [1,1,0],
              [1,1,1]])


#正解データ
y = np.array([0,1,1,1,1,1,1,1])





#学習前 混同行列（未完成）
y_true = [0,1,1,1,1,1,1,1] #正しい値

#---------------------------------------------------------------------
y_pred = [0,1,1,1,1,1,1,1] #ここをどうやって導出するか？
#---------------------------------------------------------------------

cm = confusion_matrix(y_true, y_pred)
print(cm)






#学習回数
epoch = 100


#パーセプトロンによるOR関数
def or_3in(x):
  h = np.sum(x*w)
  if h > 0:
    return 1
  else:
    return 0



#誤り訂正量の計算
def w_update(w,Y,i):
  learning_late = 0.01
  return learning_late*(y[i] - Y)*X[i]

i = 0

print("init W = {0}".format(w))



#学習ループ
for e in range(epoch):
  index = i%8

  #OR関数の計算
  Y = or_3in(X[index])

  #誤り訂正量の計算
  dw = w_update(w,Y,index)
  w += dw
  print("{5} : x = {3} : y = {0} : t = {1} : w = {2} : dw = {4}".format(Y,y[index],w,X[index],dw,i))  # y: 出力値  　t: 真値

  i += 1






#学習後 混同行列（未完成）
y_true = [0,1,1,1,1,1,1,1] #正しい値

#---------------------------------------------------------------------
y_pred = [0,1,1,1,1,1,1,1] #ここをどうやって導出するか？
#---------------------------------------------------------------------

cm = confusion_matrix(y_true, y_pred)
print(cm)






#学習結果表示
print("W = {0}".format(w))
