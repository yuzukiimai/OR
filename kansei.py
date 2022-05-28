import numpy as np
from sklearn.metrics import confusion_matrix



#重みの初期値を生成
w = np.random.rand(3)*2-1   # -1.0 ～ 1.0 までの値をランダムに3つ生成

# w = np.array([1.0, 2.0, -3.0]) #自分で値決めるときはこっち使う



#重みの初期値を表示
print("init W = {0}".format(w))



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



#学習前の混同行列用
def or_func1(x1, x2, x3):
    X = np.array([[0,0,0],
                  [0,1,0],
                  [0,0,1],
                  [0,1,1],
                  [1,0,0],
                  [1,0,1],
                  [1,1,0],
                  [1,1,1]])  
    a = w[0]*x1 + w[1]*x2 + w[2]*x3
    if a > 0:               
        return 1
    else:
        return 0



#混同行列を作成するための準備
lists1 = []
lists1.append(or_func1(0,0,0))
lists1.append(or_func1(0,1,0))
lists1.append(or_func1(0,0,1))
lists1.append(or_func1(0,1,1))
lists1.append(or_func1(1,0,0))
lists1.append(or_func1(1,0,1))
lists1.append(or_func1(1,1,0))
lists1.append(or_func1(1,1,1))
print(lists1)   #結果確認用



#学習前の混同行列を表示
y_true = [0,1,1,1,1,1,1,1]   #真値
y_pred = lists1   #学習前の重みでの値
cm = confusion_matrix(y_true, y_pred)
print(cm)



#学習回数
epoch = 500


#回数初期化
i=0



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



#学習ループ
for e in range(epoch):
  index = i%8


  #OR関数の計算
  Y = or_3in(X[index])


  #誤り訂正量の計算
  dw = w_update(w,Y,index)
  w += dw


  #途中計算の表示
  print("{5} : x = {3} : y = {0} : t = {1} : w = {2} : dw = {4}".format(Y,y[index],w,X[index],dw,i))  # y: 出力値  t: 真値


  i += 1


  
def or_func2(x1, x2, x3):
    X = np.array([[0,0,0],
                  [0,1,0],
                  [0,0,1],
                  [0,1,1],
                  [1,0,0],
                  [1,0,1],
                  [1,1,0],
                  [1,1,1]])
      
    b = w[0]*x1 + w[1]*x2 + w[2]*x3
    if b > 0:               
        return 1
    else:
        return 0



#混同行列を作成するための準備
lists2 = []
lists2.append(or_func2(0,0,0))
lists2.append(or_func2(0,1,0))
lists2.append(or_func2(0,0,1))
lists2.append(or_func2(0,1,1))
lists2.append(or_func2(1,0,0))
lists2.append(or_func2(1,0,1))
lists2.append(or_func2(1,1,0))
lists2.append(or_func2(1,1,1))
print(lists2)   #結果確認用



#学習後の混同行列を表示
y_true = [0,1,1,1,1,1,1,1]   #真値
y_pred = lists2   #学習後の重みでの値
cm = confusion_matrix(y_true, y_pred)
print(cm)



#学習後の重みを表示
print("W = {0}".format(w))
