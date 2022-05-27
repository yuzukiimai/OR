import numpy as np

#重み初期値
w = np.array([1.0, 2.0, -3.0])

#入力
input_array = np.array([[0,0,0],
                        [0,1,0],
                        [0,0,1],
                        [0,1,1],
                        [1,0,0],
                        [1,0,1],
                        [1,1,0],
                        [1,1,1]])

#正解データ
training_data = np.array([1,1,1,1,1,1,1,0])



#パーセプトロンによるAND関数
def OR(x):
  temp = np.sum(x*w)
  if temp <= 0:
    return 0
  else:
    return 1

#誤り訂正量の計算
def w_update(w,y,i):
  eta = 0.1
  return eta*(training_data[i] - y)*input_array[i]

i = 0
zeros = 0

print("init W = {0}".format(w))


#学習ループ
while 1:
  index = i%8

  #AND関数の計算
  y = OR(input_array[index])

  #誤り訂正量の計算
  dw = w_update(w,y,index)
  w += dw

  #途中計算の表示デフォルトではコメントアウト
  print("{5} : x = {3} : y = {0} : t = {1} : w = {2} : dw = {4}".format(y,training_data[index],w,input_array[index],dw,i))

  #重みの訂正量が4回続けて0の時学習を終了する
  if (np.sum(dw) == 0):
    zeros += 1
  else:
    zeros = 0
  if (zeros >= 3):
    break

  i += 1

#学習結果表示
print("W = {0}".format(w))
