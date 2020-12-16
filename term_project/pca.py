import numpy as np
from PIL import Image
# do not import any other modules

#pre-defined util
def show_np_arr(np_arr, filename):#np_arr should be 2-dim
  tmp = np.real(np_arr)
  tmp = (tmp - np.min(tmp))/(np.max(tmp) - np.min(tmp))
  tmp = np.clip(255*tmp, 0, 255)
  tmp = Image.fromarray(np.uint8(tmp)).convert('RGB')
  tmp.save(filename)
  tmp.show()  

def get_row_col(K):
  ret = 1
  for i in range(int(np.sqrt(K))):
    if K % (i+1) == 0:
      ret = i+1
  return ret, K // ret


#load and pre-process dataset, do not modify here
x_train = np.load("./x_train.npy")/255.
y_train = np.load("./y_train.npy")
x_test = x_train[10]

idx = np.argsort(y_train)
x_train = x_train[idx]
x_train = x_train[::200]


M, height, width = x_train.shape

#Q1. Eigenface
##step1: compute mean of x_train
x_train = x_train.reshape((x_train.shape[0], -1))
mu = np.mean(x_train,axis=0)

##step2: subtract the mean
phi = x_train - mu

##step3: compute covariance C
#cov = np.cov(phi.T)
cov = np.dot(phi.T, phi) / M

#step4: Compute eigenvector of C, you don't need to do anything at step4.
eigenvalues, eigenvec = np.linalg.eig(cov)

# Sort in descending order
idx = eigenvalues.argsort()[::-1]   
eigenvalues = eigenvalues[idx]
eigenvec = eigenvec[:,idx]
eigenvec = eigenvec.T
print("Shape of eigen vectors = ",eigenvec.shape)

##step5: choose K
ratio = 0
ev_sum = np.sum(eigenvalues)
p = []
for eigenvalue in eigenvalues:
    ratio += eigenvalue
    p.append(ratio)
p /= ev_sum

K = np.sum(p < 0.85)
print("Chosen K: ",K)
K = 40


##step6: show top K eigenfaces. use show_np_arr func.
row, col = get_row_col(K)
output = np.zeros((height*row, width*col), dtype='complex_')
for i in range(K):
  x = (i // col) * height
  y = (i % col) * width
  eigenface = eigenvec[i].reshape(height, width)
  eigenface = (eigenface - np.min(eigenface))/(np.max(eigenface) - np.min(eigenface))
  output[x:x+height, y:y+width] = eigenface

show_np_arr(output, "Q1_{}_Eigenfaces.png".format(K))


#Q2. Image Approximation
x = x_test
x = x.flatten()

##step1: approximate x as x_hat with top K eigenfaces and show x_hat
w = np.dot(eigenvec[:K], x-mu)
x_hat = mu + np.dot(w, eigenvec[:K])

##step2: compater mse between x and x_hat by changing the number of the eigenfaces used for reconstruction (approximation) from 1 to K
output = np.zeros((height*row, width*col), dtype='complex_')
for i in range(1, K+1):
  w = np.dot(eigenvec[:i], x-mu)
  x_hat = mu + np.dot(w, eigenvec[:i])
  mse = np.square(x - x_hat).mean()
  print(i, np.real(mse))
  
  r = ((i-1) // col) * height
  c = ((i-1) % col) * width
  x_hat = (x_hat - np.min(x_hat))/(np.max(x_hat) - np.min(x_hat))
  output[r:r+height, c:c+width] = x_hat.reshape(28,28)  
  
show_np_arr(output, "Q2_{}_Reconstruction.png".format(K))



#Q3. Implement fast version of you algorithm in Q1. Show top 'K' eigenfaces using show_np_arr(...)
print('\n\nFast Implementation!')
#cov = np.cov(phi)
cov = np.dot(phi, phi.T) / M

eigenvalues, eigenvec = np.linalg.eig(cov)

idx = eigenvalues.argsort()[::-1]   
eigenvalues = eigenvalues[idx]
eigenvec = eigenvec[:,idx]
eigenvec = eigenvec.T
print("Shape of eigen vectors = ",eigenvec.shape)

ratio = 0
ev_sum = np.sum(eigenvalues)
p = []
for eigenvalue in eigenvalues:
    ratio += eigenvalue
    p.append(ratio)
p /= ev_sum

K = np.sum(p < 0.85)
print("Chosen K: ",K)
#K = 40

row, col = get_row_col(K)
output = np.zeros((height*row, width*col), dtype='complex_')
eigenvec_origin = np.zeros((M, height*width))
for i in range(K):
  x = (i // col) * height
  y = (i % col) * width
  eigenvec_origin[i] = np.dot(phi.T, eigenvec[i])
  norm = np.linalg.norm(eigenvec_origin[i])
  eigenvec_origin[i] /= norm

  eigenface = eigenvec_origin[i].reshape(height, width)
  eigenface = (eigenface - np.min(eigenface))/(np.max(eigenface) - np.min(eigenface))
  output[x:x+height, y:y+width] = eigenface

show_np_arr(output, "Q3_{}_Fast_Eigenfaces.png".format(K))


x = x_test
x = x.flatten()

w = np.dot(eigenvec_origin[:K], x-mu)
x_hat = mu + np.dot(w, eigenvec_origin[:K])

output = np.zeros((height*row, width*col), dtype='complex_')
for i in range(1, K+1):
  w = np.dot(eigenvec_origin[:i], x-mu)
  x_hat = mu + np.dot(w, eigenvec_origin[:i])
  mse = np.square(x - x_hat).mean()
  print(i, np.real(mse))
  
  r = ((i-1) // col) * height
  c = ((i-1) % col) * width
  x_hat = (x_hat - np.min(x_hat))/(np.max(x_hat) - np.min(x_hat))
  output[r:r+height, c:c+width] = x_hat.reshape(28,28)  
  
show_np_arr(output, "Q3_{}_Fast_Reconstruction.png".format(K))