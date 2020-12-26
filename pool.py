import matplotlib.pyplot as plt
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle
from tqdm import tqdm
import scipy.stats as ss


def extract_data(filename, num_images, IMAGE_WIDTH):
    '''
    Extract images by reading the file bytestream. Reshape the read values into a 3D matrix of dimensions [m, h, w], where m 
    is the number of training examples.
    '''
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
        return data

# def Spearman(output,expected):
#     n=len(expected)
#     dif=0
#     diflist=np.array([])
#     for i in range (n):
#         diflist=np.append(diflist,[np.power(output[i]-expected[i],2)])
#         dif+=np.power(output[i]-expected[i],2)

#     den=n*(np.power(n,2)-1)
#     deflist=np.array([])
#     for dd in diflist:
#       deflist=np.append(deflist,((6*dd)/(den)))
#     return deflist   
# 
def SSS(xi,nlabel,bx):
    sum2 = 0
    s=100
    b=100/bx
    t=200
    for i in range(nlabel):
        xx=s-((i*t)/(nlabel-1))
        sum2 +=0.5*(np.tanh((-b*(xi))-(xx)))
    sum2=-1*sum2     
    sum2= sum2+(nlabel*0.5)  
    return sum2   


def Spearman(output,expected):
    n=len(expected)
    nem=0
    for i in range (n):
        nem+=np.power(output[i]-expected[i],2) 
    den=n*(np.power(n,2)-1)
    bb=1-((6*nem)/(den)) 
    return bb #1-((6*nem)/(den))

def extract_labels(filename, num_images):
    '''
    Extract label into vector of integer values of dimensions [m, 1], where m is the number of images.
    '''
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def initializeFilter(size, scale = 1.0):
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)


def RankingConvolution(image, s=1):
    '''
    Confolves `filt` over `image` using stride `s`
    '''
    (n_f, n_c_f, f, _) = filt.shape # filter dimensions
    n_c, in_dim, _ = image.shape # image dimensions
    
    out_dim = int((in_dim - f)/s)+1 # calculate output dimensions
    
#     assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"
    
    out = np.zeros((n_f,out_dim,out_dim))
    

    curr_f=0
    # curr_yb = out_y = 0
    # while curr_yb + f <= in_dim:
    #     curr_xb = out_x = 0
    #     while curr_xb + f <= in_dim:
    curr_y = out_y = 0#curr_yb
    while curr_y + f <= in_dim:
        curr_x = out_x = 0#curr_xb
        while curr_x + f <= in_dim:
            baserect=image[:,0:0+f, 0:0+f]
            steprect=image[:,curr_y:curr_y+f, curr_x:curr_x+f]
            baserectFlat=baserect[0].flatten()
            steprectFlat=steprect[0].flatten()
            roh = Spearman(ss.rankdata(baserectFlat),ss.rankdata(steprectFlat))
            if np.isnan(roh):
                roh=0
            else:
                print(roh)    
            out[curr_f, out_y, out_x] = roh #np.sum(filt[curr_f] * baserect) #+ bias[curr_f]
            curr_x += s
            out_x += 1
        curr_y += s
        out_y += 1
    curr_x += s
    # out_xb += 1         
# curr_yb += s
# # out_yb += 1
    curr_f+= 1

    return out


def convolution(image, filt, bias, s=1):
    '''
    Confolves `filt` over `image` using stride `s`
    '''
    (n_f, n_c_f, f, _) = filt.shape # filter dimensions
    n_c, in_dim, _ = image.shape # image dimensions
    
    out_dim = int((in_dim - f)/s)+1 # calculate output dimensions
    
#     assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"
    
    out = np.zeros((n_f,out_dim,out_dim))
    
     # convolve the filter over every part of the image, adding the bias at each step. 
    for curr_f in range(n_f):
        curr_y = out_y = 0
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            while curr_x + f <= in_dim:
                out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+f, curr_x:curr_x+f]) #+ bias[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        
    return out


image = plt.imread('C:\\Github\\PNN\\Data\\Images\lenna.png')

tau1,pv = ss.spearmanr([-0.41558442,-0.41558442,1442],[-0.41558442,-0.41558442,1442])
tau2,pv = ss.spearmanr([-0.41558442,-0.41558442,-0.41558442],[-0.41558442,-0.41558442,1442])
tau3,pv = ss.spearmanr([-0.41558442,1,1442],[-0.41558442,2,1442])


tau4 = Spearman([-0.41558442,-0.41558442,1442],[-0.41558442,-0.41558442,1442])
tau5 = Spearman([1,1,1],[1,1,2])
tau6 = Spearman([1,1,2],[1,1,1])

x = np.linspace(-10, 9, 10)

y =SSS(x,4,5)

plt.plot(x, y, 'b')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('ss Function')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.show()

# training data
m =100
img_dim = 28
X = extract_data('C:\\Users\\ayman\\Downloads\\Numpy-CNN-master\\Numpy-CNN-master\\train-images-idx3-ubyte.gz', m, img_dim)
y_dash = extract_labels('C:\\Users\\ayman\\Downloads\\Numpy-CNN-master\\Numpy-CNN-master\\train-labels-idx1-ubyte.gz', m).reshape(m,1)
X-= int(np.mean(X))
X/= int(np.std(X))
train_data = np.hstack((X,y_dash))
batch_size=2
dim = 28
batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]
t = tqdm(batches) 
for batch in (t):
    X = batch[:,0:-1] # get batch inputs
    X = X.reshape(len(batch), 1, dim, dim)
    filt=np.random.normal(loc = 0, scale = 0.25, size = (1,1,3,3))
    a=X[0]
    out1=RankingConvolution(a, s=1)
print(out1)
imgplot1 =plt.imshow(out1[0,:,:])
plt.show()
imgplot2 =plt.imshow(out1[1,:,:]) 
plt.show()
imgplot3 =plt.imshow(out1[2,:,:]) 
plt.show()
imgplot4 =plt.imshow(out1[3,:,:])  
plt.show()