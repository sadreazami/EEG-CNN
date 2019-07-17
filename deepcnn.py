

import matplotlib.pyplot as mtl

import numpy as np
import scipy.io
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  
from sklearn import svm
from sklearn.pipeline import Pipeline  # noqa
from sklearn.model_selection import cross_val_score  # noqa
import scipy
from CSP import CSP2
from mne.decoding import CSP
from spatfilt import csp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from spatfilt import csp
import tensorflow as tf

########################## Fully connected layers information #################################################

FullyConnected_active = True

                                ## number of input layer units (features)
M=[1000,300,200,100,50,10,2]                       ## Hiden layers units for fully_connected layers
             
############################################################################################################

def batch_norm_wrapper(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        train_mean = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
             return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, epsilon)
    else:
             return tf.nn.batch_normalization(inputs,pop_mean, pop_var, beta, scale, epsilon)              

######################### Convolutional layer information #####################################################

convolutional_active = True

################### for conv2D
epsilon = 1e-3
conv_strides = [[1,1,1,1],[1,1,1,1]]
filter_size = [[2,2,1,32],[2,2,32,64]]

################### for pooling

ksize = [[1,2,2,1],[1,2,2,1]]
pool_strides = [[1,2,2,1],[1,2,2,1]]

################################################################################################################



tf.reset_default_graph()

#data=scipy.io.loadmat("C:/Users/y_moh/Desktop/data5.mat")
data = scipy.io.loadmat('d:/data_ready/Filtered_Training_Exp_Wighted_MovingAve_Down.mat')
data2 = scipy.io.loadmat('d:/data_ready/Filtered_Test_Exp_Wighted_MovingAve_Down.mat')

xx=[]
train=data['TR_d1'][:,:,:]
test=data2['TS_d1'][:,:,:]

#train=data['train'][:,:,:]
#test=data['test'][:,:,:]
train=np.transpose(train,[2,0,1])
test=np.transpose(test,[2,0,1])



xx.append(train)
xx.append(test)
#
X_t=np.vstack(xx)

#c=[]
#X_t=X_train+np.random.randn(118,350)
#
#c.append(X_train)
#c.append(X_t)
#
#X_train=np.vstack(c)

ll=[]
label=scipy.io.loadmat("C:/Users/y_moh/Desktop/true_labels_aa.mat")
y_t =label['true_y'][0]-1
#y_test =label['true_y'][0][168::]-1


#ll.append(np.expand_dims(y_train,1))
#ll.append(np.expand_dims(y_train,1))
#l=np.vstack(ll)
#l=np.squeeze(l)

#X_train, _, y_train, _ = train_test_split( X_train, l, test_size=0, random_state=7)


#mmm = np.mean(X_train,2, keepdims = True)
#X_train=X_train-mmm
#
#mmm = np.mean(X_test,2, keepdims = True)
#X_test=X_test-mmm

#for i in range(X_train.shape[0]):
#    mm=np.mean(X_train[i,:,:],1)
#    X=X_train[i,:,:].T-mm
#    X_train[i,:,:]=X.T

#X_test=X_test-mmm

#for i in range(X_test.shape[0]):
#    mm=np.mean(X_test[i,:,:],1)
#    X=X_test[i,:,:].T-mm
#    X_test[i,:,:]=X.T

#mmm = np.mean(X_test,0)
#
#X_test=X_test-mmm


#for i in range(X_test.shape[0]):
#    mm=np.mean(X_test[i,:,:],0)
#    X_test[i,:,:]=X_test[i,:,:]-mm

m=3

kf=KFold(5, random_state=None, shuffle=False)
kf.get_n_splits(X_t)
k=0
for train_index, test_index in kf.split(X_t):
    X_train, X_test = X_t[train_index], X_t[test_index]
    y_train, y_test = y_t[train_index], y_t[test_index]
    
    if k==m:
       break 
    k=k+1


#########################################################################################################

#######################################################################################################
    

task1=[]
task2=[]

for i in range(X_train.shape[0]):
    if y_train[i]==0:
       task1.append(X_train[i,:,:])
    else:
       task2.append(X_train[i,:,:])
            
task1=np.stack(task1,axis=0)
task2=np.stack(task2,axis=0)
###class_covs=[]

####class_covs.append(np.cov(np.hstack(task1)))
####class_covs.append(np.cov(np.hstack(task2)))

###a=csp(class_covs[0],class_covs[1],6)


###XX=np.asarray([np.dot(a, trial) for trial in X_train])
#VV=np.var(XX,2)
#mmm = np.mean(VV,0, keepdims = True)
#VV=VV-mmm
###VV=XX
###YY=np.asarray([np.dot(a, trial) for trial in X_test])
#VV1=np.var(YY,2)
#VV1=VV1-mmm

###VV1=YY

b=[]

a=CSP2(task1,task2)

b1=a[0][0:3,:]
b2=a[0][-3:,:]
#####
b.append(b1)
b.append(b2)
#####
a=np.vstack(b)

#######################################################################################################
#b=[]
#
#a=CSP(task1,task2)
#
#b1=a[0]
#b2=a[1]
######
#b.append(b1)
#b.append(b2)
######
#a=np.vstack(b)
#
XX=np.asarray([np.dot(a, trial) for trial in X_train])
VV=XX
#x=np.var(XX,2)
##n=np.sum(x,1,keepdims=True)
####
#VV=np.log(x)
YY=np.asarray([np.dot(a, trial) for trial in X_test])
VV1=YY
#y=np.var(YY,2)
##N=np.sum(y,1,keepdims=True)
#VV1=np.log(y)

X = tf.placeholder(dtype=tf.float32 , shape=[None,6,350] , name='inpu')
Y = tf.placeholder(dtype=tf.int64 , shape = [None,], name='output')

phase_train = tf.placeholder(tf.bool, name='phase_train')


p = tf.placeholder(dtype=tf.float32, name = 'Drop_out')


###########################################################################################################################
################################## Convolutional layers design  ###########################################################
###########################################################################################################################
Frame= tf.expand_dims(X,3)

if convolutional_active == True :     # if convolutional is used
   #Frame= tf.reshape(X,[-1,35,35,1])  

        
   for index, (ks, Pstrd, Cstrd , fil) in enumerate(zip(ksize,pool_strides,conv_strides,filter_size)):

        with tf.variable_scope ('conv'+str(index)) as scope:
             convW= tf.get_variable('convW', fil, initializer = tf.truncated_normal_initializer(stddev=0.01))
             fcb= tf.get_variable('bias', initializer =tf.constant(0.01,shape=[fil[-1]]))
             #A  = tf.gradient(convW)

             ConV=tf.nn.conv2d(Frame,convW, strides = Cstrd,  padding='SAME', name='conv')
             ConV=batch_norm_wrapper(ConV,True)
             Relu=tf.nn.relu(ConV + fcb , name='relu')
             pool = tf.nn.max_pool(Relu, ksize=ks , strides = Pstrd , padding='SAME', name = 'pooling')
             Frame=pool
             
#             tf.summary.histogram('conv'+str(index)+'/activation', Relu)
#             tf.summary.histogram('conv'+str(index)+'/convW', convW)
#             tf.summary.histogram('conv'+str(index)+'/convb', fcb)
#             
                 
#             for i in range(fil[-1]):
#                 tf.summary.image('conv'+str(index)+"/weights"+str(i), tf.reshape(convW[:,:,:,i],[-1,5,5,1]))
#                    #tf.summary.image('conv'+str(index)+"/activation"+str(i),tf.reshape(Relu[:,:,:,i],[-1,28,28,1]), max_outputs=2 )

                    
                    
                    
###########################################################################################################################
################################### Fully _ connected layers design ##########################################
##############################################################################################################
size = Frame.get_shape().as_list()
N=size[1]*size[2]*size[3]
inpt=tf.reshape(Frame,[-1, N])
#N = size[1]  
#inpt=Frame
if FullyConnected_active == True : 
       
   #inp = tf.reshape(Frame,[100,-1]) 
   #Frame.astype('float32')
   for m in range(len(M)):        
       with tf.variable_scope ('FullyC'+str(m)) as scope:
             fcW= tf.get_variable('fcW', [N,M[m]] , initializer = tf.truncated_normal_initializer(stddev=1))* tf.sqrt(2/N)
             fcB= tf.get_variable('bias', initializer=tf.constant(0.001,shape=[M[m]])) 
             Z=tf.add(tf.matmul(inpt,fcW) , fcB, name= 'add') 
             
             if m < len(M)-1:
               
#                batch_mean1, batch_var1 = tf.nn.moments(Z,[0])
#                Z_hat = (Z - batch_mean1) / tf.sqrt(batch_var1 + 1e-3)
#                scale1 = tf.Variable(tf.ones([M[m]]))
#                beta1 = tf.Variable(tf.zeros([M[m]]))
#                BN1 = scale1 * Z_hat + beta1
#                if phase_train==True:
#                   BN1 = batch_norm_wrapper(Z, True)
#                else:
#                   BN1 = batch_norm_wrapper(Z, False)
                       
                
                inpt= tf.nn.dropout(tf.nn.relu(Z), p)                  
                N=M[m] 
             else:
                h=tf.nn.softmax(Z) 
                
                # p : the probability of keeping
           #  tf.summary.histogram('FullyC'+str(m)+'/activation', tf.nn.relu(Z))
           #  tf.summary.histogram('FullyC'+str(m)+'/fcW', fcW)
           #  tf.summary.histogram('FullyC'+str(m)+'/fcb', fcB)

###################################################################################################################



                                    
###################################################################################################################
vars   = tf.trainable_variables() 
lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'fcb' not in v.name ]) 

loss= tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=Z)) #+ 0.00001* lossL2

#loss= tf.reduce_sum(-Y * tf.log(h + 10**(-5))) / (2*168)

#tf.summary.scalar("entropy", loss)
 
var_list=tf.trainable_variables() 
optimizer=tf.train.AdamOptimizer(0.001).minimize(loss)
#grads = optimizer.compute_gradients(loss, var_list=var_list)
#train_op = optimizer.apply_gradients(grads)


#for grad,var in grads:
#    tf.summary.histogram(var.op.name + "_gradient", grad) 
#    
saver=tf.train.Saver()
init=tf.global_variables_initializer()

correct_p= tf.equal(tf.argmax(h,1), Y)

Acuraccy = tf.reduce_mean(tf.cast(correct_p, tf.float32))
#tf.summary.scalar("Acuraccy", Acuraccy)


#tf.summary.image("input_image", tf.reshape(X,[-1,35,118,1]), max_outputs=5)
#tf.summary.image("pred_annotation", correct_p, max_outputs=2)
#summary_op = tf.summary.merge_all()


#tf.get_default_graph().finalize()
###################################################################################################################
#C=int(5000/Batch_Size)
saver = tf.train.Saver()


with tf.Session() as sess:
     sess.run(init)
     #saver.restore(sess,'c:\deepgraph\-17000')
#     writer = tf.summary.FileWriter('c:/newgraph', sess.graph)
#     losst=0
#     acurracy=0
#     acurracy_train=0
#     
     for ep in range (100000):
#         a=time.time()  
#         lt=0
#         hhh=[]
#         for i in range(1000):
#             Xt= X_train[i*35:(i+1)*35,:,:]
#             yt= y_train[i*35:(i+1)*35]
     #saver.restore(sess, "c:/newgraph/model5.ckpt")

         #for j in range(2000) :
         #b=mnist.train.next_batch(Batch_Size) 
         #for i in range(8):
            #xx = XX[:,i*21:(i+1)*21-1].T
            # ll= l[i*21:(i+1)*21-1,:]
             

         _,L = sess.run([optimizer,loss], feed_dict={ X:VV , Y:y_train , p : 0.8 }) 
##             #hhh.append(hh)
         A = sess.run(Acuraccy, feed_dict={ X:VV , Y: y_train, p : 1 })
             
         At = sess.run(Acuraccy, feed_dict={ X:VV1 , Y: y_test, p : 1 })
         
             
#            
#         
#             if ep%20==0:  #losst+=L
#         summary_t = sess.run(summary_op, feed_dict={ X:XX , Y: l.eval(), p : 1 }) 
#         writer.add_summary(summary_t,ep)

             #b=time.time()-a
         print(ep,"  ",i, "  L=  ", L , "acc = ",A , "Acc = ", At )
    #At = sess.run(Acuraccy, feed_dict={ X:XXt , Y: lt.eval(), p : 1 })
         #if ep%10==0:
         #At = sess.run(Acuraccy, feed_dict={ X:X_train , Y: y_train, p : 1 })

 #########################################################        
#     
#    for i in range(1000):
#        Atest = sess.run(Acuraccy, feed_dict={ X:XX , Y: l], p : 1 })
#        
#         
#        correct_p= np.mean(np.equal(A, l))
#        print(correct_p)
################################################################       
         #print(ep,"  ", "  A =", Atest ) 

#         save_path = saver.save(sess, "c:/newgraph/model5.ckpt") 
             #if i%1000==0:
              #  save_path = saver.save(sess, "c:/deepgraph/", global_step=i)
              #   print("Model saved in file: %s" % save_path)  
       # save_path = saver.save(sess, "c:/deepgraph/", global_step=20000)
     
#with tf.Session() as sess:
#     saver.restore(sess,'c:\deepgraph\-20000')
#     A = sess.run(Acuraccy, feed_dict={ X:mnist.test.images, Y: mnist.test.labels, p : 1 }) 
##
#             
#         AA.append(ac)
     #A1 = sess.run(h, feed_dict={X:mnist.test.images[1:50] , Y: mnist.test.labels[1:50], p : 1})     
     #ac= np.mean(np.argmax(A1[1:500] , 1)== np.argmax( mnist.test.labels[1:500], 1)) 
     ##print( "acc_test = ",ac )  
#     test = tf.get_default_graph().get_tensor_by_name("conv0/relu:0")
#     sess.run(test) 
#     
     

#saver = tf.train.import_meta_graph('c:/graphdeep/model.ckpt.meta')

#graph= tf.get_default_graph()
#
#     test = tf.get_default_graph().get_tensor_by_name("conv0/relu:0")

#
#with tf.Session() as sess:
#     saver.restore(sess,'c:/graphdeep/model.ckpt')
#     print(sess.run(tensor))
     
     
#     def getActivations(layer,stimuli):
#        units = sess.run(layer,feed_dict={X:np.reshape(stimuli,[1,784],order='F')})
#        plotNNFilter(units)  
#        
#     def plotNNFilter(units):
#        filters = units.shape[3]
#        mtl.figure(1, figsize=(20,20))
#        n_columns = 6
#        n_rows = math.ceil(filters / n_columns) + 1
#        for i in range(filters):
#            mtl.subplot(n_rows, n_columns, i+1)
#            mtl.title('Filter ' + str(i))
#            mtl.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
#     imageToUse = mnist.test.images[0]
#     test = tf.get_default_graph().get_tensor_by_name("conv0/pooling:0")
#     getActivations(test,imageToUse)  
#     mtl.figure()
#     mtl.plot(LL)