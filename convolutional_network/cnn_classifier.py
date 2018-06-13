import pandas as pd
import numpy as np
import random as ran
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split as ttspilt


class cnn_classifier:

    def __init__(self, n_classes = 5, available_memory_gb=2, learning_rate=0.01, batch_size=32, n_iter=50):

        # Parameters
        self._batch_size = batch_size

        # Network Parameters
        self._n_input = 40
        self._n_classes = n_classes
        self._dropout = 0.5
        self._learning_rate = learning_rate

        self._x = tf.placeholder(dtype=np.float32, shape=(None, self._n_input), name='Batch_Input_Tensor')
        self._y = tf.placeholder(dtype=np.float32, shape=(None, self._n_classes), name='Batch_Prediction')
        self._keep_prob = tf.placeholder(dtype=np.float32, name='Dropout') #dropout (keep probability)

        self._weights = {
            # 1x3 conv, 1 input, 6 outputs
            #[filter_width, in_channels, out_channels],
            #40 - 2  = 38 / 2 = 19
            'wc1': tf.Variable(tf.random_normal([3, 1, 6])), 
            
            #19 - 3 = 16 / 2 = 8
            'wc2': tf.Variable(tf.random_normal([4, 6, 12])), 
            
            #8 - 2 = 6
            'wc3': tf.Variable(tf.random_normal([3, 12, 24])), 
        
            
            # fully connected, 6*24 inputs, 72 outputs
            'wd1': tf.Variable(tf.random_normal([144, 100])), 

            # 256 inputs, 8 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([100, n_classes])) 

        }

        self._biases = {
            'bc1': tf.Variable(tf.random_normal([6])), 
            'bc2': tf.Variable(tf.random_normal([12])),
            'bc3': tf.Variable(tf.random_normal([24])),
            'bd1': tf.Variable(tf.random_normal([100])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        print('\n------- Parameters intialised. Check your model -------\n')

        self._build()

    # Create model
    def _conv1d(self,img, w, b, name):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv1d(img, w,stride=1, 
                                                    padding='VALID'),b),name=name)

    def _max_pool(self,img, k, name, width, out_channel):
        #strides = [batch, height, width, channels].
        #value: A 4-D `Tensor` with shape `[batch, height, width, channels]` and type `tf.float32`.
        
        #Adding additional dimention for max pooling
        img = tf.reshape(img, shape=[-1,1,width, out_channel])
        
        return tf.squeeze(tf.nn.avg_pool(img, ksize=[1, 1, 2, 1], strides=[1, 1, 2,1], padding='SAME',name=name),[1])

    def _conv_net(self,_X, _weights, _biases, _dropout , test=False, printme=False):
        
        #[batch, in_width, in_channels]
        _X = tf.reshape(_X, shape=(-1,self._n_input,1),name='Input')
        if printme == True:
            print(_X)
        
        # Convolution Layer
        conv1 = self._conv1d(_X, _weights['wc1'], _biases['bc1'],name='C1')
        if printme == True:    
            print(conv1)
        # Max Pooling (down-sampling)
        conv1 = self._max_pool(conv1, k=2,name='M2', width=conv1.shape.as_list()[1], out_channel=_weights['wc1'].get_shape().as_list()[2])
        if printme == True:
            print(conv1)
        

        # Convolution Layer
        conv2 = self._conv1d(conv1, _weights['wc2'], _biases['bc2'], name='C3')
        if printme == True:
            print(conv2)
        # Max Pooling (down-sampling)
        conv2 = self._max_pool(conv2, k=2,name='M4', width=conv2.shape.as_list()[1], out_channel=_weights['wc2'].get_shape().as_list()[2])
        if printme == True:
            print(conv2)
        

        # Convolution Layer
        conv3 = self._conv1d(conv2, _weights['wc3'], _biases['bc3'], name='C5')
        if printme == True:
            print(conv3)
        # Max Pooling (down-sampling)
        #conv3 = max_pool(conv3, k=2,name='M6', width=conv3.shape.as_list()[1], out_channel=_weights['wc3'].get_shape().as_list()[2])
        #print(conv3)
        
        # Fully connected layer
        # Reshape conv2 output to fit dense layer input
        dense1 = tf.reshape(conv3, [-1, _weights['wd1'].get_shape().as_list()[0]]) 
        # Relu activation
        dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']),name='FC10')
        if printme == True:
            print(dense1)
        # Apply Dropout
        if test == False:
            dense1 = tf.nn.dropout(dense1, _dropout) # Apply Dropout

        # Output, class prediction
        out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'], name='Out')
        if printme == True:
            print(out)
        return out

    def _build(self):
        print('\n ----------- Architecture of CNN ----------- \n')
        self._pred = self._conv_net(self._x, self._weights, self._biases, self._dropout,test=False,printme=True)
        self._pred_test = self._conv_net(self._x, self._weights, self._biases, self._dropout, test=True)
        self._probability = 1+tf.nn.elu(features=self._pred_test,name='elu')


        # Define loss and optimizer
        self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._pred, labels=self._y))
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self._cost)

        # Evaluate model
        self._correct_pred = tf.equal(tf.argmax(self._pred_test,1), tf.argmax(self._y,1))
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_pred, np.float32))
        self._conf_mat= tf.confusion_matrix(tf.argmax(self._pred,1),tf.argmax(self._y,1))


        self._init = tf.global_variables_initializer()
        self._saver = tf.train.Saver()

    def train_testing_data(self,training_data_path,class_labels,test_size = 0.2,titles=[]):
        cols = list([str(i) for i in range(392)])
        cols.append('label')
        
        all_data = pd.read_csv(training_data_path,names=cols)
        self._class_labels = class_labels
        dataset = all_data.loc[all_data['label'].isin(class_labels)]
        data_array = np.asarray(dataset)
        self._data_set_array, self._test_data_set_array= ttspilt(data_array,test_size=test_size, random_state=10)
        self._short_classes = titles

    #preparing testing dataset

    def _randomize_test_data(self,reduced_bands=False,selected=[]):
        
        cols = list([str(i) for i in range(392)])
        cols.append('label')

        data = self._test_data_set_array
        class_num = self._n_classes

        np.random.shuffle(data)
        test_labels = np.asarray(pd.DataFrame(data,columns=cols)['label'])
        test_one_hot_vectors = np.zeros(shape=(len(data),class_num),dtype=np.float32)
        test_instances = np.asarray(pd.DataFrame(data,columns=cols).drop(labels=['label'],axis=1),dtype=np.float32)

        for count in range(len(test_labels)):
            test_one_hot_vectors[count][ self._class_labels.index(test_labels[count]) ] = 1.0
            
        if reduced_bands == True:
            test_instances = test_instances[:,selected]


        return test_instances, test_one_hot_vectors


    def _train_validation_split(self,train=0.7,reduced_bands = False,selected=[]):
        """ K fold cross validation data function """
        
        cols = list([str(i) for i in range(392)])
        cols.append('label')
        
        data_set_array = self._data_set_array
        np.random.shuffle(data_set_array)

        #training and validation_set
        training_data_set = data_set_array[ 0 : int(train*len(data_set_array)) ]
        validation_data_set = data_set_array[ int(train*len(data_set_array)) : len(data_set_array)  ]

        # validation inputs
        validation_labels =  np.asarray(pd.DataFrame(validation_data_set,columns=cols)['label'])
        validation_one_hot_vectors = np.zeros(shape=(len(validation_data_set),self._n_classes),dtype=np.float32)
        validation_instances = np.asarray(pd.DataFrame(validation_data_set,columns=cols).drop(labels=['label'],axis=1),dtype=np.float32)

        # training inputs
        train_labels =  np.asarray(pd.DataFrame(training_data_set,columns=cols)['label'])
        train_one_hot_vectors = np.zeros(shape=(len(training_data_set),self._n_classes),dtype=np.float32)
        train_instances = np.asarray(pd.DataFrame(training_data_set,columns=cols).drop(labels=['label'],axis=1),dtype=np.float32)

        #generating one hot vectors
        for count in range(len(validation_labels)):
            validation_one_hot_vectors[count][ self._class_labels.index( validation_labels[count] ) ] = 1.0

        for count in range(len(train_labels)):
            train_one_hot_vectors[count][ self._class_labels.index( train_labels[count] ) ] = 1.0


        if reduced_bands == True:
            train_instances = train_instances[:,selected]
            validation_instances = validation_instances[:,selected] 


        return train_instances, train_one_hot_vectors, validation_instances, validation_one_hot_vectors


    def train_model(self,best_model_path,iterations = 10,lr = 0.01,reduced_bands=False,selected=[],log=False):
        
        list_avg_training_accuracy = []
        list_avg_validation_accuracy = []
            
        for iters in range(iterations):
        

            highest_validation_accuracy = 0.0


            with tf.Session() as sess:
                sess.run(self._init)

                # Training cycle
                kfold_training_accuracy = 0.0
                kfold_validation_accuracy = 0.0

                old_validation_accuracy = 0.0

                # Loop over all batches 10 times
                for count in range(10):
                    avg_cost = 0.

                    new_train_instances, new_train_one_hot_vectors, new_validation_instances, new_validation_one_hot_vectors = self._train_validation_split(train=0.7,reduced_bands=True,selected=selected)
                    
                    
                    total_batch = int(len(new_train_instances)/self._batch_size)


                    for i in range(total_batch):
                        batch_xs = new_train_instances[i*self._batch_size:(i+1)*self._batch_size]
                        batch_ys = new_train_one_hot_vectors[i*self._batch_size:(i+1)*self._batch_size]
                        # Fit training using batch data
                        sess.run(self._optimizer, feed_dict={self._x: batch_xs, self._y: batch_ys})


                    # Calculate batch accuracy
                    acc = sess.run(self._accuracy, feed_dict={self._x: batch_xs, self._y: batch_ys})

                    # Calculate batch loss
                    loss = sess.run(self._cost, feed_dict={self._x: batch_xs, self._y: batch_ys})


                    # Calculate validation accuracy
                    vec = sess.run(self._accuracy, feed_dict={self._x: new_validation_instances, 
                                                        self._y: new_validation_one_hot_vectors })
                    
                    kfold_training_accuracy += acc
                    kfold_validation_accuracy += vec
                    

                    if log == True:
                        print ("Epoch:", '%04d' % (count+1) ,", Minibatch Loss= " + \
                            "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

                        print ('Epoch :','%04d' %(count+1)," Validation Accuracy:", vec,'\n')

                    #if np.abs(old_validation_accuracy-vec) < 0.000000001:
                    #    break
                    #else:
                    #    old_validation_accuracy = vec
                    old_validation_accuracy = vec
                        
                avg_traing_acc = kfold_training_accuracy/10
                avg_validation_acc = kfold_validation_accuracy/10
                
                
                if old_validation_accuracy >= highest_validation_accuracy:
                    highest_validation_accuracy = old_validation_accuracy
                    self._saver.save(sess, best_model_path)
                    
                    
                list_avg_training_accuracy.append(avg_traing_acc)
                list_avg_validation_accuracy.append(avg_validation_acc)

                print("Run no. : {} | Training_acc. : {:.4f} | Validation acc. : {:.4f} | Highest val. acc. : {:.4f} ".format(iters+1,avg_traing_acc,avg_validation_acc,highest_validation_accuracy))

        print('\nTraining is completed, best model is saved at: ',best_model_path)
        
        print('\n Overall training acc : {:.4f} | Overall validation acc : {:.4f}'.format(np.mean(list_avg_training_accuracy),np.mean(list_avg_validation_accuracy)))
        
        print('\n\n-------------------------------------------------------\n\n')                
        

    def _kappa_evaluation(self,confusion_mat):

        N = np.sum(confusion_mat)
        nrows = confusion_mat.shape[0]
        nominator = 0
        denominator = 0
        for row in range(nrows):
            x = np.sum(confusion_mat[row])*np.sum(confusion_mat[:,row])
            nominator += N * confusion_mat[row][row] - x
            denominator += x

        return nominator / (N*N - denominator)

    def training_validation(self,path,reduced_bands=False, selected=[]):
        """ Constructs confusion matrix by trained model for training sample"""

        short_classes = self._short_classes

        try:
            with tf.Session() as sess:
                self._saver.restore(sess,path)

                new_train_instances, new_train_one_hot_vectors, _, _ = self._train_validation_split(train=0.7,reduced_bands=reduced_bands,selected=selected)


                training_accuracy = sess.run(self._accuracy, feed_dict={self._x: new_train_instances, self._y: new_train_one_hot_vectors })

                confusion_matrix = sess.run(self._conf_mat, feed_dict={self._x: new_train_instances, self._y: new_train_one_hot_vectors })

                total_sample = [np.sum(confusion_matrix,axis=0)]


                print('\n----------Training site validation of model {}----------\n'.format(path.split('/')[-1]))

                print('Total training samples\n')
                #print(total_sample)
                print(pd.DataFrame(total_sample,columns=short_classes,index=['Total samples']))

                print("\n1. Overall accuracy : {:.4f}\n".format(training_accuracy))

                print('2. Confusion matrix: columns are prediction labels and the rows are the GT data\n')

                print(pd.DataFrame(confusion_matrix,columns=short_classes,index=short_classes))

                class_wise_acc = list()

                for i in range(len(confusion_matrix)):
                    producer_acc = np.round(confusion_matrix[i][i] / np.sum(confusion_matrix[:,i]),decimals=3)
                    consumer_acc = np.round(confusion_matrix[i][i] / np.sum(confusion_matrix[i]),decimals=3)

                    class_wise_acc.append([producer_acc,consumer_acc])

                print('\n3. Producer accuracy and Consumer accuracy:\n')

                print(pd.DataFrame(class_wise_acc,columns=(['Producer/Class acc','Consumer acc']),index=short_classes))

                print('\n4. Average accuracy : {:.4f}'.format(np.sum(np.array(class_wise_acc),axis=0)[0]/len(class_wise_acc)))
                try:
                    print('\n5. Kapp cofficient : {:.4f}\n'.format( self._kappa_evaluation(confusion_matrix)))
                except:
                    print('\n5. Kapp cofficient : undefined')
        except:
           print('Error : Please close any existing Tensorflow session and try again or restart Python Kernel')

    def blindsite_validation(self,path,reduced_bands=False, selected=[]):
        """ Constructs confusion matrix by trained model for blind sample"""

        short_classes = self._short_classes


        try:
            with tf.Session() as sess:
                self._saver.restore(sess,path)

                new_test_instances, new_test_one_hot_vectors = self._randomize_test_data(reduced_bands=reduced_bands,selected=selected)


                testing_accuracy = sess.run(self._accuracy, feed_dict={self._x: new_test_instances, self._y: new_test_one_hot_vectors })

                confusion_matrix = sess.run(self._conf_mat, feed_dict={self._x: new_test_instances, self._y: new_test_one_hot_vectors })

                total_sample = [np.sum(confusion_matrix,axis=0)]


                print('\n----------Blind site validation of model {}----------\n'.format(path.split('/')[-1]))

                print('Total blind site samples\n')
                print(pd.DataFrame(total_sample,columns=short_classes,index=['Total samples']))

                print("\n1. Overall accuracy : {:.4f}\n".format(testing_accuracy))

                print('2. Confusion matrix: columns are prediction labels and the rows are the GT data\n')

                print(pd.DataFrame(confusion_matrix,columns=short_classes,index=short_classes))

                class_wise_acc = list()

                for i in range(len(confusion_matrix)):
                    producer_acc = np.round(confusion_matrix[i][i] / np.sum(confusion_matrix[:,i]),decimals=3)
                    consumer_acc = np.round(confusion_matrix[i][i] / np.sum(confusion_matrix[i]),decimals=3)

                    class_wise_acc.append([producer_acc,consumer_acc])

                print('\n3. Producer accuracy and Consumer accuracy:\n')

                print(pd.DataFrame(class_wise_acc,columns=(['Producer/Class acc','Consumer acc']),index=short_classes))

                print('\n4. Average accuracy : {:.4f}'.format(np.sum(np.array(class_wise_acc),axis=0)[0]/len(class_wise_acc)))
                try:
                    print('\n5. Kapp cofficient : {:.4f}\n'.format( self._kappa_evaluation(confusion_matrix)))
                except:
                    print('\n5. Kapp cofficient : undefined')
        except:
           print('Error : Please close any existing Tensorflow session and try again or restart Python Kernel')

    def _combined_prediction(self,list_of_models,array_of_input,class_labels):

        list_of_model_predctions = list()

        combined_predictions = list()

        for index,model in enumerate(list_of_models):

            print(index+1, "\nModel Running ...\n")
            
            with tf.Session() as sess:
                self._saver.restore(sess,model)
                
                class_probabilities = sess.run(self._probability , feed_dict={self._x:array_of_input})

                list_of_model_predctions.append(class_probabilities)

        list_of_model_predctions = np.array(list_of_model_predctions)

        for i in range(len(array_of_input)):
            combined_predictions.append(class_labels[np.argmax(list_of_model_predctions[:,i])])

    def test_combined_acc(self,data_path,list_of_models,class_labels,titles):

        cols = list([str(i) for i in range(392)])
        
        cols.append('label')
        
        data = pd.read_csv(data_path,names=cols)

        data_array = np.array(data.drop(['label'],axis=1))
        
        y_actu = pd.Series(data.label,name='Actual')

        combined_preds = self._combined_prediction(list_of_models, data_array,class_labels)
        
        y_pred = pd.Series(combined_preds, name='Predicted')
        
        df_confusion = pd.crosstab(y_actu, y_pred)  

        print(df_confusion)






