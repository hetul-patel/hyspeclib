#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:27:46 2018

@author: hetul
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spectral import open_image
from sklearn.model_selection import train_test_split as ttspilt
import os


class make_model:
    
    def __init__(self, n_nodes=[392,150,100,50,11],learning_rate=0.01,batch_size=32,n_iter=50):
        
        # Network Parameters
        self._n_layer = len(n_nodes)-1
        
        if self._n_layer < 2 :
            print('Error : Minimum two layers are required for e.g input and output nodes.')
        
        self._n_nodes = n_nodes
        self._learning_rate = learning_rate
        self._dropout=0.1
        self._n_input = n_nodes[0] # Number of Inputs here number of bands
        self._n_classes = n_nodes[-1] # Classes in training Set
        self._batch_size = batch_size
        self._num_iteration = n_iter

        # tf Graph input
        self._x = tf.placeholder("float", [None, self._n_input],name='inputs')
        self._y = tf.placeholder("float", [None, self._n_classes],name='outputs')
        #learning_rate = tf.Variable(initial_value=0.01,name='learning_rate')
        
        self._weights = list()
        self._biases = list()
        
        for i in range(self._n_layer):
            self._weights.append(tf.Variable(tf.random_normal([self._n_nodes[i], self._n_nodes[i+1]]),name='L'+str(i)))
            self._biases.append(tf.Variable(tf.random_normal([self._n_nodes[i+1]]),name='B'+str(i)))
        # Store layers weight & bias
        
        print('Parameters intialised')
        
        self._build_model()



    def train_testing_data(self,dataset_path,titles,test_size=0.10):
        cols = list([str(i) for i in range(self._n_input)])
        cols.append('label')

        self.dataset = pd.read_csv(dataset_path,names=cols)
        self.data_array = np.asarray(self.dataset)
        self.data_set_array, self.test_data_set_array= ttspilt(self.data_array,test_size=test_size, random_state=10)
        self.short_classes = titles
        self.n_classes = self._n_classes


    def train_validation_split(self,train=0.7,reduced_bands=False,not_selected=[]):
        """ K fold cross validation data function """

        np.random.shuffle(self.data_set_array)

        #training and validation_set
        training_data_set = self.data_set_array[ 0 : int(train*len(self.data_set_array)) ]
        validation_data_set = self.data_set_array[ int(train*len(self.data_set_array)) : len(self.data_set_array)  ]

        # validation inputs
        validation_labels =  np.asarray(pd.DataFrame(validation_data_set)[self._n_input])
        validation_one_hot_vectors = np.zeros(shape=(len(validation_data_set),self.n_classes),dtype=np.float32)
        validation_instances = np.asarray(pd.DataFrame(validation_data_set).drop(labels=[self._n_input],axis=1),dtype=np.float32)

        # training inputs
        train_labels =  np.asarray(pd.DataFrame(training_data_set)[self._n_input])
        train_one_hot_vectors = np.zeros(shape=(len(training_data_set),self.n_classes),dtype=np.float32)
        train_instances = np.asarray(pd.DataFrame(training_data_set).drop(labels=[self._n_input],axis=1),dtype=np.float32)

        #generating one hot vectors
        for count in range(len(validation_labels)):
            validation_one_hot_vectors[count][ int(validation_labels[count]) - 1] = 1.0

        for count in range(len(train_labels)):
            train_one_hot_vectors[count][ int(train_labels[count]) - 1] = 1.0


        if reduced_bands == True:
            train_instances[:,not_selected] = 0.
            validation_instances[:,not_selected] = 0.


        return train_instances, train_one_hot_vectors, validation_instances, validation_one_hot_vectors

    def randomize_test_data(self,all_class=False,reduced_bands=False,not_selected=[]):

        data = self.test_data_set_array
        class_num = self.n_classes

        np.random.shuffle(data)
        test_labels = np.asarray(pd.DataFrame(data)[self._n_input])
        test_one_hot_vectors = np.zeros(shape=(len(data),class_num),dtype=np.float32)
        test_instances = np.asarray(pd.DataFrame(data).drop(labels=[self._n_input],axis=1),dtype=np.float32)

        for count in range(len(test_labels)):
            test_one_hot_vectors[count][ int(test_labels[count]) - 1] = 1.0

        if reduced_bands == True:
            test_instances[:,not_selected] = 0.


        return test_instances, test_one_hot_vectors

    # Create model
    def multilayer_perceptron(self,_X, _weights, _biases, _dropout):
        print(_X)
        
        layers = list()        
        
        layers.append(tf.nn.relu(tf.add(tf.matmul(_X, _weights[0]), _biases[0]),name='hidden_1'))
        
        for i in range(1, self._n_layer-1):
            layers.append(tf.nn.relu(tf.add(tf.matmul(layers[i-1], _weights[i]), _biases[i]),name='hidden_'+str(i+1)))
            
        layers.append(tf.add(tf.matmul(layers[-1], _weights[-1]) , _biases[-1],name='out_layer'))
        
        for layer in layers:
            print(layer)
            
        return layers[-1]

    def _build_model(self):

        # Construct model
        self.pred = self.multilayer_perceptron(self._x, self._weights, self._biases, self._dropout)
        self.pred_prob = tf.nn.softmax(self.pred)
        # Define loss and optimizer

        # Softmax loss
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self._y))
        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self.cost)

         # Evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self._y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, np.float32))
        self.conf_mat= tf.confusion_matrix(tf.argmax(self.pred,1),tf.argmax(self._y,1))

        #use model for classifying image
        threshold_classification = lambda v : tf.argmax(v,1) if (np.array(v)[tf.argmax(v,1)] > 0.5) else self.n_classes
        
        self.classified = threshold_classification(self.pred_prob)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()



    def train_model(self,iterations = 10,lr = 0.01,reduced_bands=False,not_selected=[],early_stopping=False,best_model_path="best_model/1500_model.ckpt"):


        avg_train = 0
        avg_validation = 0
        cost_list = []
        overall_old = 0.


        for iteration in range(iterations):


            with tf.Session() as sess:
                sess.run(self.init)

                #9 Fold Validation run 9 times, 8/9 training and 1/9 testing

                cnt = 0
                flag = True
                old_val_acc = 0.

                for i in range(9):

                    #Getting new Instances
                    new_train_instances, new_train_one_hot_vectors, new_validation_instances, new_validation_one_hot_vectors = self.train_validation_split(train=0.7,reduced_bands=reduced_bands,not_selected=not_selected)

                    total_batch = len(new_train_instances) // self._batch_size

                    #print(i+1,' total batch : ',total_batch)


                    for batch in range(total_batch):

                        #Preparing batches
                        batch_xs = new_train_instances[batch*self._batch_size:(batch+1)*self._batch_size]
                        batch_ys = new_train_one_hot_vectors[batch*self._batch_size:(batch+1)*self._batch_size]

                        # Fit training using batch data
                        sess.run(self.optimizer, feed_dict={self._x: batch_xs, self._y: batch_ys})

                        if batch%(self._batch_size-1) == 0:


                            loss = sess.run(self.cost,feed_dict={self._x: batch_xs, self._y: batch_ys})

                            cost_list.append(loss)

                            if early_stopping == True:

                                validation_acc = sess.run(self.accuracy,feed_dict={self._x: new_validation_instances, self._y: new_validation_one_hot_vectors})

                                increase = validation_acc - old_val_acc
                                old_val_acc = validation_acc

                                #print("Step: {} | Fold : {} | Batch : {}  |  Training acc: {:.4f} | Validation acc: {:.4f} | Increase : {:.4f}".format(i*total_batch + batch,i+1,batch+1, mini_batch_acc, validation_acc,increase))

                                if increase <= 0.0 and cnt < 4:
                                    cnt += 1
                                elif increase <= 0.0 and cnt >= 4:
                                    flag = False
                                    break
                                else:
                                    cnt = 0

                    if early_stopping == True and flag == False:
                        break

                new_test_instances, new_test_one_hot_vectors = self.randomize_test_data(reduced_bands=reduced_bands,not_selected=not_selected)
                new_train_instances, new_train_one_hot_vectors, new_validation_instances, new_validation_one_hot_vectors = self.train_validation_split(train=0.7,reduced_bands=reduced_bands,not_selected=not_selected)


                training_accuracy = sess.run(self.accuracy, feed_dict={self._x: new_train_instances,
                                                                  self._y: new_train_one_hot_vectors })

                validation_accuracy = sess.run(self.accuracy, feed_dict={self._x: new_validation_instances,
                                                                    self._y: new_validation_one_hot_vectors })


                #testing_accuracy = sess.run(self.accuracy, feed_dict={self._x: new_test_instances,
                #                                                 self._y: new_test_one_hot_vectors })



                avg_train += training_accuracy
                avg_validation += validation_accuracy


                #print("step : {} | Training_acc: {:.4f} | Validation acc: {:.4f} | Old Vec: {:.4f} | Testing acc: {:.4f}".format(iteration+1,training_accuracy,validation_accuracy,overall_old,testing_accuracy))
                print("step : {} | Training_acc. : {:.4f} | Validation acc. : {:.4f} | Highest val. acc. : {:.4f} ".format(iteration+1,training_accuracy,validation_accuracy,overall_old))

                if validation_accuracy > overall_old:
                    if reduced_bands==False:
                        self.saver.save(sess, best_model_path)
                        print('Saving....')
                    else:
                        print('Saving....')
                        self.saver.save(sess, best_model_path)

                    overall_old = validation_accuracy


        return {'Overall training acc : ' : avg_train/iterations,'Overall validation acc : ' : avg_validation/iterations}
    
    def kappa_evaluation(self,confusion_mat):
        
        N = np.sum(confusion_mat)
        nrows = confusion_mat.shape[0]
        nominator = 0
        denominator = 0
        for row in range(nrows):
            x = np.sum(confusion_mat[row])*np.sum(confusion_mat[:,row])
            nominator += N * confusion_mat[row][row] - x
            denominator += x
            
        return nominator / (N*N - denominator)
        

    def training_validation(self,path,reduced_bands=False,not_selected=[]):
        """ Constructs confusion matrix by trained model for training sample"""

        short_classes = self.short_classes

        #try: 
        with tf.Session() as sess:
            self.saver.restore(sess,path)

            new_train_instances, new_train_one_hot_vectors, _, _ = self.train_validation_split(train=0.7,reduced_bands=reduced_bands,not_selected=not_selected)


            training_accuracy = sess.run(self.accuracy, feed_dict={self._x: new_train_instances, self._y: new_train_one_hot_vectors })

            confusion_matrix = sess.run(self.conf_mat, feed_dict={self._x: new_train_instances, self._y: new_train_one_hot_vectors })
            
            total_sample = [np.sum(confusion_matrix,axis=0)]


            print('\n----------Training site validation of model {}----------\n'.format(path.split('/')[-1]))
            
            print('Total training samples\n')
            #print(total_sample)
            print(pd.DataFrame(total_sample,columns=short_classes))

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
                print('\n5. Kapp cofficient : {:.4f}'.format( self.kappa_evaluation(confusion_matrix)))
            except:
                print('\n5. Kapp cofficient : undefined')
        #except:
           #print('Please close any existing Tensorflow session and try again or restart Python Kernel')
           
    def blindsite_validation(self,path,reduced_bands=False,not_selected=[]):
        """ Constructs confusion matrix by trained model for blind sample"""

        short_classes = self.short_classes

        try: 
            with tf.Session() as sess:
                self.saver.restore(sess,path)
    
                new_test_instances, new_test_one_hot_vectors = self.randomize_test_data(reduced_bands=reduced_bands,not_selected=not_selected)
    
    
                testing_accuracy = sess.run(self.accuracy, feed_dict={self._x: new_test_instances, self._y: new_test_one_hot_vectors })
    
                confusion_matrix = sess.run(self.conf_mat, feed_dict={self._x: new_test_instances, self._y: new_test_one_hot_vectors })
    
                total_sample = [np.sum(confusion_matrix,axis=0)]
    
    
                print('\n----------Blind site validation of model {}----------\n'.format(path.split('/')[-1]))
                
                print('Total blind site samples\n')
                print(pd.DataFrame(total_sample,columns=short_classes))
    
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
                    print('\n5. Kapp cofficient : {:.4f}'.format( self.kappa_evaluation(confusion_matrix)))
                except:
                    print('\n5. Kapp cofficient : undefined')
        except:
           print('Please close any existing Tensorflow session and try again or restart Python Kernel')

    
    def make_a_list(self,limit):
        list_of_numbers = list()

        for i in range(limit):
            list_of_numbers.append(i)

        return list_of_numbers



    def reduction_function(self,array_of_weights, in_array, out_array,red_per=0.30,printing=True):
        """This function returns fraction of neurons only, reduced from original input neurons
            Sum of the weight approach"""


        dtype_of_dict = [('input_neuron',int),('aggregate',float)]

        index_weight_map = [ [] for i in range(len(in_array))]
        #reduced_index_weight_map = [ [] for i in range(len(in_array))]
        #reduced_indexes = [ [] for i in range(len(in_array))]
        sum_of_weights  = []

        for i in in_array:
            count = 0
            for j in out_array:
                index_weight_map[i].append((i,j,abs(array_of_weights[i][j])))
                count += abs(array_of_weights[i][j])
            sum_of_weights.append((i,count))

        array = np.array(sum_of_weights,dtype=dtype_of_dict)
        sorted_aggregate = np.sort(array,order='aggregate')

        if printing == True:

            #Printing the aggregate score
            df = pd.DataFrame(sorted_aggregate)
            print(df['aggregate'].describe())

            df['aggregate'].plot()
            plt.grid()
            plt.legend('Score',loc=2)
            plt.title('Band wise Aggegate Scores')

            plt.show()

        top_bands = list()

        for i in range(int((1-red_per)*len(in_array)),len(in_array)):
            top_bands.append(sorted_aggregate[i][0])


        return list(np.sort(top_bands))


    def select_best_bands(self,model_path,reduction_percentage = 0.21):
        """ Returns selected, not selected bands """
        
        weights_saved = None
        with tf.Session() as sess:
            self.saver.restore(sess,model_path)
            weights_saved = sess.run(self._weights)

        red_per_inter = 0.3
        
        for i in range(len(weights_saved)-1,0,-1):
            temp_weights = weights_saved[i]
            temp_bands = self.reduction_function(temp_weights, self.make_a_list(temp_weights.shape[0]),self.make_a_list(temp_weights.shape[1]),printing=False,red_per=red_per_inter)


        h1 = weights_saved[0]
        selected = self.reduction_function(h1, self.make_a_list(h1.shape[0]),temp_bands,red_per=reduction_percentage,printing=False)

        not_selected = list(set(np.arange(self._n_input)) - set(selected))

        return selected, not_selected

    def validation(self,instance, mean, t = 0.05):
        
        if np.mean(instance) == 0:
            return False
    

        distance = np.sqrt(np.sum(np.square(np.subtract(instance,mean))))

        if distance <= t:
            return True
        else:
            return False

    def save_data_to_file(self,file,pixel,class_index):
        """Save a newly identified pixel to file"""
        
        for i in range(self._n_input):
            
            file.write(str(pixel[i])+',')
        
        file.write(str(class_index+1)+'\n')
        
    def _augmentation_mask(self, file, class_index, img_cols):
        
        if self._tmp_count % (img_cols-1) or self._tmp_count == 0:
            file.write(str(class_index+1)+',')
            self._tmp_count += 1
        else:
            file.write(str(class_index+1)+'\n')
            self._tmp_count = 0


	#def train_augmentation(self,image_path, model_path, correct_array, new_trainig_data_set, threshold = 0.05, non_selected = [], rows_at_time = 80):
    def train_augmentation(self,image_path, model_path, correct_array, rows_at_time = 80, threshold=0.05, not_selected = [], save_data=False,save_directory=''):
        
        img = open_image(image_path)
        
        image_name = image_path.split('/')[-1].split('.')[0]
        
        self._tmp_count = 0
        
        img_cols = img.ncols
        last_block_row_count = img.nrows%rows_at_time
        total_blocks = int(np.ceil(img.nrows/rows_at_time))
        
        if save_data==True:
            out_file = open(save_directory+'augmented_dataset.csv','a')
            augmentation_mask_file = open(save_directory+image_name+'augmentation_mask.csv','w')
        
        print('Image : {} divided into {} blocks..'.format(image_path.split('/')[-1],total_blocks))
        
    
        self.sess = tf.Session()
        
        try:
            self.saver.restore(self.sess,model_path)
            
    
            for block in range(total_blocks):
                
                print('Currently {} / {}'.format(block+1,total_blocks))
                    
                start_row = block*rows_at_time
    
                block_rows_count = rows_at_time
                if block == total_blocks-1:
                    block_rows_count = last_block_row_count
    
                end_row = start_row + block_rows_count
                
                img_block = img[start_row:end_row,:]
                            
                masked_block = np.reshape(np.copy(img_block),newshape=(block_rows_count*img_cols,-1))
                
                img_block = np.reshape(img_block,newshape=(block_rows_count*img_cols,-1))

                masked_block[:,not_selected] = 0.
                
               
                classified_block = self.sess.run(self.classified, feed_dict={self._x: masked_block} )
                
                    
                for index,pixel in enumerate(masked_block):
                    if np.sum(pixel) == 0 :
                        classified_block[index] = self.n_classes
                        
    
                for i in range(len(masked_block)):
                
                    if np.mean(masked_block[i]) > 0 and self.validation(masked_block[i],self.class_wise_mean[classified_block[i]],t=threshold) == True:
                        correct_array[classified_block[i]] +=1
                        
                        if save_data == True:
                            #self.save_data_to_file(out_file,img_block[i],classified_block[i])
                            self._augmentation_mask(augmentation_mask_file,classified_block[i],img_cols)
                    elif save_data == True:
                        self._augmentation_mask(augmentation_mask_file,-1,img_cols)
                        
                
                del img_block
                del masked_block
                
            self.sess.close()
            
            print('Image : {} classified..'.format(image_path.split('/')[-1]))
            
            if save_data == True:
                out_file.close()
                augmentation_mask_file.close()
                
            del img
    
            return correct_array
    
        except:
           self.sess.close()
           print('Error : Try again after restrating kernel')
           return correct_array
        
        
        
    

        
    
    
    def increase_train_data(self,images,model_path,threshold,save_data=False,save_directory='',not_selected=[],rows_at_time=80):
        
        self.class_wise_mean = np.array(self.dataset.groupby(['label']).mean())
        self.class_wise_mean[:,not_selected] = 0.
        
    
        correct_array = [0 for i in range(self.n_classes)]
        
        for image in images:
            correct_array = self.train_augmentation( image,model_path,
                                                     correct_array,rows_at_time,
                                                     threshold,not_selected,
                                                     save_data=save_data,
                                                     save_directory=save_directory)
   
            
        
        return correct_array
    
    def classify_image(self,image_path,model_path,save_path,not_selected=[],rows_at_time = 80):
        
        #level = [1,2,3,4,5,6,7,8,9,10,11,12]
        #color_list = ['#3BCBD5','#F7CD0A','#990033','#FF3399','#339900','#666600',
        #        '#000000','#0000FF','#CC7755','#FF8866','#FF9988']
    
        img = open_image(image_path)
        
        
        img_cols = img.ncols
        last_block_row_count = img.nrows%rows_at_time
        total_blocks = int(np.ceil(img.nrows/rows_at_time))
        
        
        out_file = open(save_path,'w')
        
        print('Image : {} divided into {} blocks..'.format(image_path.split('/')[-1],total_blocks))
        
    
        self.sess = tf.Session()
        
        try:
            self.saver.restore(self.sess,model_path)
            
    
            for block in range(total_blocks):
                
                print('Currently {} / {}'.format(block+1,total_blocks))
                    
                start_row = block*rows_at_time
    
                block_rows_count = rows_at_time
                if block == total_blocks-1:
                    block_rows_count = last_block_row_count
    
                end_row = start_row + block_rows_count
                
                img_block = img[start_row:end_row,:]
                            
                masked_block = np.reshape(np.copy(img_block),newshape=(block_rows_count*img_cols,-1))
                
                img_block = np.reshape(img_block,newshape=(block_rows_count*img_cols,-1))

                masked_block[:,not_selected] = 0.
                
               
                classified_block = self.sess.run(self.classified, feed_dict={self._x: masked_block} )
                
                    
                for index,pixel in enumerate(masked_block):
                    if np.sum(pixel) == 0 :
                        classified_block[index] = self.n_classes
                        
                for row in range(block_rows_count):
                    for column in range(img_cols-1):
                        out_file.write(str(classified_block[column + row*img_cols]+1)+',')
                    out_file.write(str(classified_block[column+1 + row*img_cols]+1)+'\n')
                            
    
        
                del img_block
                del masked_block
                
            self.sess.close()
        except:
            print('Error in tensorflow: close any existing sessions')
            self.sess.close()
            return
        
        del img
        
        print('Image : {} classified and saved to :'.format(image_path.split('/')[-1]),save_path)
        
                 
     
                 
        

