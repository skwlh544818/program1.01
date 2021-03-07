'''
fun:利用最优参数组合，预测测试集精度
lingpai:1291f328ed6db3cb01e944cf8962c414d4238386
time:2018-7-12
'''
import datetime
import numpy as np
from keras import Model,Input,regularizers
from sklearn import linear_model, datasets
from sklearn.metrics import classification_report, cohen_kappa_score,f1_score
from sklearn.neural_network import BernoulliRBM
from keras.models import Sequential
from keras.layers import Dense, Activation,Concatenate,Dropout
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import np_utils,plot_model
import pandas as pd
import os
from time import time
import json
import pickle
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from readpoint import readpoint
from sklearn import svm
from sklearn.svm import SVC
# np.random.seed(22)

import sys
import codecs

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
output = sys.stdout
outputfile = open("log.txt", "a+",encoding='utf-8')
sys.stdout = outputfile
type = sys.getfilesystemencoding()


import keras.backend.tensorflow_backend as KTF
os.environ['CUDA_VISIBLE_DEVICES']="0"
config = tf.ConfigProto(device_count={'gpu': 0})
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction =0.4
session = tf.Session(config=config)
# 设置session
KTF.set_session(session )


class DBN():
    def __init__(
            self,
            train_data, targets, labeltrain,first_targets,first_labeltrain,
            test_data, test_targets, labeltest,first_test_targets,first_labeltest,
            layers,
            outputs,
            rbm_iters,
            rbm_lr,  # 学习率，
            epochs=25,
            fine_tune_batch_size=512,
            outdir="./temp/",
            logdir="./logs/",
            pre = [],
            tre = [],
            valdata = None,
            valtarget = None,
            first_valtarget=None,
    ):

        self.hidden_sizes = layers
        self.outputs = outputs

        self.data = train_data
        self.targets = targets
        self.labeltrain = labeltrain
        self.first_targets=first_targets
        self.first_labeltrain=first_labeltrain

        self.test_data = test_data
        self.test_targets = test_targets
        self.labeltest =labeltest
        self.first_test_targets=first_test_targets
        self.first_labeltest=first_labeltest
        # print('self.labeltest:-----------------------')
        # print(self.labeltest[:5])



        self.valdata = valdata
        self.valtarget = valtarget
        self.first_valtarget=first_valtarget

        self.pre = []
        self.tre = labeltest
        self.rbm_learning_rate = rbm_lr
        self.rbm_iters = rbm_iters

        self.epochs = epochs
        self.nn_batch_size = fine_tune_batch_size

        self.rbm_weights = []
        self.rbm_biases = []
        self.rbm_h_act = []

        self.model = None
        self.history = None

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        if outdir[-1] != '/':
            outdir = outdir + '/'

        self.outdir = outdir
        self.logdir = logdir


    #预训练
    def pretrain(self):

        visual_layer = self.data
        print(self.data)
        for i in range(len(self.hidden_sizes)):
            print(visual_layer.shape)
            print("[DBN] Layer {} Pre-Training".format(i + 1))

            rbm = BernoulliRBM(n_components=self.hidden_sizes[i], n_iter=self.rbm_iters,
                               learning_rate=self.rbm_learning_rate,verbose=2, batch_size=64)
            rbm.fit(visual_layer)
            self.rbm_weights.append(rbm.components_)

            self.rbm_biases.append(rbm.intercept_hidden_)

            self.rbm_h_act.append(rbm.transform(visual_layer))


            visual_layer = self.rbm_h_act[-1]
            print(visual_layer.shape)
        print(visual_layer)
        #self.data=visual_layer

    #微调
    def finetune(self,dbnm,sl,change):
        dbn_s_start = time()
        print(self.hidden_sizes,dbnm.hidden_sizes)
        #输入层
        inputs_01=Input((self.data.shape[1],),name='inputs_01')
        inputs_02=Input((dbnm.data.shape[1],),name='inputs_02')
        #全连接层
        #dense_d01 = Dropout(0.2)(inputs_01)
        dense_010=Dense(units=self.hidden_sizes[0],name='rbm_10',activation='sigmoid')(inputs_01)
        dense_011 = Dense(units=self.hidden_sizes[0], name='rbm_11', activation='sigmoid')(dense_010)
        dense_012 = Dense(units=self.hidden_sizes[0], name='rbm_12', activation='sigmoid')(dense_011)
        dense_013 = Dense(units=self.hidden_sizes[0],name='rbm_13',activation='sigmoid')(dense_012)
        dense_014 = Dense(units=self.hidden_sizes[0], name='rbm_14', activation='sigmoid')(dense_013)
        #dense_d02=Dropout(0.2)(inputs_02)
        dense_020 = Dense(units=dbnm.hidden_sizes[0], name='rbm_20', activation='sigmoid')(inputs_02)
        dense_021 = Dense(units=dbnm.hidden_sizes[0], name='rbm_21', activation='sigmoid')(dense_020)
        dense_022 = Dense(units=dbnm.hidden_sizes[0], name='rbm_22', activation='sigmoid')(dense_021)
        dense_023 = Dense(units=dbnm.hidden_sizes[0], name='rbm_23', activation='sigmoid')(dense_022)
        dense_024 = Dense(units=dbnm.hidden_sizes[0], name='rbm_24', activation='sigmoid')(dense_023)
        #合并层
        merge = Concatenate(name='merge')([dense_014,dense_024])
        #dense_00=Dropout(0.1,name='dense_00')(merge)
        #dense_01=Dense(units=1000,activation='sigmoid',name='dense_01')(merge)
        #dense_02=Dense(units=512,activation='sigmoid',name='dense_02')(dense_01)
        #dense_03=Dense(units=256,activation='sigmoid',name='dense_03')(dense_02)
        #输出层
        output_1=Dense(units=20,activation='softmax',name='output_1')(merge)
        output_2=Dense(units=7,activation='softmax',name='output_2')(merge)
        model = Model(inputs=[inputs_01, inputs_02], outputs=[output_1,output_2])
        # 显示模型情况
        plot_model(model, show_shapes=True)
        print(model.summary())
        '''
        for i in range(len(self.hidden_sizes)):               #对每一个rbm
                                                              #hidden_sizes =[400,200,60,10]
            if i == 0:
                model.add(Dense(self.hidden_sizes[i], activation='sigmoid', input_dim=self.data.shape[1],
                                name='rbm_{}'.format(i)))
                model2.add(Dense(self.hidden_sizes[i], activation='sigmoid', input_dim=self.data.shape[1],
                                name='rbm_{}'.format(i)))

            else:
                model.add(Dense(self.hidden_sizes[i], activation='sigmoid', name='rbm_{}'.format(i)))


        model.add(Dense(self.outputs, activation='softmax'))           #
        '''
        #from keras import optimizers
        #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(optimizer='adam',
                      loss={
                          'output_1':'categorical_crossentropy',
                          'output_2':'categorical_crossentropy'
                      },
                      metrics=['accuracy'],
                      loss_weights={'output_1': 1,
                                    'output_2':change,
                                    }
                      )

        for i in range(len(self.hidden_sizes)):
            layer1 = model.get_layer('rbm_1{}'.format(i))
            layer1.set_weights([self.rbm_weights[i].transpose(), self.rbm_biases[i]])
            layer2=model.get_layer('rbm_2{}'.format(i))
            layer2.set_weights([dbnm.rbm_weights[i].transpose(),dbnm.rbm_biases[i]])
        #checkpointer = ModelCheckpoint(filepath=self.outdir + "dbn_weights.hdf5", verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir=self.logdir)


        self.history = model.fit({'inputs_01':self.data,'inputs_02':dbnm.data}, {'output_1':self.targets,'output_2':self.first_targets},
                                 epochs=self.epochs,
                                 batch_size=self.nn_batch_size,verbose=2,callbacks=[tensorboard], validation_data=([self.valdata,dbnm.valdata],[self.valtarget,self.first_valtarget]),
                                 )
        plot_picture(self.history,sl,change)
        #print(self.history)
        dbn_s_end = time()
        dbn_s_span = dbn_s_end-dbn_s_start
        print("dbn_s 训练时间 : {}".format(dbn_s_span))
        self.model = model
        ###################################################
        model.save("DBN-S-{0}-{1}.h5".format(sl,change))
        return self.history.history['val_output_1_acc']
        dbn_s_pre_start = time()
        predict_y = model.predict(x=[self.test_data,dbnm.test_data], batch_size=64)
        a=predict_y[0]
        print(len(a),len(a[0]))

        dbn_s_pre_end = time()
        dbn_s_pre_span = dbn_s_pre_end-dbn_s_pre_start

        print("dbn_s 预测时间 : {}".format(dbn_s_pre_span))
        self.pre=[]
        for i in range(len(a)):
            label = int(np.argmax(a[i]))
            self.pre.append(label)



        acc = 0.0
        self.tre_new = []
        for i in range(len(a)):
            self.tre_new.append(int(self.tre[i]))
            if  int(self.pre[i]) == int(self.tre[i]):
                acc = acc + 1

        res = []
        res.append(self.tre_new)
        res.append(self.pre)
        print(len(res),len(res[0]))
        res = pd.DataFrame(np.array(res).T)
        print(res.shape)
        res.to_csv(
            "dbn-s{}.txt".format(int(change*10)),
            header=[
                "tclass2",
                "pclass2"],
            index=None)

        score = float((acc / len(a)))
        kappa_ = cohen_kappa_score(self.tre_new,self.pre)
        f1score = f1_score(self.tre_new,self.pre,average="weighted")

        # print('DBN的accuracy为：%f' % score)
        f1_measure = my_confusion_matrix(self.tre_new,self.pre)
        # juzheng = confusion_matrix(self.tre_new,self.pre)  #打印混淆矩阵
        # print(juzheng)    #对比混淆矩阵，看是否出错
        print('DBN的accuracy、Kappa系数、f1score依次为：%0.4f,%0.4f,%0.4f' % (score, kappa_, f1score))
        print(f1_measure)
        # print('DBN的kappa=系数为：%f' % kappa_ )
        #print('########################################################')
        self.labeltrain_new = []
        self.labeltest_new = []
        print(self.labeltrain.shape)
        print(self.labeltest.shape)
        # self.labeltest = list(self.labeltest)
        for i in range(len(self.labeltrain)):
            self.labeltrain_new.append(int(self.labeltrain[i]))
        #     self.labeltest_new.append(str(self.labeltest[i]))
        # print(self.labeltrain_new[:5])
        # print(self.labeltest_new[:5])
        for i in range(len(a)):
            label = int(np.argmax(self.test_targets[i]))
            self.labeltest_new.append(label)
        layer_merge=self.model.get_layer(name='dense_01')

        model_extract = Model(inputs=self.model.input, outputs=layer_merge.output)
        tmp1 = time()
        feature_trainx  = model_extract.predict(x=[self.data,dbnm.data], batch_size=512)  # +++++++++++++++++++++++++++
        feature_testx = model_extract.predict(x=[self.test_data,dbnm.test_data], batch_size=512)
        tmp2 = time()
        print("提取dbn特征耗时 : {}".format(tmp2-tmp1))
        feature_trainx = np.array(feature_trainx)
        print(feature_trainx.shape)
        # tr_x = pd.DataFrame(feature_trainx)
        # tr_y = pd.DataFrame({'trainlabel': self.labeltrain_new})  # label
        # train_features = pd.concat([tr_x, tr_y], axis=1)  # 合并
        #train_features.to_csv(r'.\output\train_features3.csv', index=False)

        feature_testx = np.array(feature_testx)
        print(feature_testx.shape)
        # te_x = pd.DataFrame(feature_testx)
        # te_y = pd.DataFrame({'testlabel': self.labeltest_new})  # label
        # test_features = pd.concat([te_x, te_y], axis=1)  # 合并
        #test_features.to_csv(r'.\output\test_features3.csv', index=False)
        rf_start = time()
        rfacc=RF(feature_trainx, self.labeltrain_new, feature_testx, self.labeltest_new,change)
        rf_end = time()
        rf_span = rf_end - rf_start
        print("rf预测时间为 : {}".format(rf_span))
        svc_start = time()
        #svcacc=svc(feature_trainx,self.labeltrain_new,feature_testx,self.labeltest_new)
        svc_end = time()
        svc_span = svc_end-svc_start
        print("svc预测时间为 : {}".format(svc_span))

        return [score,rfacc]

def my_confusion_matrix(y_true, y_pred):
    # 输出混淆矩阵
    #mylabels = list(set(y_true))
    mylabels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    y_true = list(y_true)
    y_pred = list(y_pred)
    conf_mat = confusion_matrix(y_true, y_pred, labels=mylabels)
    print("confusion_matrix(left labels: y_true, up labels: y_pred):")
    print("labels", " ", end='')
    for i in range(len(mylabels)):
        print(mylabels[i], " ", end='')
    print('\n')
    for i in range(len(conf_mat)):
        print(i, " ", end='')
        for j in range(len(conf_mat[i])):
            print(conf_mat[i][j], " ", end='')
        print('\n')
    print('\n')
    # print('混淆矩阵-----')
    # print(type(conf_mat))
    # print(conf_mat)
    print('kappa系数--------------')
    # 输出kappa系数
    dataMat = np.array(conf_mat)
    # print(dataMat)
    P0 = 0.0
    for i in range(len(dataMat)):
        P0 += dataMat[i][i] * 1.0
    xsum = np.sum(dataMat, axis=1)
    xsum = np.array(xsum)
    # print(xsum)
    ysum = np.sum(dataMat, axis=0)
    ysum = np.array(ysum)
    # print(ysum)
    ############计算Kappa系数###########
    # t = 0
    # for i in range(len(xsum)):
    #     t += (xsum[i] * ysum[i]) * 1.0
    # # xsum是个k行1列的向量，ysum是个1行k列的向量
    # Pe = float(t / (dataMat.sum() ** 2))
    # P0 = float(P0 / dataMat.sum() * 1.0)
    # cohens_coefficient = float((P0 - Pe) / (1 - Pe))
    # print('DBN的kappa系数为: %f' % cohens_coefficient)

    recall = []
    precision = []
    f1_measure = []
    for i in range(len(xsum)):
        reca = (dataMat[i][i] / xsum[i])
        recall.append(reca)

        prec = (dataMat[i][i] / ysum[i])
        precision.append(prec)

        f1 = 2 * recall[i] * precision[i] / (recall[i] + precision[i])
        f1_measure.append(f1)
    # print(recall)
    # print(precision)
    # print(f1_measure)
    # print(np.average(f1_measure))   #f1-score值
    return f1_measure

from sklearn.ensemble import RandomForestClassifier

def RF(traindata,trainlabel,testdata,testlabel,change):
    clf = RandomForestClassifier(n_estimators=500,max_features=100,n_jobs=6)  #n_estimators=30,max_features='log2'
    clf.fit(traindata, trainlabel)
    # print(traindata.shape)
    # print(len(trainlabel))
    # print(testdata.shape)
    # print(len(testlabel))
    pred_testlabel1 = clf.predict(testdata)  # 预测的标签

    res = []
    res.append(testlabel)
    res.append(pred_testlabel1)
    res = pd.DataFrame(np.array(res).T)
    res.to_csv(
        "dbn-rf{}.txt".format(change*10),
        header=[
            "tclass2",
            "pclass2"],
        index=None)
    num = len(pred_testlabel1)
    acc = 0.0

    for i in range(num):
        if str(pred_testlabel1[i]) == str(testlabel[i]):
            acc += 1

    score = float((acc / num))
    kappa_ = cohen_kappa_score(testlabel, pred_testlabel1)
    f1score = f1_score(testlabel, pred_testlabel1,average="weighted")

    # score = float((acc / num))
    # print('dbn+RF的accuracy为：%f' % score)
    f1_measure = my_confusion_matrix(testlabel, pred_testlabel1)
    # print('dbn+RF的Kappa系数为：%f' % kappa_)
    print('DBN+RF的accuracy、Kappa系数、f1score依次为：%0.4f,%0.4f,%0.4f' % (score, kappa_, f1score))
    print(f1_measure)
    return score

def svc(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")

    svcClf = SVC(C=2,kernel="rbf", gamma=pow(2,-5), decision_function_shape='ovr')
    svcClf.fit(traindata,trainlabel)
    pred_testlabel = svcClf.predict(testdata)  #预测的标签

    res = []
    res.append(testlabel)
    res.append(pred_testlabel)
    res = pd.DataFrame(np.array(res).T)
    res.to_csv(
        "dbn-svm.txt",
        header=[
            "tclass2",
            "pclass2"],
        index=None)


    num = len(pred_testlabel)
    acc = 0.0
    testlabel_new = []
    for i in range(num):
        testlabel_new.append(int(testlabel[i]))
        if int(pred_testlabel[i]) == int(testlabel[i]):
            acc = acc + 1


    score = float((acc / num))
    kappa_ = cohen_kappa_score(testlabel_new, pred_testlabel)
    f1score = f1_score(testlabel_new, pred_testlabel,average="weighted")
    f1_measure = my_confusion_matrix(testlabel_new, pred_testlabel)
    print('DBN+SVM的accuracy、Kappa系数、f1score依次为：%0.4f,%0.4f,%0.4f' % (score, kappa_, f1score))
    print(f1_measure)
    return score

def plot_picture(history,i,j):#需要等待
    from matplotlib import pyplot as plt
    '''
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs=history.epoch
    plt.plot(epochs, loss, 'c', label='Training loss')
    plt.plot(epochs,val_loss,'b',label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('fig9/save_loss{}.png'.format(i))
    plt.clf()
    plt.plot(epochs, acc, 'c', label='Training acc ')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig('fig9/save_acc{}.png'.format(i))

    plt.clf()
    '''
    acc1=history.history['output_1_acc']
    acc2=history.history['output_2_acc']
    val_acc1=history.history['val_output_1_acc']
    val_acc2=history.history['val_output_2_acc']
    loss = history.history['loss']
    val_loss=history.history['val_loss']
    output_1_loss=history.history['output_1_loss']
    output_2_loss=history.history['output_2_loss']
    val_output_1_loss=history.history['val_output_1_loss']
    val_output_2_loss=history.history['val_output_2_loss']
    epochs=history.epoch
    plt.plot(epochs,loss,'c',label='Training loss')
    plt.plot(epochs,output_1_loss,'m','Training loss 1(20)')
    plt.plot(epochs,output_2_loss,'y','Training loss 2(7)')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Tr-Loss')
    plt.legend()
    plt.savefig('fig9/save_Tr_loss-{0}-{1}.png'.format(i,j * 10))
    plt.clf()

    plt.plot(epochs,val_loss,'b',label='Validation loss')
    plt.plot(epochs,val_output_1_loss,'r',label='Validation loss 1(20)')
    plt.plot(epochs,val_output_2_loss,'g',label='Validation loss 2(7)')
    plt.title('validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Val-Loss')
    plt.legend()
    plt.savefig('fig9/save_Va_loss-{0}-{1}.png'.format(i,j*10))

    plt.clf()

    plt.plot(epochs,acc1,'c',label='Training acc 1(20)')
    plt.plot(epochs,acc2,'m',label='Training acc2(7)')
    plt.title('Training acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig('fig9/save_Tr_acc-{0}-{1}.png'.format(i,j * 10))

    plt.clf()
    plt.plot(epochs,val_acc1,'b',label='Validation acc 1(20)')
    plt.plot(epochs,val_acc2,'g',label='Validation acc 2(7)')
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig('fig9/save_Va_acc-{0}-{1}.png'.format(i,j * 10))

    plt.clf()

if __name__ == '__main__':

    print("开始试验")
    starttime = datetime.datetime.now()
    all_start = time()
    sl=61
    # print(starttime)
    __spec__ = None
    num_class = 20
    first_num_class=7
    fs = 106  #特征数量
    # print('trainx的shape为——{}'.format(trainx.shape)
    # print('trainy的shape为——{}'.format(trainy.shape))
    # print('testx的shape为——{}'.format(testx.shape))
    # print('testy的shape为——{}'.format(testy.shape))

    cengshu = [5]
    jiedian = [1500]
    jiedian2=[540,540,710,710,710,710,710,740,740,740,740,740,910,910,910,910,910]
    # cengshu = [1]
    # jiedian = [60]-+++++++++++++

    # suiji = 666
    for i in cengshu:
        for j in jiedian:
            for k in jiedian2:

                #sum=np.zeros(20)
                #for ml in range(5):
                lk=0.1
                acc = []
                while lk <= 10:
                    print('这次训练有--------{}层,其中每层有--------{}个节点'.format(i, j))
                    mylist = []
                    mylist2=[]
                    mylist.append(j)
                    mylist2.append(k)
                    mylayer = mylist * i
                    mylayer2=mylist2*i
                    print(i,j,mylayer,mylayer2)
                    # suiji += 5···············
                    # np.random.seed(22)
                    trainx, trainy,first_trainy, y_train,first_y_train = readpoint(r'data/train1.txt', num_class,first_num_class, fs)  # y_train
                    trainx1=trainx[:,:102]
                    trainx2=trainx[:,102:105]
                    trainx1=np.insert(trainx1,-1,trainx[:,105],axis=1)
                    #print(trainx1.shape,trainx2.shape)
                    valdata, valtarget,first_valtarget,y_vjal,first_y_vjal = readpoint(r'data/val1.txt',num_class,first_num_class,fs)   #y_train
                    valdata1=valdata[:,:102]
                    valdata2=valdata[:,102:105]
                    valdata1=np.insert(valdata1,-1,valdata[:,105],axis=1)

                    testx, testy, first_testy,y_test,first_y_test = readpoint(r'data/test1.txt', num_class, first_num_class,fs)  # y_test 非onehot编码
                    testx1=testx[:,:102]
                    testx2=testx[:,102:105]
                    testx1=np.insert(testx1,-1,testx[:,105],axis=1)
                    #print(trainx1.shape,trainx2.shape)
                    #print(testx[:5,:5])


                    dbn1 = DBN(train_data=trainx1, targets=trainy, labeltrain=y_train, first_targets=first_trainy,
                            first_labeltrain=first_y_train,
                            test_data=testx1, test_targets=testy, labeltest=y_test, first_test_targets=first_testy,
                            first_labeltest=first_y_test,
                            layers=mylayer,
                            outputs=20,
                            rbm_lr=0.0001,
                            epochs=800,
                            rbm_iters=5,
                            fine_tune_batch_size=2048,
                            outdir=r".\output",
                            logdir=r".\output\log",
                            valdata = valdata1,
                            valtarget = valtarget,
                            first_valtarget=first_valtarget,
                            )
                    dbn2 = DBN(
                            train_data=trainx2, targets=trainy, labeltrain=y_train, first_targets=first_trainy,
                            first_labeltrain=first_y_train,
                            test_data=testx2, test_targets=testy, labeltest=y_test, first_test_targets=first_testy,
                            first_labeltest=first_y_test,
                            layers=mylayer2,
                            outputs=20,
                            rbm_lr=0.0001,
                            epochs=800,
                            rbm_iters=5,
                            fine_tune_batch_size=2048,
                            outdir=r".\output",
                            logdir=r".\output\log1",
                            valdata = valdata2,
                            valtarget = valtarget,
                            first_valtarget=first_valtarget,

                            )
                    pre_start = time()
                    dbn1.pretrain()
                    dbn2.pretrain()
                    pre_end = time()
                    pre_span = pre_end - pre_start
                    print("预训练耗时 : {}".format(pre_span))
                    fine_start = time()
                    '''
                        f=open('acc1.txt','a')
                        acc=dbn1.finetune(dbn2)
                        f.write(str(acc))
                        fine_end=time()
                        fine_span=fine_end-fine_start
                        print("微调耗时 : {}".format(fine_span))
                        print(acc)
                        f.close()
                    '''
                    #lk = 0.1

                    f=open('acc1.txt','a')

                    acc1=dbn1.finetune(dbn2,sl,lk)
                    acc.append(acc1[799])
                    f.write(str(acc)+'\n')
                    fine_end = time()
                    fine_span = fine_end-fine_start
                    print("微调耗时 : {}".format(fine_span))
                    if lk<1.0:
                        lk+=0.1
                    else:
                        lk=int(lk)
                        lk+=1
                    print(acc)
                    f.close()
                    sl+=1

    all_end = time()
    time_span = all_end-all_start
    print("整个实验耗时 : {}".format(time_span))
    endtime = datetime.datetime.now()
    # print(endtime-starttime)
