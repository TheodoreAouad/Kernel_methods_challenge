import numpy as np
import csv
import os


def create_sol(models, kernels):
    '''This function creates the last vector of labels
    Input: 3 models, 3 sets of data
    output: 1 vector of (-1,1) of size 3000'''

    s1, s2, s3 = kernels
    Xtr1, ytr1 = s1.get_train2000()
    Xtest1 = s1.get_test2000()[0]
    Xtr2, ytr2 = s2.get_train2000()
    Xtest2 = s2.get_test2000()[0]
    Xtr3, ytr3 = s3.get_train2000()
    Xtest3 = s3.get_test2000()[0]

    m1, m2, m3 = models
    m1.train(Xtr1, ytr1)
    m2.train(Xtr2, ytr2)
    m3.train(Xtr3, ytr3)

    y1 = m1.predict(Xtest1)
    y2 = m2.predict(Xtest2)
    y3 = m3.predict(Xtest3)
    ytest = np.concatenate((y1, y2, y3))
    ytest[ytest == -1] = 0
    return ytest


def write_csv(Ytest, path):
    '''This function writes a vector in a csv file'''
    with open(path, mode="w", newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Id', 'Bound'])
        for idx, label in enumerate(Ytest):
            writer.writerow([str(idx), str(label)])


def write_sol(models, kernels, file="res"):
    '''This function writes the solution in the csv file.
    Input: models and kernels
    Output: none'''

    # Create directory
    dirName = 'result'
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory ", dirName, " created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

    ytest = create_sol(models, kernels)
    write_csv(ytest, dirName + "/" + file + ".csv")
    print("Writing complete.")


def create_sol_pool(pool_models, kernels_lists):
    '''This function creates the last vector of labels
    Input: 3 pooling models, 3 sets of data
    output: 1 vector of (-1,1) of size 3000'''

    Kl1, Kl2, Kl3 = kernels_lists

    Kltrain1 = [K.get_train2000()[0] for K in Kl1]
    ytrain1 = Kl1[0].get_train2000()[1]
    Kltest1 = [K.get_test2000()[0] for K in Kl1]

    Kltrain2 = [K.get_train2000()[0] for K in Kl2]
    ytrain2 = Kl2[0].get_train2000()[1]
    Kltest2 = [K.get_test2000()[0] for K in Kl2]

    Kltrain3 = [K.get_train2000()[0] for K in Kl3]
    ytrain3 = Kl3[0].get_train2000()[1]
    Kltest3 = [K.get_test2000()[0] for K in Kl3]

    m1, m2, m3 = pool_models
    m1.train(Kltrain1, ytrain1)
    m2.train(Kltrain2, ytrain2)
    m3.train(Kltrain3, ytrain3)

    y1 = m1.predict(Kltest1)
    y2 = m2.predict(Kltest2)
    y3 = m3.predict(Kltest3)
    ytest = np.concatenate((y1, y2, y3))
    ytest[ytest == -1] = 0
    return ytest
