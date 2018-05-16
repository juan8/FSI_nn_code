import gzip
import cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

train_y = one_hot(train_y, 10)
valid_y = one_hot(valid_y, 10)
test_y = one_hot(test_y, 10)



# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print train_y[57]


# TODO: the neural net!!


x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)


print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
arrayGraficaValidation=[]
epoch=0
error=0


while True:
    epoch = epoch + 1

    for jj in xrange(len(train_x) / batch_size):
        batch_train_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_train_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_train_xs, y_: batch_train_ys})

    # print "Train"
    # print "Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_train_xs, y_: batch_train_ys})
    #result = sess.run(y, feed_dict={x: batch_train_xs})
    # print "----------------------------------------------------------------------------------"
    # print ""


    for jj in xrange(len(valid_x) / batch_size):
        batch_validation_xs = valid_x[jj * batch_size: jj * batch_size + batch_size]
        batch_validation_ys = valid_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_validation_xs, y_: batch_validation_ys})

    print "Validation"
    print "Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_validation_xs, y_: batch_validation_ys})
    result = sess.run(y, feed_dict={x: batch_validation_xs})
    print ""

    arrayGraficaValidation.append(sess.run(loss, feed_dict={x: batch_validation_xs, y_: batch_validation_ys}))


    for jj in xrange(len(test_x) / batch_size):
        batch_test_xs = test_x[jj * batch_size: jj * batch_size + batch_size]
        batch_test_ys = test_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_test_xs, y_: batch_test_ys})

    # print "Test"
    # print "Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_test_xs, y_: batch_test_ys})
    #result = sess.run(y, feed_dict={x: batch_test_xs})
    # print """
    # ----------------------------------------------------------------------------------
    # """

    if (abs(arrayGraficaValidation[len(arrayGraficaValidation)-1]-error)<0.02):
        break

    error=arrayGraficaValidation[len(arrayGraficaValidation)-1]



print "Test"
result = sess.run(y, feed_dict={x: test_x})
aciertos=0
fallos=0

for b, r in zip(test_y, result):
    if b.argmax() !=r.argmax():
        fallos = fallos + 1
    else:
        aciertos = aciertos + 1

print "Aciertos: ",aciertos
print "Fallos: ",fallos
total=aciertos+fallos
porcentaje=(float(aciertos)/float(total))*100
print "Porcentaje aciertos:",porcentaje,"%"


plt.title("Validation")
plt.plot(arrayGraficaValidation)
plt._show()