import tensorflow as tf
import numpy as np

def run():
    lr = 0.1
    epochs = 100
    reduce_epochs = [40, 70]
    d = 2
    x = np.random.randn(d).astype(np.float32)
    xp = tf.placeholder(shape=[d], dtype=tf.float32)
    loss = tf.reduce_sum(xp ** 2)
    reg = (tf.reduce_sum(xp) - 1) ** 2
    dl_dx_op = tf.gradients(loss, xp)[0]
    dreg_dx_op = tf.gradients(reg, xp)[0]

    session = tf.Session()
    tf.global_variables_initializer().run(session=session)

    for e in xrange(epochs):
        if e in reduce_epochs:
            lr *= 0.1
        l, r, dl_dx, dreg_dx = session.run([loss, reg, dl_dx_op, dreg_dx_op], feed_dict={xp: x})
        print dl_dx
        print dreg_dx
        dl_dx = dl_dx / (np.linalg.norm(dl_dx))
        dreg_dx = dreg_dx / (np.linalg.norm(dreg_dx))
        grad = 0.49 * dl_dx + .51 * dreg_dx
        grad /= np.linalg.norm(grad)
        print dl_dx
        print dreg_dx
        print x
        print l
        print r
        print grad
        print ''
        x -= lr * grad


run()