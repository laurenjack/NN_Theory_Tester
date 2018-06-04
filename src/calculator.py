import numpy as np
logits = np.array([[-8.125, -5.28125], [-2.5, -55.125], [-56.125, -18.28125]], dtype=np.float32)
rbf = np.exp(logits)
exps = np.exp(4.0 * rbf)
a = exps / np.sum(exps, axis=1).reshape(3, 1)

tau_sq_z_diff = np.array([[[-0.5, -0.625], [8.0, -9.0]], [[2.0, 0], [-2.0, -31.5]], [[3.5, 0.375], [20.0, 18.0]]], dtype=np.float32)


y = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
mag = np.sum(tau_sq_z_diff ** 2.0, axis=1) ** 0.5
mag_shaped = mag.reshape([3, 1, 2])
# z_bar_sm_grad = (y - a) #Not correct for sparse version
z_bar_sm_grad = np.array([[-0.50478989, 0.50478989],
                          [-0.41864461, 0.41864458],
                          [0.49999997, -0.5]], dtype=np.float32)
z_bar_sm_grad_shaped = z_bar_sm_grad.reshape([3, 1, 2])
z_sm_grad = y * (1 - a)
z_sm_grad_shaped = z_sm_grad.reshape([3, 1, 2])
dz = z_sm_grad_shaped * -tau_sq_z_diff / mag_shaped
z_update = - 2.0 ** 0.5 * dz

d_z_bar = z_bar_sm_grad_shaped * tau_sq_z_diff / mag_shaped
# print z_bar_sm_grad
print - d_z_bar
#print -tau_sq_z_diff / mag_shaped

