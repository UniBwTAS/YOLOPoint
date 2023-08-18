import numpy as np
from object_msgs.msg import Object


class RV(object):
    X = 0
    Y = 1
    Z = 2
    RX = 3
    RY = 4
    RZ = 5
    dX = 6
    dY = 7
    dZ = 8
    dRX = 9
    dRY = 10
    dRZ = 11
    ddX = 12
    ddY = 13
    ddZ = 14
    ddRX = 15
    ddRY = 16
    ddRZ = 17
    LF = 18
    LR = 19
    WL = 20
    WR = 21
    HT = 22
    HB = 23


class CovarianceHelper(object):

    @staticmethod
    def covariance_from_msg(msg):
        matrix_size = np.array(msg.state_validity, np.bool).sum()
        cov = np.zeros((matrix_size, matrix_size))
        cov[np.triu_indices(matrix_size)] = np.array(msg.complete_covariance)
        cov = cov + cov.T - np.diag(np.diag(cov))  # mirror triangle matrix
        return cov

    @staticmethod
    def covariance_to_msg(cov, mask, msg):
        matrix_size = cov.shape[0]
        msg.complete_covariance = cov[np.triu_indices(matrix_size)]
        msg.state_validity = mask
        return msg

    @staticmethod
    def get_sub_matrix(cov, indices):
        if isinstance(indices, tuple) or isinstance(indices, list):
            indices = np.array(indices, np.int)
        return cov[indices[:, None], indices[None, :]]

    @staticmethod
    def set_sub_matrix(cov, indices, sub):
        if isinstance(indices, tuple) or isinstance(indices, list):
            indices = np.array(indices, np.int)
        cov[indices[:, None], indices[None, :]] = sub
        return cov

    @staticmethod
    def get_mask_from_random_variables(rvs):
        mask = np.zeros(24, np.bool)
        mask[rvs] = True
        return mask

    @staticmethod
    def get_random_variables_from_mask(mask):
        return np.arange(24)[mask]


def example():
    # generate symmetric matrix
    # actually it is no covariance matrix as it is not positive semi definite, but this does not matter here
    a = np.random.normal(size=(4, 4))
    cov = a + a.T
    mask = CovarianceHelper.get_mask_from_random_variables(
        [RV.X, RV.Y, RV.Z, RV.dX])
    print('Original Covariance:')
    print(cov)

    # convert covariance matrix to msg
    new_msg = Object()
    new_msg = CovarianceHelper.covariance_to_msg(cov, mask, new_msg)
    print('Compressed Covariance:')
    print(new_msg.complete_covariance)
    print('Random Variables in Msg')
    print(CovarianceHelper.get_random_variables_from_mask(new_msg.state_validity))

    # convert msg to covariance matrix
    cov = CovarianceHelper.covariance_from_msg(new_msg)
    print('Covariance:')
    print(cov)

    # obtain cov matrix just for certain random variables
    sub_cov = CovarianceHelper.get_sub_matrix(cov, np.array([0, 3], np.int))
    print('Sub Covariance:')
    print(sub_cov)

    # assign cov matrix just for certain random variables
    cov = CovarianceHelper.set_sub_matrix(cov, [0, 3], np.ones((2, 2)) * 99)
    print('Modified Covariance:')
    print(cov)


if __name__ == '__main__':
    example()
