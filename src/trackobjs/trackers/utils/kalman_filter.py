import numpy
import scipy.linalg


class KalmanFilterXYAH:
    """
    A KalmanFilterXYAH class for tracking bounding boxes in image space using a Kalman filter.

    Implements a simple Kalman filter for tracking bounding boxes in image space. The 8-dimensional state space
    (x, y, a, h, vx, vy, va, vh) contains the bounding box center position (x, y), aspect ratio a, height h, and their
    respective velocities. Object motion follows a constant velocity model, and bounding box location (x, y, a, h) is
    taken as a direct observation of the state space (linear observation model).

    Attributes:
        _motion_mat (numpy.ndarray): The motion matrix for the Kalman filter.
        _update_mat (numpy.ndarray): The update matrix for the Kalman filter.
        _std_weight_position (float): Standard deviation weight for position.
        _std_weight_velocity (float): Standard deviation weight for velocity.

    Methods:
        initiate: Create a track from an unassociated measurement.
        predict: Run the Kalman filter prediction step.
        project: Project the state distribution to measurement space.
        multi_predict: Run the Kalman filter prediction step (vectorized version).
        update: Run the Kalman filter correction step.
        gating_distance: Compute the gating distance between state distribution and measurements.

    Examples:
        Initialize the Kalman filter and create a track from a measurement
        >>> kf = KalmanFilterXYAH()
        >>> measurement = numpy.array([100, 200, 1.5, 50])
        >>> mean, covariance = kf.initiate(measurement)
        >>> print(mean)
        >>> print(covariance)
    """

    def __init__(self):
        """
        Initialize Kalman filter model matrices with motion and observation uncertainty weights.

        The Kalman filter is initialized with an 8-dimensional state space (x, y, a, h, vx, vy, va, vh), where (x, y)
        represents the bounding box center position, 'a' is the aspect ratio, 'h' is the height, and their respective
        velocities are (vx, vy, va, vh). The filter uses a constant velocity model for object motion and a linear
        observation model for bounding box location.

        Examples:
            Initialize a Kalman filter for tracking:
            >>> kf = KalmanFilterXYAH()
        """
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices
        self._motion_mat = numpy.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = numpy.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current state estimate
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: numpy.ndarray):
        """
        Create a track from an unassociated measurement.

        Args:
            measurement (numpy.ndarray): Bounding box coordinates (x, y, a, h) with center position (x, y), aspect ratio a,
                and height h.

        Returns:
            mean (numpy.ndarray): Mean vector (8-dimensional) of the new track. Unobserved velocities are initialized to 0 mean.
            covariance (numpy.ndarray): Covariance matrix (8x8 dimensional) of the new track.

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> measurement = numpy.array([100, 50, 1.5, 200])
            >>> mean, covariance = kf.initiate(measurement)
        """
        mean_pos = measurement
        mean_vel = numpy.zeros_like(mean_pos)
        mean = numpy.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = numpy.diag(numpy.square(std))
        return mean, covariance

    def predict(self, mean: numpy.ndarray, covariance: numpy.ndarray):
        """
        Run Kalman filter prediction step.

        Args:
            mean (numpy.ndarray): The 8-dimensional mean vector of the object state at the previous time step.
            covariance (numpy.ndarray): The 8x8-dimensional covariance matrix of the object state at the previous time step.

        Returns:
            mean (numpy.ndarray): Mean vector of the predicted state. Unobserved velocities are initialized to 0 mean.
            covariance (numpy.ndarray): Covariance matrix of the predicted state.

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> mean = numpy.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = numpy.eye(8)
            >>> predicted_mean, predicted_covariance = kf.predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = numpy.diag(numpy.square(numpy.r_[std_pos, std_vel]))

        mean = numpy.dot(mean, self._motion_mat.T)
        covariance = numpy.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: numpy.ndarray, covariance: numpy.ndarray):
        """
        Project state distribution to measurement space.

        Args:
            mean (numpy.ndarray): The state's mean vector (8 dimensional array).
            covariance (numpy.ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            mean (numpy.ndarray): Projected mean of the given state estimate.
            covariance (numpy.ndarray): Projected covariance matrix of the given state estimate.

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> mean = numpy.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = numpy.eye(8)
            >>> projected_mean, projected_covariance = kf.project(mean, covariance)
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = numpy.diag(numpy.square(std))

        mean = numpy.dot(self._update_mat, mean)
        covariance = numpy.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: numpy.ndarray, covariance: numpy.ndarray):
        """
        Run Kalman filter prediction step for multiple object states (Vectorized version).

        Args:
            mean (numpy.ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (numpy.ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            mean (numpy.ndarray): Mean matrix of the predicted states with shape (N, 8).
            covariance (numpy.ndarray): Covariance matrix of the predicted states with shape (N, 8, 8).

        Examples:
            >>> mean = numpy.random.rand(10, 8)  # 10 object states
            >>> covariance = numpy.random.rand(10, 8, 8)  # Covariance matrices for 10 object states
            >>> predicted_mean, predicted_covariance = kalman_filter.multi_predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * numpy.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * numpy.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = numpy.square(numpy.r_[std_pos, std_vel]).T

        motion_cov = [numpy.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = numpy.asarray(motion_cov)

        mean = numpy.dot(mean, self._motion_mat.T)
        left = numpy.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = numpy.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean: numpy.ndarray, covariance: numpy.ndarray, measurement: numpy.ndarray):
        """
        Run Kalman filter correction step.

        Args:
            mean (numpy.ndarray): The predicted state's mean vector (8 dimensional).
            covariance (numpy.ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (numpy.ndarray): The 4 dimensional measurement vector (x, y, a, h), where (x, y) is the center
                position, a the aspect ratio, and h the height of the bounding box.

        Returns:
            new_mean (numpy.ndarray): Measurement-corrected state mean.
            new_covariance (numpy.ndarray): Measurement-corrected state covariance.

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> mean = numpy.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = numpy.eye(8)
            >>> measurement = numpy.array([1, 1, 1, 1])
            >>> new_mean, new_covariance = kf.update(mean, covariance, measurement)
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), numpy.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + numpy.dot(innovation, kalman_gain.T)
        new_covariance = covariance - numpy.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: numpy.ndarray,
        covariance: numpy.ndarray,
        measurements: numpy.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> numpy.ndarray:
        """
        Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If `only_position` is False, the chi-square
        distribution has 4 degrees of freedom, otherwise 2.

        Args:
            mean (numpy.ndarray): Mean vector over the state distribution (8 dimensional).
            covariance (numpy.ndarray): Covariance of the state distribution (8x8 dimensional).
            measurements (numpy.ndarray): An (N, 4) matrix of N measurements, each in format (x, y, a, h) where (x, y) is the
                bounding box center position, a the aspect ratio, and h the height.
            only_position (bool, optional): If True, distance computation is done with respect to box center position only.
            metric (str, optional): The metric to use for calculating the distance. Options are 'gaussian' for the squared
                Euclidean distance and 'maha' for the squared Mahalanobis distance.

        Returns:
            (numpy.ndarray): Returns an array of length N, where the i-th element contains the squared distance between
                (mean, covariance) and `measurements[i]`.

        Examples:
            Compute gating distance using Mahalanobis metric:
            >>> kf = KalmanFilterXYAH()
            >>> mean = numpy.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = numpy.eye(8)
            >>> measurements = numpy.array([[1, 1, 1, 1], [2, 2, 1, 1]])
            >>> distances = kf.gating_distance(mean, covariance, measurements, only_position=False, metric="maha")
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return numpy.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = numpy.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return numpy.sum(z * z, axis=0)  # square maha
        else:
            raise ValueError("Invalid distance metric")


class KalmanFilterXYWH(KalmanFilterXYAH):
    """
    A KalmanFilterXYWH class for tracking bounding boxes in image space using a Kalman filter.

    Implements a Kalman filter for tracking bounding boxes with state space (x, y, w, h, vx, vy, vw, vh), where
    (x, y) is the center position, w is the width, h is the height, and vx, vy, vw, vh are their respective velocities.
    The object motion follows a constant velocity model, and the bounding box location (x, y, w, h) is taken as a direct
    observation of the state space (linear observation model).

    Attributes:
        _motion_mat (numpy.ndarray): The motion matrix for the Kalman filter.
        _update_mat (numpy.ndarray): The update matrix for the Kalman filter.
        _std_weight_position (float): Standard deviation weight for position.
        _std_weight_velocity (float): Standard deviation weight for velocity.

    Methods:
        initiate: Create a track from an unassociated measurement.
        predict: Run the Kalman filter prediction step.
        project: Project the state distribution to measurement space.
        multi_predict: Run the Kalman filter prediction step in a vectorized manner.
        update: Run the Kalman filter correction step.

    Examples:
        Create a Kalman filter and initialize a track
        >>> kf = KalmanFilterXYWH()
        >>> measurement = numpy.array([100, 50, 20, 40])
        >>> mean, covariance = kf.initiate(measurement)
        >>> print(mean)
        >>> print(covariance)
    """

    def initiate(self, measurement: numpy.ndarray):
        """
        Create track from unassociated measurement.

        Args:
            measurement (numpy.ndarray): Bounding box coordinates (x, y, w, h) with center position (x, y), width, and height.

        Returns:
            mean (numpy.ndarray): Mean vector (8 dimensional) of the new track. Unobserved velocities are initialized to 0 mean.
            covariance (numpy.ndarray): Covariance matrix (8x8 dimensional) of the new track.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> measurement = numpy.array([100, 50, 20, 40])
            >>> mean, covariance = kf.initiate(measurement)
            >>> print(mean)
            [100.  50.  20.  40.   0.   0.   0.   0.]
            >>> print(covariance)
            [[ 4.  0.  0.  0.  0.  0.  0.  0.]
             [ 0.  4.  0.  0.  0.  0.  0.  0.]
             [ 0.  0.  4.  0.  0.  0.  0.  0.]
             [ 0.  0.  0.  4.  0.  0.  0.  0.]
             [ 0.  0.  0.  0.  0.25  0.  0.  0.]
             [ 0.  0.  0.  0.  0.  0.25  0.  0.]
             [ 0.  0.  0.  0.  0.  0.  0.25  0.]
             [ 0.  0.  0.  0.  0.  0.  0.  0.25]]
        """
        mean_pos = measurement
        mean_vel = numpy.zeros_like(mean_pos)
        mean = numpy.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = numpy.diag(numpy.square(std))
        return mean, covariance

    def predict(self, mean: numpy.ndarray, covariance: numpy.ndarray):
        """
        Run Kalman filter prediction step.

        Args:
            mean (numpy.ndarray): The 8-dimensional mean vector of the object state at the previous time step.
            covariance (numpy.ndarray): The 8x8-dimensional covariance matrix of the object state at the previous time step.

        Returns:
            mean (numpy.ndarray): Mean vector of the predicted state. Unobserved velocities are initialized to 0 mean.
            covariance (numpy.ndarray): Covariance matrix of the predicted state.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> mean = numpy.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = numpy.eye(8)
            >>> predicted_mean, predicted_covariance = kf.predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = numpy.diag(numpy.square(numpy.r_[std_pos, std_vel]))

        mean = numpy.dot(mean, self._motion_mat.T)
        covariance = numpy.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: numpy.ndarray, covariance: numpy.ndarray):
        """
        Project state distribution to measurement space.

        Args:
            mean (numpy.ndarray): The state's mean vector (8 dimensional array).
            covariance (numpy.ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            mean (numpy.ndarray): Projected mean of the given state estimate.
            covariance (numpy.ndarray): Projected covariance matrix of the given state estimate.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> mean = numpy.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = numpy.eye(8)
            >>> projected_mean, projected_cov = kf.project(mean, covariance)
        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = numpy.diag(numpy.square(std))

        mean = numpy.dot(self._update_mat, mean)
        covariance = numpy.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: numpy.ndarray, covariance: numpy.ndarray):
        """
        Run Kalman filter prediction step (Vectorized version).

        Args:
            mean (numpy.ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (numpy.ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            mean (numpy.ndarray): Mean matrix of the predicted states with shape (N, 8).
            covariance (numpy.ndarray): Covariance matrix of the predicted states with shape (N, 8, 8).

        Examples:
            >>> mean = numpy.random.rand(5, 8)  # 5 objects with 8-dimensional state vectors
            >>> covariance = numpy.random.rand(5, 8, 8)  # 5 objects with 8x8 covariance matrices
            >>> kf = KalmanFilterXYWH()
            >>> predicted_mean, predicted_covariance = kf.multi_predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = numpy.square(numpy.r_[std_pos, std_vel]).T

        motion_cov = [numpy.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = numpy.asarray(motion_cov)

        mean = numpy.dot(mean, self._motion_mat.T)
        left = numpy.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = numpy.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean: numpy.ndarray, covariance: numpy.ndarray, measurement: numpy.ndarray):
        """
        Run Kalman filter correction step.

        Args:
            mean (numpy.ndarray): The predicted state's mean vector (8 dimensional).
            covariance (numpy.ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (numpy.ndarray): The 4 dimensional measurement vector (x, y, w, h), where (x, y) is the center
                position, w the width, and h the height of the bounding box.

        Returns:
            new_mean (numpy.ndarray): Measurement-corrected state mean.
            new_covariance (numpy.ndarray): Measurement-corrected state covariance.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> mean = numpy.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = numpy.eye(8)
            >>> measurement = numpy.array([0.5, 0.5, 1.2, 1.2])
            >>> new_mean, new_covariance = kf.update(mean, covariance, measurement)
        """
        return super().update(mean, covariance, measurement)
