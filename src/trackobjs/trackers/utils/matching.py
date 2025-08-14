import numpy
from scipy.spatial.distance import cdist


def embedding_distance(tracks: list, detections: list, metric: str = "cosine") -> np.ndarray:
    """
    Compute distance between tracks and detections based on embeddings.

    Args:
        tracks (List[STrack]): List of tracks, where each track contains embedding features.
        detections (List[BaseTrack]): List of detections, where each detection contains embedding features.
        metric (str): Metric for distance computation. Supported metrics include 'cosine', 'euclidean', etc.

    Returns:
        (numpy.ndarray): Cost matrix computed based on embeddings with shape (N, M), where N is the number of tracks
            and M is the number of detections.

    Examples:
        Compute the embedding distance between tracks and detections using cosine metric
        >>> tracks = [STrack(...), STrack(...)]  # List of track objects with embedding features
        >>> detections = [BaseTrack(...), BaseTrack(...)]  # List of detection objects with embedding features
        >>> cost_matrix = embedding_distance(tracks, detections, metric="cosine")
    """
    cost_matrix = numpy.zeros((len(tracks), len(detections)), dtype=numpy.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = numpy.asarray([track.curr_feat for track in detections], dtype=numpy.float32)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = numpy.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = numpy.asarray([track.smooth_feat for track in tracks], dtype=numpy.float32)
    cost_matrix = numpy.maximum(0.0, cdist(track_features, det_features, metric))  # Normalized features
    return cost_matrix


def iou_distance(atracks: list, btracks: list) -> numpy.ndarray:
    """
    Compute cost based on Intersection over Union (IoU) between tracks.

    Args:
        atracks (List[STrack] | List[numpy.ndarray]): List of tracks 'a' or bounding boxes.
        btracks (List[STrack] | List[numpy.ndarray]): List of tracks 'b' or bounding boxes.

    Returns:
        (numpy.ndarray): Cost matrix computed based on IoU with shape (len(atracks), len(btracks)).

    Examples:
        Compute IoU distance between two sets of tracks
        >>> atracks = [numpy.array([0, 0, 10, 10]), numpy.array([20, 20, 30, 30])]
        >>> btracks = [numpy.array([5, 5, 15, 15]), numpy.array([25, 25, 35, 35])]
        >>> cost_matrix = iou_distance(atracks, btracks)
    """
    if atracks and isinstance(atracks[0], numpy.ndarray) or btracks and isinstance(btracks[0], numpy.ndarray):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xywha if track.angle is not None else track.xyxy for track in atracks]
        btlbrs = [track.xywha if track.angle is not None else track.xyxy for track in btracks]

    ious = numpy.zeros((len(atlbrs), len(btlbrs)), dtype=numpy.float32)
    if len(atlbrs) and len(btlbrs):
        if len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5:
            ious = batch_probiou(
                numpy.ascontiguousarray(atlbrs, dtype=numpy.float32),
                numpy.ascontiguousarray(btlbrs, dtype=numpy.float32),
            ).numpy()
        else:
            ious = bbox_ioa(
                numpy.ascontiguousarray(atlbrs, dtype=numpy.float32),
                numpy.ascontiguousarray(btlbrs, dtype=numpy.float32),
                iou=True,
            )
    return 1 - ious  # cost matrix


def fuse_score(cost_matrix: numpy.ndarray, detections: list) -> numpy.ndarray:
    """
    Fuse cost matrix with detection scores to produce a single similarity matrix.

    Args:
        cost_matrix (numpy.ndarray): The matrix containing cost values for assignments, with shape (N, M).
        detections (List[BaseTrack]): List of detections, each containing a score attribute.

    Returns:
        (numpy.ndarray): Fused similarity matrix with shape (N, M).

    Examples:
        Fuse a cost matrix with detection scores
        >>> cost_matrix = numpy.random.rand(5, 10)  # 5 tracks and 10 detections
        >>> detections = [BaseTrack(score=numpy.random.rand()) for _ in range(10)]
        >>> fused_matrix = fuse_score(cost_matrix, detections)
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = numpy.array([det.score for det in detections])
    det_scores = numpy.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # fuse_cost
