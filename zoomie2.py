from concurrent.futures import Future
import os
import os.path
import warnings
from typing import Any, Callable, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, Executor

import numpy as np
import PIL.Image
from morphocut.core import (
    Node,
    Output,
    RawOrVariable,
    ReturnOutputs,
    Stream,
    closing_if_closable,
)
from morphocut.utils import stream_groupby
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.feature import ORB
from skimage.measure import ransac
from skimage.transform import EuclideanTransform


class DummyExecutor(Executor):
    def submit(self, fn, *args, **kwargs) -> Future:
        fut = Future()

        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            fut.set_exception(exc)
        else:
            fut.set_result(result)

        return fut


class _MatchObject:
    def __init__(self, id, kpd) -> None:
        self.id = id
        self.kpd = kpd


def match_hungarian(desc0, desc1, metric=None, quantile=0.9):
    if metric is None:
        if np.issubdtype(desc0.dtype, bool):
            metric = "hamming"
        else:
            metric = "euclidean"

    distances = cdist(desc0, desc1, metric=metric)

    ii, jj = linear_sum_assignment(distances)

    if quantile < 1.0:
        mask = distances[ii, jj].argsort() < len(ii) * quantile
        ii, jj = ii[mask], jj[mask]

    return np.column_stack((ii, jj))


def _match_pair(kpd0, kpd1):
    if kpd0 is None or kpd1 is None:
        return 0

    keypts0, desc0 = kpd0
    keypts1, desc1 = kpd1

    matches = match_hungarian(desc0, desc1)

    if matches.shape[0] < 2:
        return 0

    min_samples = min(len(matches) - 1, 2)
    transform = EuclideanTransform

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "No inliers found.", UserWarning)

        model, inliers = ransac(
            (keypts0[matches[:, 0]], keypts1[matches[:, 1]]),
            transform,
            min_samples=min_samples,
            initial_inliers=None,
            is_data_valid=None,
            is_model_valid=None,
            random_state=None,
            max_trials=100,
            residual_threshold=3,
            stop_probability=1,
            stop_residuals_sum=0,
        )

    if inliers is None:
        return 0

    return inliers.mean()


DetectorExtractor = Callable[
    [np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray]]
]


def default_detector_extractor(img):
    detector_extractor = ORB()
    detector_extractor.detect_and_extract(img)
    return detector_extractor.keypoints, detector_extractor.descriptors


class _DuplicateMatcher:
    def __init__(
        self,
        min_similarity=0.25,
        detector_extractor: Optional[DetectorExtractor] = None,
        verbose=False,
        n_workers=None,
    ) -> None:
        self.min_similarity = min_similarity

        if detector_extractor is None:
            detector_extractor = default_detector_extractor

        self.detector_extractor = detector_extractor

        self.verbose = verbose

        self._prev_objects = []
        self._executor = (
            DummyExecutor() if n_workers == 1 else ProcessPoolExecutor(n_workers)
        )  # ThreadPoolExecutor(n_workers)

    def match_and_update(self, images: Tuple[Any, np.ndarray]):
        # ids, imgs = zip(*images)
        # new_objects = list(self._executor.map(self._make_obj, ids, imgs))
        # # new_objects = [self._make_obj(id, img) for id, img in images]

        futures = [
            (id, self._executor.submit(self.detector_extractor, img))
            for id, img in images
        ]
        new_objects = [_MatchObject(id, fut.result()) for id, fut in futures]

        # Store objects if first frame
        if not self._prev_objects:
            self._prev_objects = new_objects
            return [obj.id for obj in new_objects]

        # Match all pairs
        futures = [
            (i, j, self._executor.submit(_match_pair, prev.kpd, cur.kpd))
            for i, prev in enumerate(self._prev_objects)
            for j, cur in enumerate(new_objects)
        ]
        sim_matrix = np.zeros((len(self._prev_objects), len(new_objects)))
        for i, j, fut in futures:
            sim_matrix[i, j] = fut.result()

        # Find best-matching pairs
        ii, jj = linear_sum_assignment(sim_matrix, maximize=True)

        # Update dupset IDs
        for i, j in zip(ii, jj):
            sim = sim_matrix[i, j]
            if sim >= self.min_similarity:
                old_id = new_objects[j].id
                new_objects[j].id = self._prev_objects[i].id

                if self.verbose:
                    print(
                        f"  '{old_id}' is dup of '{self._prev_objects[i].id}' ({sim_matrix[i, j]:.2f})"
                    )

        self._prev_objects = new_objects
        return [obj.id for obj in new_objects]


@ReturnOutputs
@Output("dupset_id")
class DetectDuplicates(Node):
    def __init__(
        self,
        image_id,
        image,
        groupby,
        min_similarity=0.25,
        detector_extractor: Optional[DetectorExtractor] = None,
        verbose=False,
        n_workers=None,
    ):
        super().__init__()

        self.image_id = image_id
        self.image = image
        self.groupby = groupby
        self.min_similarity = min_similarity
        self.detector_extractor = detector_extractor
        self.verbose = verbose
        self.n_workers = n_workers

    def transform_stream(self, stream: Stream) -> Stream:
        duplicate_matcher = _DuplicateMatcher(
            min_similarity=self.min_similarity,
            detector_extractor=self.detector_extractor,
            verbose=self.verbose,
            n_workers=self.n_workers,
        )

        with closing_if_closable(stream):
            for _, substream in stream_groupby(stream, self.groupby):
                objs, images, image_ids = zip(*self._complete_substream(substream))

                # Find matches in frame cache
                matches = duplicate_matcher.match_and_update(
                    (id, img) for id, img in zip(image_ids, images)
                )

                for obj, match in zip(objs, matches):
                    yield self.prepare_output(obj, match)

    def _complete_substream(self, substream):
        for obj in substream:
            image, image_id = self.prepare_input(obj, ("image", "image_id"))
            yield obj, image, image_id


class StoreDupsets(Node):
    def __init__(
        self,
        image_id: RawOrVariable[str],
        dupset_id: RawOrVariable[str],
        image: RawOrVariable[np.ndarray],
        groupby: RawOrVariable[str],
        output_dir: str,
        save_singletons: bool = False,
    ):
        super().__init__()

        self.image_id = image_id
        self.dupset_id = dupset_id
        self.image = image
        self.groupby = groupby
        self.output_dir = output_dir
        self.save_singletons = save_singletons

    def transform_stream(self, stream: Stream) -> Stream:
        with closing_if_closable(stream):
            masters_old = masters = {}
            for (output_dir, _), substream in stream_groupby(
                stream, (self.output_dir, self.groupby)
            ):
                for obj in substream:
                    image_id, dupset_id, image = self.prepare_input(
                        obj, ("image_id", "dupset_id", "image")
                    )

                    dupset_path = os.path.join(output_dir, dupset_id)

                    if image_id == dupset_id:
                        # Save master for later
                        masters[image_id] = image
                    else:
                        # Store image
                        self._store_image(dupset_path, image_id, image)
                        # ... and also the master image, if available
                        master_img = masters_old.pop(dupset_id, None)
                        if master_img is not None:
                            self._store_image(dupset_path, dupset_id, master_img)

                    yield obj

                if self.save_singletons:
                    # Store masters without duplicates
                    for image_id, image in masters_old.items():
                        self._store_image(output_dir, image_id, image)

                masters_old = masters
                masters = {}

    @staticmethod
    def _store_image(path, image_id, image):
        img = PIL.Image.fromarray(image)
        os.makedirs(path, exist_ok=True)
        img.save(os.path.join(path, f"{image_id}.jpg"))
