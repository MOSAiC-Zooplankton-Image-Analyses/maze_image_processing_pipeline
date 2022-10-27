from concurrent.futures import Future
import itertools
import os
import os.path
import warnings
from typing import Any, Callable, Iterable, Optional, Tuple, TypeVar
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
    def __init__(
        self, id: Any, img: np.ndarray, score_args: Any, kpd: Any = None
    ) -> None:
        self.id = id
        self.img = img
        self.kpd = kpd
        self.score_args = score_args
        self.age = 0

    def inc_age(self):
        self.age += 1
        return self.age


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


def _calc_match_score(kpd0, kpd1) -> float:
    if kpd0 is None or kpd1 is None:
        return 0

    keypts0, desc0 = kpd0
    keypts1, desc1 = kpd1

    matches = match_hungarian(desc0, desc1)

    if matches.shape[0] < 2:
        return 0

    min_samples = min(len(matches) - 1, 8)
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

    return inliers.mean()  # type: ignore


def _match_pair(
    score_fn,
    prev: _MatchObject,
    cur: _MatchObject,
):
    score = _calc_match_score(prev.kpd, cur.kpd)

    if score_fn is None:
        return score

    return score_fn(score, prev.score_args, cur.score_args)


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
        score_fn=None,
        pre_score_thr=None,
        max_age=1,
    ) -> None:
        self.min_similarity = min_similarity

        if detector_extractor is None:
            detector_extractor = default_detector_extractor

        self.detector_extractor = detector_extractor

        self.verbose = verbose

        self.score_fn = score_fn
        self.pre_score_thr = pre_score_thr

        self.max_age = max_age

        self._prev_objects = []
        self._executor = (
            DummyExecutor() if n_workers == 1 else ProcessPoolExecutor(n_workers)
        )  # ThreadPoolExecutor(n_workers)

    def match_and_update(self, image_ids, images, score_args):
        new_objects = [
            _MatchObject(id, img, score_arg)
            for id, img, score_arg in zip(image_ids, images, score_args)
        ]

        # Store objects if first frame
        if not self._prev_objects:
            self._prev_objects = new_objects
            return [obj.id for obj in new_objects]

        # Find matches quickly without expensive feature calculation
        prev_matched = set()
        new_matched = set()
        if self.score_fn is not None and self.pre_score_thr is not None:
            sim_matrix = np.zeros((len(self._prev_objects), len(new_objects)))
            for i, prev in enumerate(self._prev_objects):
                for j, cur in enumerate(new_objects):
                    sim_matrix[i, j] = self.score_fn(0, prev.score_args, cur.score_args)

            # Find best-matching pairs
            ii, jj = linear_sum_assignment(sim_matrix, maximize=True)

            # Update dupset IDs
            for i, j in zip(ii, jj):
                sim = sim_matrix[i, j]
                if sim >= self.pre_score_thr:
                    old_id = new_objects[j].id
                    new_objects[j].id = self._prev_objects[i].id
                    prev_matched.add(i)
                    new_matched.add(j)

                    if self.verbose:
                        print(
                            f"  '{old_id}' is dup of '{self._prev_objects[i].id}' ({sim_matrix[i, j]:.2f})"
                        )

        # Calculate features of unmatched objects asynchronously
        prev_obj_fut = [
            (obj, self._executor.submit(self.detector_extractor, obj.img))
            for i, obj in enumerate(self._prev_objects)
            if i not in prev_matched
            and obj.kpd is None  # kpd could have been calculated previously
        ]

        new_obj_fut = [
            (obj, self._executor.submit(self.detector_extractor, obj.img))
            for j, obj in enumerate(new_objects)
            if j not in new_matched  # obj.kpd is None anyways
        ]

        # Update match objects after feature calculation finished
        for (obj, fut) in itertools.chain(prev_obj_fut, new_obj_fut):
            obj.kpd = fut.result()

        # Match all pairs of unmatched objects asynchronously
        futures = [
            (
                i,
                j,
                self._executor.submit(_match_pair, self.score_fn, prev, cur),
            )
            for i, prev in enumerate(self._prev_objects)
            if i not in prev_matched
            for j, cur in enumerate(new_objects)
            if j not in new_matched
        ]

        # Collect result of matching
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

        # Update previously seen objects
        prev_objects = {
            obj.id: obj for obj in self._prev_objects if obj.inc_age() <= self.max_age
        }
        prev_objects.update({obj.id: obj for obj in new_objects})
        self._prev_objects = [obj for obj in prev_objects.values()]

        return [obj.id for obj in new_objects]


T = TypeVar("T")


@ReturnOutputs
@Output("dupset_id")
class DetectDuplicates(Node):
    def __init__(
        self,
        image_id,
        image,
        groupby,
        score_fn: Optional[Callable[[float, T, T], float]] = None,
        score_arg: RawOrVariable[T] = None,
        pre_score_thr=None,
        min_similarity=0.25,
        detector_extractor: Optional[DetectorExtractor] = None,
        max_age=1,
        verbose=False,
        n_workers=None,
    ):
        super().__init__()

        self.image_id = image_id
        self.image = image
        self.groupby = groupby
        self.score_fn = score_fn
        self.score_arg = score_arg
        self.pre_score_thr = pre_score_thr
        self.min_similarity = min_similarity
        self.detector_extractor = detector_extractor
        self.verbose = verbose
        self.n_workers = n_workers
        self.max_age = max_age

    def transform_stream(self, stream: Stream) -> Stream:
        duplicate_matcher = _DuplicateMatcher(
            min_similarity=self.min_similarity,
            detector_extractor=self.detector_extractor,
            verbose=self.verbose,
            n_workers=self.n_workers,
            score_fn=self.score_fn,
            pre_score_thr=self.pre_score_thr,
            max_age=self.max_age,
        )

        with closing_if_closable(stream):
            for _, substream in stream_groupby(stream, self.groupby):
                objs, images, image_ids, score_args = zip(
                    *self._complete_substream(substream)
                )

                # Find matches in frame cache
                matches = duplicate_matcher.match_and_update(
                    image_ids, images, score_args
                )

                for obj, match in zip(objs, matches):
                    yield self.prepare_output(obj, match)

    def _complete_substream(self, substream):
        for obj in substream:
            image, image_id, score_arg = self.prepare_input(
                obj, ("image", "image_id", "score_arg")
            )
            yield obj, image, image_id, score_arg


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

                    dupset_path = os.path.join(output_dir, dupset_id)  # type: ignore

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
