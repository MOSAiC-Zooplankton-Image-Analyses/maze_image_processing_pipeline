class PytorchSegmenter:
    def __init__(self, model, size_multiple_of=None, device=None):
        self.model = model
        self.size_multiple_of = size_multiple_of
        self.device = device

    def __call__(self, image: np.ndarray) -> np.ndarray:
        orig_shape = image.shape[:2]
        h, w = orig_shape

        padding_tb = (0, 0)
        padding_lr = (0, 0)

        if self.size_multiple_of is not None:
            h_new = math.ceil(h / self.size_multiple_of) * self.size_multiple_of
            w_new = math.ceil(w / self.size_multiple_of) * self.size_multiple_of

            padding_tb = (
                (h_new - h) // 2 if h_new > h else 0,
                (h_new - h + 1) // 2 if h_new > h else 0,
            )
            padding_lr = (
                (w_new - w) // 2 if w_new > w else 0,
                (w_new - w + 1) // 2 if w_new > w else 0,
            )

            image = np.pad(image, (padding_tb, padding_lr))

            assert image.shape[:2] == (
                h_new,
                w_new,
            ), f"{image.shape[:2]} vs. {(h_new, w_new)}"

        if image.ndim == 2:
            image = skimage.color.gray2rgb(image)

        import torchvision.transforms.functional as tvtf

        image_tensor = tvtf.to_tensor(image).to(self.device).unsqueeze(0)
        result = self.model(image_tensor)[0].squeeze().cpu().numpy()

        assert (
            result.shape[:2] == image.shape[:2]
        ), f"{result.shape[:2]} vs. {image.shape[:2]}"

        result = result[
            padding_tb[0] : padding_tb[0] + h, padding_lr[0] : padding_lr[0] + w
        ]

        assert result.shape[:2] == (h, w), f"{result.shape[:2]} vs. {(h,w)}"

        return result


def crop_uncrop(fn, arr: np.ndarray):
    # Crop
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    result = fn(arr[rmin:rmax, cmin:cmax])

    # Pad
    padding_tb = (rmin, arr.shape[0] - rmax)
    padding_lr = (cmin, arr.shape[1] - cmax)
    result = np.pad(result, (padding_tb, padding_lr))

    return result


def buffered_generator(buf_size: int):
    def wrap(gen):
        # Don't do multithreading if nothing should be buffered
        if buf_size == 0:
            return gen

        @functools.wraps(gen)
        def wrapper(*args, **kwargs):
            q = queue.Queue(buf_size)
            _sentinel = object()

            def fill_queue():
                for item in gen(*args, **kwargs):
                    q.put(item)

                q.put(_sentinel)

            threading.Thread(target=fill_queue, daemon=True).start()

            while True:
                item = q.get()

                # print("qsize", q.qsize())

                if item is _sentinel:
                    return

                yield item

        return wrapper

    return wrap


class TilingPytorchSegmenter:
    def __init__(
        self,
        model,
        tile_shape: Tuple[int, int] = (1024, 1024),
        overlap: float = 0.125,
        device=None,
    ):
        self.model = model
        self.tile_shape = tile_shape
        self.overlap = overlap
        self.device = device

    def __call__(self, image: np.ndarray) -> np.ndarray:
        import torch
        import torchvision.transforms.functional as tvtf

        # Append channel dimension (if any)
        tile_shape = self.tile_shape + image.shape[len(self.tile_shape) :]
        channel_dim = None if len(tile_shape) == len(self.tile_shape) else 2

        image_tiler = tiler.Tiler(
            data_shape=image.shape,
            tile_shape=tile_shape,
            overlap=self.overlap,
            channel_dimension=channel_dim,
        )

        mask_tiler = tiler.Tiler(
            data_shape=image.shape[:2],
            tile_shape=self.tile_shape,
            overlap=self.overlap,
            channel_dimension=None,
        )

        mask_merger = tiler.Merger(mask_tiler)

        @buffered_generator(8)
        def result_gen():
            for tile_id, tile in image_tiler(image):
                if not np.any(tile):
                    continue

                if tile.ndim == 2:
                    tile = skimage.color.gray2rgb(tile)

                tile_tensor = (
                    tvtf.to_tensor(tile).to(self.device, non_blocking=True).unsqueeze(0)
                )
                with torch.no_grad():
                    result = self.model(tile_tensor)[0].squeeze()

                yield tile_id, result

        for tile_id, result in result_gen():
            mask_merger.add(tile_id, result.cpu().numpy())

        return mask_merger.merge(dtype=bool)
