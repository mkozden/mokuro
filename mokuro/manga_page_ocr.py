from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Lock, Thread

import cv2
import numpy as np
from PIL import Image
from loguru import logger
from scipy.signal.windows import gaussian

from comic_text_detector.inference import TextDetector
from comic_text_detector.utils.textmask import REFINEMASK_INPAINT, refine_mask
from manga_ocr import MangaOcr
from manga_ocr.ocr import post_process
from mokuro import __version__
from mokuro.cache import cache
from mokuro.utils import imread
import torch


class MangaPageOcr:
    def __init__(
        self,
        pretrained_model_name_or_path="kha-white/manga-ocr-base",
        force_cpu=False,
        detector_input_size=1024,
        text_height=64,
        max_ratio_vert=16,
        max_ratio_hor=8,
        anchor_window=2,
        disable_ocr=False,
        mocr_batch_size=64,
        crop_queue_maxsize=512,
    ):
        self.text_height = text_height
        self.max_ratio_vert = max_ratio_vert
        self.max_ratio_hor = max_ratio_hor
        self.anchor_window = anchor_window
        self.disable_ocr = disable_ocr
        self.mocr_batch_size = mocr_batch_size
        self.crop_queue_maxsize = crop_queue_maxsize
        # Detector forward passes must be serialized: concurrent calls from multiple threads cause severe
        # contention during cuDNN/MIOpen kernel autotuning (and add GIL/CUDA-stream contention generally),
        # without any real GPU parallelism since it's a single CUDA stream anyway.
        self._detector_lock = Lock()

        if not self.disable_ocr:
            if not force_cpu and torch.cuda.is_available():
                device = "cuda"
            elif not force_cpu and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            half = device == "cuda"
            if device == "cuda":
                # Fixed 1024x1024 detector input, benefits from cuDNN/MIOpen kernel autotuning (also works under ROCm).
                torch.backends.cudnn.benchmark = True
            logger.info(f"Initializing text detector, using device {device}")
            self.text_detector = TextDetector(
                model_path=cache.comic_text_detector,
                input_size=detector_input_size,
                device=device,
                half=half,
                act="leaky",
            )
            self.mocr = MangaOcr(pretrained_model_name_or_path, force_cpu)
            if half:
                self.mocr.model.half()

    def __call__(self, img_path):
        result = self.call_batch([img_path])[0]
        if result is None:
            raise RuntimeError(f"Failed to process {img_path}")
        return result

    def _bucket_size(self, n):
        # Next power of two, capped at mocr_batch_size, so the whole run only ever exercises a handful
        # of distinct batch shapes (bounds cuDNN/MIOpen autotuning).
        size = 1
        while size < n:
            size *= 2
        return min(size, self.mocr_batch_size)

    def _mocr_batch(self, pixel_values_list):
        if not pixel_values_list:
            return []

        real_count = len(pixel_values_list)
        pixel_values = torch.cat(pixel_values_list, dim=0)

        bucket_size = self._bucket_size(real_count)
        if bucket_size > real_count:
            pad = pixel_values[-1:].expand(bucket_size - real_count, -1, -1, -1)
            pixel_values = torch.cat([pixel_values, pad], dim=0)

        device = self.mocr.model.device
        if device.type == "cuda":
            pixel_values = pixel_values.pin_memory().to(device, non_blocking=True)
        else:
            pixel_values = pixel_values.to(device)
        pixel_values = pixel_values.to(self.mocr.model.dtype)

        generated_ids = self.mocr.model.generate(pixel_values, max_length=300)
        generated_ids = generated_ids[:real_count]

        texts = self.mocr.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return [post_process(text) for text in texts]

    def call_batch(self, img_paths, on_result=None):
        """
        Process many pages through a pipelined detector -> batched OCR flow.
        Returns a list of results aligned with img_paths; a failed page yields None.
        If on_result is given, it's called as on_result(img_idx, result) as soon as each page is complete.
        """
        results = [None] * len(img_paths)
        crop_queue = Queue(maxsize=self.crop_queue_maxsize)

        # Per-page crop bookkeeping so on_result can fire as soon as a page's last crop is OCRed,
        # instead of waiting for the whole img_paths batch to drain.
        page_lock = Lock()
        outstanding = [0] * len(img_paths)
        finalized = [False] * len(img_paths)
        fired = [False] * len(img_paths)

        def join_lines(res):
            for blk in res["blocks"]:
                blk["lines"] = ["".join(chunks) for chunks in blk["lines"]]

        def maybe_fire(img_idx):
            if on_result is None:
                return
            with page_lock:
                if not (finalized[img_idx] and outstanding[img_idx] == 0 and not fired[img_idx]):
                    return
                fired[img_idx] = True
            res = results[img_idx]
            if res is not None:
                join_lines(res)
            on_result(img_idx, res)

        def process_single_image(img_idx, img_path):
            try:
                img = imread(img_path)
                H, W, *_ = img.shape
                result = {"version": __version__, "img_width": W, "img_height": H, "blocks": []}

                if self.disable_ocr:
                    results[img_idx] = result
                    return

                with self._detector_lock:
                    mask, _, blk_list = self.text_detector(img, refine_mode=1, skip_refine=True)

                # refine_mask is expensive (Otsu thresholding, connected components) and only needed for lines
                # whose aspect ratio requires splitting into chunks; compute it lazily, once per block.
                mask_refined_cache = {}

                def refine_fn(blk):
                    key = id(blk)
                    if key not in mask_refined_cache:
                        mask_refined_cache[key] = refine_mask(img, mask, [blk], refine_mode=REFINEMASK_INPAINT)
                    return mask_refined_cache[key]

                crops = []
                for blk_idx, blk in enumerate(blk_list):
                    result_blk = {
                        "box": list(blk.xyxy),
                        "vertical": blk.vertical,
                        "font_size": blk.font_size,
                        "lines_coords": [],
                        "lines": [],
                    }
                    result["blocks"].append(result_blk)

                    for line_idx, line in enumerate(blk.lines_array()):
                        if blk.vertical:
                            max_ratio = self.max_ratio_vert
                        else:
                            max_ratio = self.max_ratio_hor

                        line_crops, cut_points = self.split_into_chunks(
                            img,
                            mask,
                            blk,
                            line_idx,
                            textheight=self.text_height,
                            max_ratio=max_ratio,
                            anchor_window=self.anchor_window,
                            refine_fn=refine_fn,
                        )

                        result_blk["lines_coords"].append(line.tolist())
                        result_blk["lines"].append([""] * len(line_crops))

                        for line_crop_idx, line_crop in enumerate(line_crops):
                            if blk.vertical:
                                line_crop = cv2.rotate(line_crop, cv2.ROTATE_90_CLOCKWISE)
                            crop_img = Image.fromarray(line_crop).convert("L").convert("RGB")
                            pixel_values = self.mocr.processor(crop_img, return_tensors="pt").pixel_values
                            crops.append((img_idx, blk_idx, line_idx, line_crop_idx, pixel_values))

                # Publish the result only once the page is fully built, so a failure midway leaves it as None.
                results[img_idx] = result
                with page_lock:
                    outstanding[img_idx] += len(crops)
                for crop in crops:
                    crop_queue.put(crop)
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {e}")
            finally:
                with page_lock:
                    finalized[img_idx] = True
                maybe_fire(img_idx)

        def producer():
            try:
                # Threads parallelize the CPU-heavy imread, crop extraction and lazy refine_mask;
                # detector calls themselves are serialized by _detector_lock.
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for img_idx, img_path in enumerate(img_paths):
                        executor.submit(process_single_image, img_idx, img_path)
            finally:
                crop_queue.put(None)

        producer_thread = Thread(target=producer, daemon=True)
        producer_thread.start()

        while True:
            batch = []
            info = []
            while len(batch) < self.mocr_batch_size:
                item = crop_queue.get()
                if item is None:
                    crop_queue.put(None)
                    break
                batch.append(item[4])
                info.append(item[:4])

            if not batch:
                break

            ocr_results = self._mocr_batch(batch)

            for (img_idx, blk_idx, line_idx, crop_idx), ocr_text in zip(info, ocr_results):
                results[img_idx]["blocks"][blk_idx]["lines"][line_idx][crop_idx] = ocr_text
                with page_lock:
                    outstanding[img_idx] -= 1
                maybe_fire(img_idx)

        producer_thread.join()

        for img_idx, res in enumerate(results):
            if res is None or fired[img_idx]:
                continue
            join_lines(res)

        return results

    @staticmethod
    def split_into_chunks(img, mask, blk, line_idx, textheight, max_ratio=16, anchor_window=2, refine_fn=None):
        line_crop = blk.get_transformed_region(img, line_idx, textheight)

        h, w, *_ = line_crop.shape
        ratio = w / h

        if ratio <= max_ratio:
            return [line_crop], []

        else:
            k = gaussian(textheight * 2, textheight / 8)

            mask_refined = refine_fn(blk) if refine_fn is not None else mask
            line_mask = blk.get_transformed_region(mask_refined, line_idx, textheight)
            num_chunks = int(np.ceil(ratio / max_ratio))

            anchors = np.linspace(0, w, num_chunks + 1)[1:-1]

            line_density = line_mask.sum(axis=0)
            line_density = np.convolve(line_density, k, "same")
            line_density /= line_density.max()

            anchor_window *= textheight

            cut_points = []
            for anchor in anchors:
                anchor = int(anchor)

                n0 = np.clip(anchor - anchor_window // 2, 0, w)
                n1 = np.clip(anchor + anchor_window // 2, 0, w)

                p = line_density[n0:n1].argmin()
                p += n0

                cut_points.append(p)

            return np.split(line_crop, cut_points, axis=1), cut_points
