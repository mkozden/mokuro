import cv2
import numpy as np
from PIL import Image
from loguru import logger
from scipy.signal.windows import gaussian
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

from comic_text_detector.inference import TextDetector
from manga_ocr import MangaOcr
from manga_ocr.ocr import post_process
from mokuro import __version__
from mokuro.cache import cache
from mokuro.utils import imread
import torch

class MangaPageOcr:
    def __init__(self,
                 pretrained_model_name_or_path='kha-white/manga-ocr-base',
                 force_cpu=False,
                 detector_input_size=1024,
                 text_height=64,
                 max_ratio_vert=16,
                 max_ratio_hor=8,
                 anchor_window=2,
                 mocr_batch_size=64,
                 crop_queue_maxsize=512,
                 ):

        self.text_height = text_height
        self.max_ratio_vert = max_ratio_vert
        self.max_ratio_hor = max_ratio_hor
        self.anchor_window = anchor_window
        self.mocr_batch_size = mocr_batch_size
        self.crop_queue_maxsize = crop_queue_maxsize

        cuda = torch.cuda.is_available()
        device = 'cuda' if cuda and not force_cpu else 'cpu'
        logger.info(f'Initializing text detector using {device}')
        self.text_detector = TextDetector(model_path=cache.comic_text_detector, input_size=detector_input_size,
                                          device=device,
                                          half=False,
                                          act='leaky')
        self.mocr = MangaOcr(pretrained_model_name_or_path, force_cpu)
        if cuda and not force_cpu:
            self.mocr.model.half()

    def __call__(self, img_path):
        img = imread(img_path)
        H, W, *_ = img.shape
        mask, mask_refined, blk_list = self.text_detector(img, refine_mode=1, keep_undetected_mask=True)

        result = {'version': __version__, 'img_width': W, 'img_height': H, 'blocks': []}

        for blk_idx, blk in enumerate(blk_list):

            result_blk = {
                'box': list(blk.xyxy),
                'vertical': blk.vertical,
                'font_size': blk.font_size,
                'lines_coords': [],
                'lines': []
            }

            for line_idx, line in enumerate(blk.lines_array()):
                if blk.vertical:
                    max_ratio = self.max_ratio_vert
                else:
                    max_ratio = self.max_ratio_hor

                line_crops, cut_points = self.split_into_chunks(
                    img, mask_refined, blk, line_idx,
                    textheight=self.text_height, max_ratio=max_ratio, anchor_window=self.anchor_window)

                line_text = ''
                for line_crop in line_crops:
                    if blk.vertical:
                        line_crop = cv2.rotate(line_crop, cv2.ROTATE_90_CLOCKWISE)
                    line_text += self.mocr(Image.fromarray(line_crop))

                result_blk['lines_coords'].append(line.tolist())
                result_blk['lines'].append(line_text)

            result['blocks'].append(result_blk)

        return result

    def _mocr_batch(self, images):
        if not images:
            return []

        images = [img.convert("L").convert("RGB") for img in images]

        pixel_values = self.mocr.processor(images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.mocr.model.device)
        if self.mocr.model.dtype == torch.float16:
            pixel_values = pixel_values.half()

        generated_ids = self.mocr.model.generate(pixel_values, max_length=300)

        texts = self.mocr.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return [post_process(text) for text in texts]

    def call_batch(self, img_paths):
        results = [None] * len(img_paths)
        crop_queue = Queue(maxsize=self.crop_queue_maxsize)

        def process_single_image(img_idx, img_path):
            try:
                img = imread(img_path)
                H, W, *_ = img.shape
                mask, mask_refined, blk_list = self.text_detector(img, refine_mode=1, keep_undetected_mask=True)

                result = {'version': __version__, 'img_width': W, 'img_height': H, 'blocks': []}
                results[img_idx] = result

                for blk_idx, blk in enumerate(blk_list):
                    result_blk = {
                        'box': list(blk.xyxy),
                        'vertical': blk.vertical,
                        'font_size': blk.font_size,
                        'lines_coords': [],
                        'lines': []
                    }
                    result['blocks'].append(result_blk)

                    for line_idx, line in enumerate(blk.lines_array()):
                        if blk.vertical:
                            max_ratio = self.max_ratio_vert
                        else:
                            max_ratio = self.max_ratio_hor

                        line_crops, cut_points = self.split_into_chunks(
                            img, mask_refined, blk, line_idx,
                            textheight=self.text_height, max_ratio=max_ratio, anchor_window=self.anchor_window)

                        result_blk['lines_coords'].append(line.tolist())
                        result_blk['lines'].append([''] * len(line_crops))
                        
                        for line_crop_idx, line_crop in enumerate(line_crops):
                            if blk.vertical:
                                line_crop = cv2.rotate(line_crop, cv2.ROTATE_90_CLOCKWISE)
                            crop_queue.put((img_idx, blk_idx, line_idx, line_crop_idx, Image.fromarray(line_crop)))
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {e}")

        def producer():
            try:
                # Use a limited number of workers to not overwhelm the GPU with too many parallel detection calls,
                # while still parallelizing the CPU-heavy refine_mask and imread.
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

            for i, ocr_text in enumerate(ocr_results):
                img_idx, blk_idx, line_idx, crop_idx = info[i]
                results[img_idx]['blocks'][blk_idx]['lines'][line_idx][crop_idx] = ocr_text

        producer_thread.join()

        for res in results:
            if res is None: continue
            for blk in res['blocks']:
                blk['lines'] = ["".join(chunks) for chunks in blk['lines']]

        return results

    @staticmethod
    def split_into_chunks(img, mask_refined, blk, line_idx, textheight, max_ratio=16, anchor_window=2):
        line_crop = blk.get_transformed_region(img, line_idx, textheight)

        h, w, *_ = line_crop.shape
        ratio = w / h

        if ratio <= max_ratio:
            return [line_crop], []

        else:
            k = gaussian(textheight * 2, textheight / 8)

            line_mask = blk.get_transformed_region(mask_refined, line_idx, textheight)
            num_chunks = int(np.ceil(ratio / max_ratio))

            anchors = np.linspace(0, w, num_chunks + 1)[1:-1]

            line_density = line_mask.sum(axis=0)
            line_density = np.convolve(line_density, k, 'same')
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
