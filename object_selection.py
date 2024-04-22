import os
import cv2
import matplotlib.pyplot as plt
from object_selection_helper import ObjectSelectionHelper

class ObjectSelection:
    def __init__(self, input_image_path, verbose=True):
        self.input_image_path = input_image_path
        self.verbose = verbose
        self.helper = None
        self.image = None
        self.base_output_dir = self.create_output_directory(input_image_path)

    def create_output_directory(self, input_path):
        path_parts = input_path.split('/')
        path_parts[0] += '_masked'
        output_base = '/'.join(path_parts[:-1])
        os.makedirs(output_base, exist_ok=True)
        return output_base

    def load_image(self):
        self.image = cv2.imread(self.input_image_path)
        if self.verbose:
            print(f"Image loaded from {self.input_image_path}")

    def setup_helper(self):
        self.helper = ObjectSelectionHelper(verbose=self.verbose)

    def process_image(self, method='color_segmentation'):
        images = {'original': cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)}

        if method == 'color_segmentation':
            threshold = self.helper.color_segmentation_lab(self.image)
            suffix = '_mask1'
        elif method == 'edge_detection':
            threshold = self.helper.detect_and_invert_edges(self.image)
            suffix = '_mask2'

        images['threshold'] = threshold
        eroded_image = self.helper.erode_until_max_area(threshold)
        images['eroded'] = eroded_image
        top_regions = self.helper.retain_top_regions_thresholded(eroded_image)
        images['top_regions'] = top_regions
        closed_image = self.helper.perform_morphological_closing(top_regions)
        images['closed'] = closed_image

        # todo debug
        #mask = self.helper.detect_and_score_regions(closed_image, self.image)
        mask = closed_image

        masked_image = self.helper.apply_mask(self.image, mask)
        images['masked'] = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

        self.visualize_and_save(images, suffix + '_steps.jpg')
        self.save_final_masked_image(masked_image, suffix + '.jpg')

    def visualize_and_save(self, images, suffix):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()
        for ax, (title, img) in zip(axs, images.items()):
            ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            ax.set_title(title)
            ax.axis('off')
        output_path = os.path.join(self.base_output_dir, os.path.basename(self.input_image_path).replace('.jpg', suffix))
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        if self.verbose:
            print(f"Visualization saved to {output_path}")

    def save_final_masked_image(self, masked_image, suffix):
        final_mask_path = os.path.join(self.base_output_dir, os.path.basename(self.input_image_path).replace('.jpg', suffix))
        cv2.imwrite(final_mask_path, masked_image)
        if self.verbose:
            print(f"Final masked image saved to {final_mask_path}")

    def run(self):
        self.load_image()
        self.setup_helper()
        self.process_image(method='color_segmentation')
        self.process_image(method='edge_detection')

# Example usage (This code is commented out for execution purposes):
object_selector = ObjectSelection('dataset/preprocessing_test/2022_09_Malvine-Gutmane.jpg')
object_selector.run()
