import os
import cv2
import matplotlib.pyplot as plt
from object_selection_helper import ObjectSelectionHelper
import numpy as np

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

    def apply_color_overlay(self, image, color='blue', alpha=0.5):
        # Define the color values
        color_values = {
            'blue': [255, 0, 0],  # Blue color in BGR
            'pink': [255, 105, 180]  # Pink color in BGR
        }

        # Get the color value from the color name
        color = color_values.get(color.lower())
        if color is None:
            print(f"Invalid color name: {color}")
            return image

        # Create a 3D mask with the specified color where the original mask is white
        overlay = np.zeros((*image.shape, 4), dtype=np.uint8)  # 4th channel for Alpha
        overlay[image == 255, :3] = color
        overlay[image == 255, 3] = 255 * alpha  # Apply alpha to the 4th channel


        return overlay
    
    def process_image(self, method='color_segmentation'):
        images = {'original': cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)}

        if method == 'color_segmentation':
            threshold = self.helper.color_segmentation_lab(self.image)
            suffix = '_color_segmentation'
        elif method == 'edge_detection':
            threshold = self.helper.detect_and_invert_edges(self.image)
            suffix = '_edge_detection'

        images['thresholded'] = threshold
        eroded_image = self.helper.erode_until_max_area(threshold)
        images['eroded'] = eroded_image
        top_regions = self.helper.retain_top_regions_thresholded(eroded_image)
        
        picked_region = self.helper.detect_and_score_regions(top_regions, self.image)
        overlayed_picked_region = self.apply_color_overlay(picked_region, color='blue')
        # Convert picked_region to a 3-channel image
        picked_region_color = cv2.cvtColor(picked_region, cv2.COLOR_GRAY2BGRA)
        hl_picked_region = cv2.addWeighted(overlayed_picked_region, 0.5, picked_region_color, 0.5, 0)
        # todo figure out how to display both images
        images['best_candidate_regions'] = hl_picked_region
        
        closed_image = self.helper.perform_morphological_closing(picked_region)
        images['morphologically_closed'] = closed_image

        # make mask rectangular shape
        rect_mask = self.helper.rectify_mask(picked_region)
        # todo paint the mask other color and display over the mask

        masked_image = self.helper.apply_mask(self.image, rect_mask)
        images['masked'] = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

        self.visualize_and_save(images, suffix + '_steps.png')
        output_path = self.save_final_masked_image(masked_image, suffix + '.png')

        return output_path


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
        # Ensure the masked image has an alpha channel
        if masked_image.shape[2] < 4:
            # Create a new image with an alpha channel
            transparent_image = np.zeros((masked_image.shape[0], masked_image.shape[1], 4), dtype=np.uint8)

            # Copy the RGB channels from the masked image
            transparent_image[:, :, :3] = masked_image

            # Set the alpha channel to the inverse of the mask
            transparent_image[:, :, 3] = (cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY) > 0) * 255
        else:
            transparent_image = masked_image

        # Find the non-transparent points
        non_transparent_points = np.argwhere(transparent_image[:, :, 3] != 0)

        if non_transparent_points.size == 0:
            output_path = os.path.join(self.base_output_dir, os.path.basename(self.input_image_path).replace('.jpg', suffix))
            cv2.imwrite(output_path, transparent_image)

            if self.verbose:
                print(f"Final masked image saved to {transparent_image}")

            return output_path

        # Find the bounding box of those points
        top_left = non_transparent_points.min(axis=0)[:2]
        bottom_right = non_transparent_points.max(axis=0)[:2]

        # Crop the image accordingly
        cropped_image = transparent_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

        output_path = os.path.join(self.base_output_dir, os.path.basename(self.input_image_path).replace('.jpg', suffix))
        cv2.imwrite(output_path, cropped_image)
        if self.verbose:
            print(f"Final masked image saved to {output_path}")

        return output_path

    def run(self):
        self.load_image()
        self.setup_helper()
        color_segmentation = self.process_image(method='color_segmentation')
        edge_detection = self.process_image(method='edge_detection')

        return color_segmentation, edge_detection

# Example usage (This code is commented out for execution purposes):
object_selector = ObjectSelection('dataset/preprocessing_test/2018_08_Antonina-Fedcenko.jpg')
object_selector.run()
