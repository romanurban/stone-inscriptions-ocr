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

    def apply_color_overlay(self, image, color='red'):
        # Define the color values
        color_values = {
            'blue': [255, 0, 0],  # Blue color in BGR
            'red': [0, 0, 255]  # Red color in BGR
        }

        # Get the color value from the color name
        color = color_values.get(color.lower())
        if color is None:
            print(f"Invalid color name: {color}")
            return image

        # Create a 3D mask with the specified color where the original mask is white
        overlay = np.zeros((*image.shape, 4), dtype=np.uint8)  # 4th channel for Alpha
        overlay[image == 255, :3] = color

        return overlay
    
    def overlay_region(self, top_regions, picked_region):
        # Convert top_regions to BGR if it's not already
        if len(top_regions.shape) == 2 or top_regions.shape[2] == 1:
            top_regions_color = cv2.cvtColor(top_regions, cv2.COLOR_GRAY2BGR)
        else:
            top_regions_color = top_regions.copy()

        # Ensure picked_region is binary
        picked_region_binary = (picked_region > 0).astype(np.uint8)

        # Create the kernel for dilation
        kernel = np.ones((6,6), np.uint8)

        # Dilate the picked_region to get the outline
        dilated_region = cv2.dilate(picked_region_binary, kernel, iterations=1)

        # Subtract the picked_region from the dilated_region to get the outline
        outline = dilated_region - picked_region_binary

        # Debug: Check if the outline has any non-zero values
        if np.count_nonzero(outline) == 0:
            print("No outline detected. Check the picked_region array and dilation process.")

        # Apply the outline to the top_regions_color
        top_regions_color[outline == 1] = [0, 255, 0]  # BGR for red in OpenCV

        return top_regions_color
    
    def process_image(self, method='color_segmentation'):
        images = {'original': cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)}

        if method == 'color_segmentation':
            threshold = self.helper.color_segmentation_lab(self.image)
            suffix = '_color_segmentation'
            images['thresholded'] = threshold
            top_regions = self.helper.retain_top_regions_thresholded(threshold)
            eroded_image = self.helper.erode_until_max_area(top_regions)
            dilated_image = self.helper.dilate_image(eroded_image)
            images['top_regions tresholded & dilated'] = dilated_image
            picked_region = self.helper.detect_and_score_regions(dilated_image, self.image)
            overlayed = self.overlay_region(dilated_image, picked_region)
        elif method == 'edge_detection':
            threshold = self.helper.detect_and_invert_edges(self.image)
            suffix = '_edge_detection'
            images['thresholded'] = threshold
            eroded_image = self.helper.erode_until_max_area(threshold)
            dilated_image = self.helper.dilate_image(eroded_image)
            images['eroded & dilated'] = dilated_image
            top_regions = self.helper.retain_top_regions_thresholded(dilated_image)
            picked_region = self.helper.detect_and_score_regions(top_regions, self.image)
            overlayed = self.overlay_region(top_regions, picked_region)

        images['best_candidate_regions'] = overlayed

        closed_image = self.helper.perform_morphological_closing(picked_region)
        # make mask rectangular shape
        rect_mask = self.helper.rectify_mask(picked_region)
        overlayed_rect = self.overlay_region(closed_image, rect_mask)

        images['morphologically_closed'] = overlayed_rect

        masked_image = self.helper.apply_mask(self.image, rect_mask)
        images['rectangular_mask'] = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

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

# Example usage (This code is commented out for execution purposes)
#object_selector = ObjectSelection('dataset/preprocessing_test/2016_07_Arija-Dumbravs.jpg')
#object_selector.run()
