import cv2
import numpy as np
from skimage import measure
from pytesseract import image_to_string

class ObjectSelectionHelper:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def log(self, message):
        if self.verbose:
            print(message)

    def perform_morphological_closing(self, image):
        # Check if image is already a single-channel image
        if len(image.shape) == 2 or image.shape[2] == 1:
            closing = image  # Use the image as is if it is already one channel
            self.log("Image is already in single-channel format.")
        else:
            # Convert the image to grayscale if it has more than one channel
            closing = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.log("Converted image to grayscale.")

        # Constants for the morphological operations
        kernel_size = (3, 3)
        max_iterations = 50
        stop_threshold = 5000
        kernel = np.ones(kernel_size, np.uint8)

        # Perform the closing operation iteratively
        for i in range(max_iterations):
            new_closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel, iterations=i)
            diff = cv2.absdiff(closing, new_closing).sum()

            self.log(f"Iteration {i+1}: Difference = {diff}")

            closing = new_closing

            if diff < stop_threshold and diff != 0:
                break

        self.log(f"Number of iterations for closing: {i+1}")
        return closing

    def retain_top_regions_thresholded(self, image):
        area_threshold = 1000  # Hard-coded threshold for area size
        label_img = measure.label(image)
        props = measure.regionprops(label_img)
        areas = [prop.area for prop in props if prop.area > area_threshold]
        areas.sort(reverse=True)

        if len(areas) >= 2:
            largest_areas = areas[:2]
        elif len(areas) == 1:
            largest_areas = areas[:1]
        else:
            self.log("There are no areas larger than 1000")
            largest_areas = []

        filtered_img = np.zeros_like(image)

        for prop in props:
            if prop.area in largest_areas:
                filtered_img[label_img == prop.label] = 255

        return filtered_img

    def retain_top_regions(self, image):
        # Label the regions in the image
        label_img = measure.label(image)
        # Analyze region properties to get areas
        props = measure.regionprops(label_img)
        # Sort properties by area in descending order
        props_sorted = sorted(props, key=lambda prop: prop.area, reverse=True)

        # Initialize a blank image of the same size as the input
        filtered_img = np.zeros_like(image)

        # Fill in the top two largest regions into the filtered image
        for i, prop in enumerate(props_sorted):
            if i < 2:  # Only the two largest areas are retained
                filtered_img[label_img == prop.label] = 255

        return filtered_img

    def erode_until_max_area(self, image):
        max_area = 20000  # Hard-coded maximum area threshold
        kernel = np.ones((3, 3), np.uint8)
        iterations = 1
        while iterations <= 20:
            eroded = cv2.erode(image, kernel, iterations=iterations)

            # Label connected components in the eroded image
            label_img = measure.label(eroded)
            props = measure.regionprops(label_img)

            # Sort the regions by area
            props_sorted = sorted(props, key=lambda prop: prop.area, reverse=True)

            # Check if the largest area is less than max_area
            if props_sorted[0].area < max_area:
                break

            # Increase the number of iterations for the next erosion
            iterations += 1

        self.log(f"Number of iterations for erosion: {iterations}")
        return eroded

    def color_segmentation_lab(self, image):
        # Convert the image to Lab color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        self.log("Converted image to Lab color space.")

        # Split the Lab image into L, a, and b channels
        lab_planes = cv2.split(lab_image)
        b_channel = lab_planes[2]  # Retrieve only the 'b' channel

        # Thresholding on the 'b' channel
        _, b_thresh = cv2.threshold(b_channel, 128, 255, cv2.THRESH_BINARY_INV)
        self.log("Applied thresholding to the 'b' channel.")

        return b_thresh

    def detect_and_invert_edges(self, image):
        # Apply edge detection with increased thresholds for stronger edges
        refined_edges = cv2.Canny(image, 100, 200)
        self.log("Edge detection applied with thresholds 100 and 200.")

        # Invert the edge image
        inverted_edges = np.invert(refined_edges)
        self.log("Inverted the edge image.")

        return inverted_edges
    
    def apply_mask(self, image, mask):
        masked = cv2.bitwise_and(image, image, mask=mask)
        return masked

    def _calculate_score(self, masked_image):
        # Edge density calculation (lower is better)
        edge_density = np.sum(cv2.Canny(masked_image, 100, 200)) / masked_image.size
        # If edge_density is 0 (which is unlikely), we avoid division by zero
        if edge_density == 0:
            edge_density_score = float('inf')  # Maximal score if no edges are found
        else:
            edge_density_score = 1 / edge_density  # Inverse of edge density
        
        # Text content (higher is better)
        text_content = len(image_to_string(masked_image, config='--psm 3 --oem 3'))
        
        # Mean color calculation (higher is better)
        mean_color = cv2.meanStdDev(cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV))[1][0][0]
        
        # Define weights based on importance of each feature
        edge_density_weight = 0.5  # Increase weight if edge density is a crucial factor
        text_content_weight = 0.1
        mean_color_weight = 0.2
        
        # Calculate final weighted score
        score = (edge_density_score * edge_density_weight +
                text_content * text_content_weight +
                mean_color * mean_color_weight)
        return score

    def detect_and_score_regions(self, closed_image, original_image):
        label_img = measure.label(closed_image)
        props = measure.regionprops(label_img)

        if len(props) != 2:
            if len(props) < 2:
                print("Less than two regions detected, no scoring will be done.")
                return closed_image
            else:
                print("More than two regions detected, only scoring the first two.")
            return closed_image # unexpected number of regions

        masks = []
        scores = []
        for prop in props:
            mask = (label_img == prop.label).astype(np.uint8) * 255
            masked_image = self.apply_mask(original_image, mask)
            score = self._calculate_score(masked_image)
            masks.append(mask)
            scores.append(score)
            self.log(f"Score for region with label {prop.label}: {score}")

        # Find the index of the mask with the highest score
        best_index = np.argmax(scores)
        print(f"Region {best_index+1} has the highest score: {scores[best_index]}")
        print(f"Scores: {scores}")

        # Return the mask with the highest score
        return masks[best_index]

