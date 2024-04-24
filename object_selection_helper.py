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
        max_iterations = 30
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
        if mask is None:
            print("Mask is None.")
            return image
        elif type(mask) != np.ndarray:
            print(f"Mask is of type {type(mask)}, not numpy.ndarray.")
            return image
        elif mask.shape != image.shape[:2]:
            print(f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}.")
            return image
        else:
            masked = cv2.bitwise_and(image, image, mask=mask)
            return masked

    def calculate_average_saturation(self, image):
        # Convert the image from RGB to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Extract the Saturation channel
        saturation_channel = hsv_image[:, :, 1]
        
        # Calculate the average saturation
        average_saturation = np.mean(saturation_channel)
        return average_saturation
    
    def calculate_mask_area(self, mask):
        # Count the number of white pixels (255)
        area = np.count_nonzero(mask == 255)
        return area
    
    def calculate_centroid_distance_score(self, mask):
        # Calculate image moments
        M = cv2.moments(mask)
        
        # Check if there are any white pixels (area should not be zero)
        if M["m00"] == 0:
            print("No white pixels found in mask.")
            return np.sqrt(mask.shape[0]**2 + mask.shape[1]**2)  # Maximum distance from the center
        
        # Calculate the centroid from moments
        centroid_x = M["m10"] / M["m00"]
        centroid_y = M["m01"] / M["m00"]
        
        # Calculate the center of the image
        center_x = mask.shape[1] / 2
        center_y = mask.shape[0] / 2
        
        # Calculate and return the Euclidean distance from the centroid to the center
        distance = np.sqrt((centroid_y - center_y)**2 + (centroid_x - center_x)**2)
        return distance
    
    def calculate_combined_score(self, masked_image, mask, max_saturation=255, max_area=None, max_distance=None):
        if max_area is None:
            max_area = masked_image.shape[0] * masked_image.shape[1]  # height * width
        if max_distance is None:
            max_distance = np.sqrt(masked_image.shape[0]**2 + masked_image.shape[1]**2)  # Diagonal

        average_saturation_score = self.calculate_average_saturation(masked_image)
        area_score = self.calculate_mask_area(mask)
        centroid_distance_score = self.calculate_centroid_distance_score(mask)

        # Normalize scores
        normalized_saturation = 1 - (average_saturation_score / max_saturation)  # Inverted because lower is better
        normalized_area = area_score / max_area
        normalized_centroid_distance = 1 - (centroid_distance_score / max_distance)  # Inverted because lower is better

        # print(f"Normalized scores: Saturation={normalized_saturation}, Area={normalized_area}, Centroid Distance={normalized_centroid_distance}")

        # Weights (can be adjusted)
        w1, w2, w3 = 1, 2, 2

        # print(f"Weighted scores: Saturation={w1*normalized_saturation}, Area={w2*normalized_area}, Centroid Distance={w3*normalized_centroid_distance}")

        # Calculate combined score
        combined_score = (w1 * normalized_saturation) + \
                        (w2 * normalized_area) + \
                        (w3 * normalized_centroid_distance)
        

        return combined_score

    def detect_and_score_regions(self, closed_image, original_image):
        label_img = measure.label(closed_image)
        props = measure.regionprops(label_img)

        if len(props) != 2:
            print(f"Unexpected number of regions detected: {len(props)}")
            return closed_image

        masks = []
        scores = []
        for prop in props:
            mask = (label_img == prop.label).astype(np.uint8) * 255
            masked_image = self.apply_mask(original_image, mask)
            combined_score = self.calculate_combined_score(masked_image, mask)
            
            scores.append(combined_score)
            masks.append(mask)

            print(f"Combined dominance score for region with label {prop.label}: {combined_score}")


        # Checking if the scores are close; if so, consider both regions
        if False: #abs(scores[0] - scores[1]) < 0.1:  # Threshold to keep both segments if close
            print(f"Both regions have similar gray dominance. Scores: {scores}")
            return closed_image
        else:
            # Return the mask of the region with higher gray dominance
            best_index = np.argmax(scores)
            print(f"Region {best_index+1} has the higher gray dominance score: {scores[best_index]}")
            return masks[best_index]

    def rectify_mask(self, mask):
        _, thresh_img = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Compute the bounding rectangle for the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        #bounding_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
        return mask