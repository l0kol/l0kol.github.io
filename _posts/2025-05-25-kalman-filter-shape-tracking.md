---
layout:     post
title:      Kalman filter shape tracking
date:       2025-05-25 20:00:00
summary:    Post abuot implementing a Kalman filter for shape tracking in Python.
author:     Luka Levac
categories: computer-vision python 
tags: kalman-filter shape-tracking
---

👷 - Work in progress, the blog is not completed. I'm writing this in my free time.

This is an overview of a project I did a while ago where the goal was to succesfully track shapes in video with classical computer vision methods. The task was to track the shapes from the moment they entered into the picture frame, to the moment they dissapeared from the frame. Sound trivial, but the challenging part was that the shapes dissaperad randomly for a frame or more, and they intersected eachother. And I wanted to do this completely with traditional CV methods, so no ML/AI help with the tracking. 

Bellow is the video of the tracking algorithm in action. I used a bunch of methods, but the most crucial was the Kalman filter, it was the thing that helped me to trach the shapes even when they dissapeared for a few frames, or went over eachother. 

![Shape tracking](/images/tracked-shapes.gif)

<div style="font-size: 0.7em;">
{% highlight python %}
class Shape:
    def __init__(self, shape_id, center, area, perimeter, bbox, color, shape_type):
        self.shape_id = shape_id
        self.center = center
        self.area = area
        self.perimeter = perimeter
        self.path = [center] 
        self.missing_frames = 0
        self.bbox = bbox
        self.shape_color = color
        self.type = shape_type
        self.trace_color = None

        # (x, y, dx, dy) for 4 states, (x, y) for 2 measurements
        self.kf = cv2.KalmanFilter(4, 2)  
        # Transition matrix, linear motion model
        self.kf.transitionMatrix = np.array([[1, 0, 0.1, 0], [0, 1, 0, 1], [0, 0, 0.1, 0], [0, 0, 0, 1]], dtype=np.float32)
        # Directly observe the position
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        # Uncertainty in the process
        self.kf.processNoiseCov = np.array([[1e-1, 0, 0, 0], [0, 1e-1, 0, 0], [0, 0, 1e-2, 0], [0, 0, 0, 1e-2]], dtype=np.float32)
        # Uncertainty in the measurements
        self.kf.measurementNoiseCov = np.array([[1e-1, 0], [0, 1e-1]], dtype=np.float32)

        # Initial update with the starting center position
        self.kf.predict()
        self.kf.correct(np.array([center[0], center[1]], dtype=np.float32))

    def update(self, new_center):
        """Update the Kalman filter with a new center."""
        self.kf.predict()
        self.kf.correct(np.array([new_center[0], new_center[1]], dtype=np.float32))
        self.center = new_center
        self.path.append(new_center)
        self.missing_frames = 0  # Reset missing frames

    def predict(self):
        """Predict the next position of the shape using Kalman filter."""
        self.kf.predict()
        predicted_center = (int(self.kf.statePost[0, 0]), int(self.kf.statePost[1, 0]))
        self.center = predicted_center


    def get_bbox(self):
        """Generate the bounding box based on the current position."""
        x, y = self.center
        w = self.bbox[2]  # Width
        h = self.bbox[3]
        return x - w // 2, y - h // 2, w, h
{% endhighlight %}
</div>


<div style="font-size: 0.7em;">
{% highlight python %}
class ShapeTracker:
    def __init__(self, frameShape):
        self.frameHeight:int = frameShape[0]
        self.frameWidth:int = frameShape[1]
        self.tracked_shapes:Shape = {}
        self.tracked_colors:list = {}
        self.next_shape_id:int = 0    

    @staticmethod
    def _random_color()->tuple:
        return tuple(random.randint(0, 255) for _ in range(3))

    @staticmethod
    def _calculate_hsv_bounds(color, tolerance):
        color = color.astype(np.uint16)
        lower_hue = ((int(color[0]) - tolerance) % 180) % 180
        upper_hue = (color[0] + tolerance) % 180
        lower_saturation = max(color[1] - tolerance, 0)
        upper_saturation = min(color[1] + tolerance, 255)
        lower_value = max(color[2] - tolerance, 0)
        upper_value = min(color[2] + tolerance, 255)
        lower_bound = np.array([lower_hue, lower_saturation, lower_value], dtype=np.uint8)
        upper_bound = np.array([upper_hue, upper_saturation, upper_value], dtype=np.uint8)

        return lower_bound, upper_bound
    
    def _threshold_colors(self, frame, frame_gray)->list:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        contours = []
        for color in self.tracked_colors.values():
            
            lower, upper = color['bounds']
            mask = cv2.inRange(hsv_frame, lower, upper)

            # Dialate the mask to remove noise around the shape
            kernel = np.ones((5, 5), np.uint8)
            cleaned_mask = cv2.dilate(mask, kernel, iterations=1)
            # Find contours
            _contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(_contours)

            # Subtract the mask from the gray_img, only remove the detected shapes
            frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=cv2.bitwise_not(cleaned_mask))

        return frame_gray, contours
    
    def _threshold_grayscale(self, frame_gray)->list:
        blurred_img = cv2.GaussianBlur(frame_gray, (5, 5), 0)

        # Threshold the image
        binary_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned_binary = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

        # https://stackoverflow.com/questions/66924925/how-can-i-remove-double-lines-detected-along-the-edges
        cv2.floodFill(cleaned_binary, mask=None, seedPoint=(int(0), int(0)), newVal=(255))

        cv2.floodFill(cleaned_binary, mask=None, seedPoint=(int(0), int(0)), newVal=(0))

        # Find contours
        contours, _ = cv2.findContours(cleaned_binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

        return contours

    def detect_shapes(self, frame) -> list:
        detected_shapes = []
        contours = []

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(self.tracked_colors) > 0:
            # If we have detected shapes, threshold the image based on the shapes colors
            frame_gray, contours = self._threshold_colors(frame, frame_gray)

        contours.extend(self._threshold_grayscale(frame_gray))

        for contour in contours:
            # Check if the contour area is big enough
            if cv2.contourArea(contour) < 1700:
                continue

            # Compute moments for the contour
            M = cv2.moments(contour)

            if (M["m00"] == 0):
                continue

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            area = M['m00']
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate circularity
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

            # Classify the shape
            if len(cv2.approxPolyDP(contour, 0.02 * perimeter, True)) == 4:
                shape_type = 'SQUARE'  # It's a square
            elif circularity > 0.8:
                shape_type = 'CIRCLE'  # It's a circle
            else:
                shape_type = 'OTRH'  # All other shapes

            # Get color of the shape, take center pixel
            cx = x + w // 2
            cy = y + h // 2
            color = frame[cy, cx]
            # Get the HSV color
            color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0] 
                        
            detected_shapes.append({
                "bbox": (x, y, w, h), 
                "contour": contour, 
                "center": (cx, cy),
                "area": area,
                "perimeter": perimeter,
                "color_hsv": color_hsv,
                "shape_type": shape_type
                })

        return detected_shapes

    def track_shapes(self, detected_shapes):
        new_tracked_shapes = {}
        unmatched_tracked_shapes:Shape = self.tracked_shapes.copy()

        for shape in detected_shapes:
            cx, cy = shape["center"]
            matched_id = None

            # Try to match this shape with existing tracked shapes
            for shape_id, shape_obj in self.tracked_shapes.items():
                # Calculate the Euclidean distance between the centers of the shapes
                dist = np.sqrt((cx - shape_obj.center[0]) ** 2 + (cy - shape_obj.center[1]) ** 2)

                # Check color difference in hsv space
                tracked_color = self.tracked_colors[shape_id]["hsv"].astype(np.int16)
                color_diff = np.linalg.norm(shape["color_hsv"].astype(np.int16) - tracked_color)
                
                if dist < 300 and color_diff < 10:
                    matched_id = shape_id
                    break

            if matched_id is not None:

                # Update the Kalman filter and tracked shape
                shape_obj = self.tracked_shapes[matched_id]
                shape_obj.update((cx, cy))
                shape_obj.perimeter = shape["perimeter"]
                shape_obj.bbox = list(shape["bbox"])
                new_tracked_shapes[matched_id] = shape_obj

                del self.tracked_shapes[matched_id]
                del unmatched_tracked_shapes[matched_id]
            else:
                # Assign new ID and create a new shape object with Kalman filter
                self.next_shape_id += 1
                new_shape = Shape(self.next_shape_id, 
                                  (cx, cy), shape["area"], 
                                  shape["perimeter"], 
                                  list(shape["bbox"]),
                                  shape["color_hsv"],
                                  shape["shape_type"])
                new_shape.trace_color = self._random_color()
                new_tracked_shapes[self.next_shape_id] = new_shape

                # Store HSV color and precomputed bounds
                lower_bound, upper_bound = self._calculate_hsv_bounds(shape["color_hsv"], 4)
                self.tracked_colors[self.next_shape_id] = {
                    "hsv": shape["color_hsv"],
                    "bounds": (lower_bound, upper_bound)
                }

        # Handle missing shapes (objects not detected in the current frame)
        for shape_id, shape_obj in unmatched_tracked_shapes.items():
                
                # If the shape is not at the edge of the frame, allow more missing frames
                max_missing_frames = 5 if self._is_at_edge(shape_obj.bbox) else 15

                if shape_obj.missing_frames < max_missing_frames: #
                    if len(shape_obj.path) > 5: # Kalman filter need a few frames to predict the position, otherwise it will be too inaccurate
                        shape_obj.predict()  # Get predicted position using Kalman filter
                    else:
                        shape_obj.update(shape_obj.center) # Update the shape with the same position if we don't have enough frames
                    shape_obj.missing_frames += 1
                    new_tracked_shapes[shape_id] = shape_obj

                else:
                    # Remove the shape if it's missing for too many frames
                    del self.tracked_colors[shape_id]

        self.tracked_shapes = new_tracked_shapes

    def draw_paths(self, frame):
        for _, shape in self.tracked_shapes.items():
            path = shape.path
            color = tuple(map(int, shape.trace_color))

            for i in range(1, len(path)):
                cv2.line(frame, path[i - 1], path[i], color, 4)

            cx, cy = shape.center
            cv2.circle(frame, (cx, cy), 5, color, -1)

    def draw_bounding_boxes(self, frame) :
        for shape_id, shape in self.tracked_shapes.items():
            
            if shape.missing_frames == 0:
                x, y, w, h = shape.bbox 
            else:
                x, y, w, h = shape.get_bbox()
                
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"[{shape_id}] {shape.type}", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)


    def _is_at_edge(self, bbox, edge_margin=3)->bool:
        x, y, w, h = bbox
        return (x <= edge_margin or y <= edge_margin or x + w >= self.frameWidth - edge_margin or y + h >= self.frameHeight - edge_margin)
{% endhighlight %}
</div>

[^1]: https://en.wikipedia.org/wiki/Kalman_filter