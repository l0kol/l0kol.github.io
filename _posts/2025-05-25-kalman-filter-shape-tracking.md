---
layout:     post
title:      Kalman filter shape tracking
date:       2025-05-25 20:00:00
summary:    Post abuot implementing a Kalman filter for shape tracking in Python.
author:     Luka Levac
categories: computer-vision python 
tags: kalman-filter shape-tracking
---

Post in progress.

Whole code block bellow, just a placeholder for now. Will break it down into smaller parts later.


```python
class ShapeTrackerv1:
    def __init__(self) -> None:
        self.tracked_shapes = {}  
        self.next_shape_id = 0
        self.shape_paths = []
        self.shape_colors = {}
        self.shape_history = []

    def random_color(self):
        """Generate a random RGB color."""
        return tuple(random.randint(0, 255) for _ in range(3))

    def detect_shapes(self, frame, draw_contours = True) -> list:
        detected_shapes = []
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

        # Threshold the image
        # _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # edges = cv2.Canny(blurred_img, 0, 50)
        binary_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned_binary = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

        # https://stackoverflow.com/questions/66924925/how-can-i-remove-double-lines-detected-along-the-edges
        cv2.floodFill(cleaned_binary, mask=None, seedPoint=(int(0), int(0)), newVal=(255))

        cv2.floodFill(cleaned_binary, mask=None, seedPoint=(int(0), int(0)), newVal=(0))

        # Find contours
        contours, _ = cv2.findContours(cleaned_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            hull = cv2.convexHull(contour)

            # cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

            x, y, w, h = cv2.boundingRect(hull)

            area = cv2.contourArea(hull)

            aspect_ratio = float(w) / h if h != 0 else 0

            # Get the perimeter of the contour
            perimeter = cv2.arcLength(hull, True)

            # Approximate the polygon
            approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)

            # Get the bounding box
            x, y, w, h = cv2.boundingRect(approx)

            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            detected_shapes.append({
                "bbox": (x, y, w, h), 
                "contour": approx, 
                "center": (x + w // 2, y + h // 2),
                "area": area,
                "aspect_ratio": aspect_ratio,
                "perimeter": perimeter
                })

        return detected_shapes
    

    def track_shapes(self, detected_shapes) -> None:
        """Track shapes across frames."""
        new_tracked_shapes = {}
        unmatched_tracked_shapes = copy.deepcopy(self.tracked_shapes)  # Deep copy of tracked shapes

        for shape in detected_shapes:
            cx, cy = shape["center"]
            matched_id = None
            for shape_id, data in self.tracked_shapes.items():
                prev_cx, prev_cy = data["center"]
                dist = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)

                area_diff = abs(shape["area"] - data["area"])
                aspect_ratio_diff = abs(shape["aspect_ratio"] - data["aspect_ratio"])

                if dist < 50 and area_diff < 800 and aspect_ratio_diff < 0.2:
                    matched_id = shape_id
                    break

            if matched_id is not None:
                # Update matched shape
                new_tracked_shapes[matched_id] = {
                    "bbox": shape["bbox"],
                    "center": (cx, cy),
                    "path": self.tracked_shapes[matched_id]["path"] + [(cx, cy)],
                    "missing_frames": 0,  # Reset missing count,
                    "area": shape["area"],
                    "aspect_ratio": shape["aspect_ratio"],
                    "perimeter": shape["perimeter"]
                }
                del unmatched_tracked_shapes[matched_id]
            else:
                # Assign a new ID to the shape
                self.next_shape_id += 1
                new_tracked_shapes[self.next_shape_id] = {
                    "bbox": shape["bbox"],
                    "center": (cx, cy),
                    "path": [(cx, cy)],
                    "missing_frames": 0,  # New shape starts with 0 missing frames
                    "area": shape["area"],
                    "aspect_ratio": shape["aspect_ratio"],
                    "perimeter": shape["perimeter"]
                }
                self.shape_colors[self.next_shape_id] = self.random_color()  # Assign unique color

        # Handle unmatched shapes (shapes that disappeared)
        for shape_id, data in unmatched_tracked_shapes.items():
            if data["missing_frames"] < 3:  # Allow up to 3 missing frames
                new_tracked_shapes[shape_id] = {
                    "bbox": data["bbox"],
                    "center": data["center"],
                    "path": data["path"],  # Retain the path
                    "missing_frames": data["missing_frames"] + 1,
                    "area": data["area"],
                    "aspect_ratio": data["aspect_ratio"],
                    "perimeter": data["perimeter"]
                }
            # else:
            #     # Debug: Print shapes being removed
            #     # print(f"Shape ID {shape_id} removed after missing {data['missing_frames']} frames.")

        # Update tracked shapes
        self.tracked_shapes = new_tracked_shapes

        self.shape_history.append(detected_shapes)
        if len(self.shape_history) > 3:
            self.shape_history.pop(0)


    def draw_paths(self, frame) -> None:
        for shape_id, data in self.tracked_shapes.items():
            path = data["path"]
            color = self.shape_colors[shape_id]

            for i in range(1, len(path)):
                cv2.line(frame, path[i - 1], path[i], color, 2)

            # Draw the current position as a circle
            cx, cy = data["center"]
            cv2.circle(frame, (cx, cy), 5, color, -1)

    def draw_bounding_boxes(self, frame) -> None:
        for shape_id, data in self.tracked_shapes.items():
            color = self.shape_colors[shape_id]  # Get the unique color for this shape
            
            if data["missing_frames"] == 0:
                # Normal bounding box
                x, y, w, h = data["bbox"] # Use the latest center
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                if len(data["path"]) > 1:
                    # Dashed bounding box
                    last_three_centers = np.array(data["path"][-3:])
                    predicted_center = np.mean(last_three_centers, axis=0).astype(int)

                else:
                    predicted_center = data["path"][-1]

                cx, cy = predicted_center
                w, h = data["bbox"][2:]

                x = cx - w // 2
                y = cy - h // 2

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
```
