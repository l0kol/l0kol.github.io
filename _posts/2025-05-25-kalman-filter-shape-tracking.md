---
layout:     post
title:      Kalman filter shape tracking
date:       2025-05-25 20:00:00
summary:    Post about implementing a Kalman filter for shape tracking in Python.
author:     Luka Levac
categories: computer-vision python 
tags: kalman-filter shape-tracking
---

👷 - Work in progress, the blog is not completed. I'm writing this in my free time.

This is an overview of a project I did a while ago, where the goal was to succesfully track shapes in a video using only classical computer vision methods. 
The task was to track the shapes from the moment they entered into the picture frame, to the moment they dissapeared from the frame. Sounds trivial, but there were a couple of tricky parts. 
The shapes dissaperad randomly for a (couple of) frame(s), the shapes also moved over eachother, sometimes completely abstracting one another. 
All of this would be trivial using ML methods, but with traditional CV, we have to use a bunch of algorithms to acheave this. It's also not that complicated, we just have more code
to stick together to acheave the same. But the bonus point is the speedm this algo works in real time (much faster than the frame rate of teh video). I will now walk you over the procces. 

But first, bellow is the video of the finished result.

![Shape tracking](/images/tracked-shapes.gif)

## Overview of the main code

Before we start going through the methods I used, let's check the flow of the main code. 
1. First we have to detect the shapes in the image, 
2. then we "track" those shapes (save their current position, match them with their previous id etc.),
3. and after that we visualize our tracking info on the image (bounding boxes + a line path).

This post will also be broken into 3 sub chapters:
- [Shape Detecting](#shape-detecting)
- [Shape Tracking](#shape-tracking)
- [Visualization](#visualization)

Main code:

<div style="font-size: 0.7em;">
{% highlight python %}
if __name__ == "__main__":
    cap = cv2.VideoCapture("video.mp4")
    frameShape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    tracker = ShapeTracker(frameShape)

    allTimes = np.empty(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # Detect shapes
        detected_shapes = tracker.detect_shapes(frame)

        # Track shapes
        tracker.track_shapes(detected_shapes)
        
        # Draw the bounding boxes
        tracker.draw_bounding_boxes(frame)

        # Draw the tracked paths
        tracker.draw_paths(frame)

        # Show the frame
        cv2.imshow("Shape Tracking", frame)

        # Exit if 'ESC' is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

{% endhighlight %}
</div>

## <a id="shape-detecting"></a>Shape Detecting
The first method called is `detect_shapes(frame)`. Let's see what's inside.

#### Contour detection
First thing we do is contour detection.
   
<div style="font-size: 0.7em;">
{% highlight python %}
detected_shapes = []
contours = []

frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

if len(self.tracked_colors) > 0:
    # If we have detected shapes, threshold the image based on the shapes colors
    frame_gray, contours = self._threshold_colors(frame, frame_gray)

contours.extend(self._threshold_grayscale(frame_gray))    
{% endhighlight %}
</div>

On the first run, we won't have any prior contours saved in the attribute self.tracked_colors, so the _threshold_grayscale() function will be triggered.

<div style="font-size: 0.7em;">
{% highlight python %}
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

{% endhighlight %}
</div>

Before doing the contour searching, we have to prepare the frame. First we run it through a gaussian blur using a 5x5 kernel. Gaussian blurr removes some of the noise in the image. This is also reccomended in opencv docs. But the results without the blurr were also good in my case as the scene was quite simple. 

<div style="display: flex; justify-content: space-between; text-align: center;">
  <div style="margin: 0 10px;">
    <img src="/images/Frame_gray.png" width="450" /> 
    <p style="margin-top: 8px;"><em>Grayscale Version</em></p>
  </div>
  <div style="margin: 0 10px;">
    <img src="/images/Blurred_frame.png" width="450" />
    <p style="margin-top: 8px;"><em>Blurred Effect</em></p>
  </div>
</div>

Blurred frame is then inputed into the threshodling function. But as you can see in the image bellow, the result is not perfect to say the least. 

<div style="display: flex; justify-content: space-between; text-align: center;">
  <div style="margin: 0 10px;">
    <img src="/images/Threshold.png" width="600" /> 
    <p style="margin-top: 8px;"><em> Binary image</em></p>
  </div>
</div>

## <a id="shape-tracking"></a>Shape Tracking
### Kalman Filter
Content for shape tracking section...

## <a id="visualization"></a>Visualization Content for visualization section...
[^1]: https://en.wikipedia.org/wiki/Kalman_filter
