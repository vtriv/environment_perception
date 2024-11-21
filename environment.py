import cv2
import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial import distance
import open3d as o3d

class Environment:
    def __init__(self, video_path: str):
        """
        Initialize the Environment with the given .mov video file.

        Args:
            video_path (str): Path to the input video.
        """
        self.video_path = video_path
        self.frames = []  # List of video frames
        self.map_representation = None  # Spatial map representation
        self.objects = []  # List of detected objects
        self.keypoints = []  # Keypoints for spatial mapping (SLAM)
        self.dimensions = {}  # Object/location dimensions

        self.net = cv2.dnn.readNet("yolov3-openimages.cfg", "yolov3-openimages.weights")
        with open("openimages.names", "r") as f:
            self.class_labels = [line.strip() for line in f.readlines()]

        self._process_video()

    def _process_video(self):
        """
        Process the video to extract relevant spatial and semantic data.
        """
        cap = cv2.VideoCapture(self.video_path)

        # annotated video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 format
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))  # Original video frame rate
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter("annotated_output.mp4", fourcc, self.fps, (self.frame_width, self.frame_height))

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Frame rate: {self.fps}")
        print(f"Total frames: {self.total_frames}")

        frame_count = 0
        n = round(self.fps / 15)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1

            if frame_count % n == 0:
                # Add frame processing logic here (e.g., object detection, SLAM)
                self._process_frame(frame)
                self.frames.append(frame)
                out.write(frame)
                print("len frames: ", len(self.frames))  # Placeholder: Display frame count 
                print("frame count: ", frame_count)
                print("num objects: ", len(self.objects))
                # print(len(self.keypoints))
                print('---')

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        self._generate_map()

    def _process_frame(self, frame):
        """
        Process a single frame to extract relevant data.

        Args:
            frame (np.ndarray): Input frame.
        """
        # object detection
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False) # preprocess frame
        self.net.setInput(blob)

        layer_names = self.net.getUnconnectedOutLayersNames()  # Get YOLO output layers
        detections = self.net.forward(layer_names)

        height, width = frame.shape[:2]
        detected_objects = []

        # Process detections
        for detection_layer in detections:
            for detection in detection_layer:
                scores = detection[5:]  # Class probabilities
                class_id = np.argmax(scores)  # Get the class ID with the highest score
                confidence = scores[class_id]  # Confidence of the best class

                if confidence > 0.3:  # Confidence threshold
                    # Scale bounding box to original frame size
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    box_width = int(detection[2] * width)
                    box_height = int(detection[3] * height)

                    # Calculate top-left corner of the bounding box
                    x = int(center_x - box_width / 2)
                    y = int(center_y - box_height / 2)

                    # Add the object and its bounding box
                    detected_objects.append({
                        "class_id": class_id,
                        "label": self.class_labels[class_id],
                        "confidence": confidence,
                        "bbox": (x, y, box_width, box_height),
                        "center": (center_x, center_y)
                    })

                    # # Optionally, draw the bounding box on the frame
                    # Draw the bounding box on the frame
                    cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (0, 255, 0), 2)
                    cv2.putText(frame, f"{self.class_labels[class_id]}: {confidence:.2f}",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        
        # self.objects.extend(detected_objects)
        self._update_tracked_objects(detected_objects)

        # preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # extract keypoints
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if keypoints:
            self.keypoints.append((keypoints, descriptors))

        # # Update spatial map after processing the frame
        # map_width = 800
        # map_height = 600
        # spatial_map = np.zeros((map_height, map_width, 3), dtype=np.uint8)

        # for obj in self.tracked_objects:
        #     x, y, w, h = obj["bbox"]
        #     label = obj["label"]

        #     center_x = int((x + w / 2) / frame.shape[1] * map_width)
        #     center_y = int((y + h / 2) / frame.shape[0] * map_height)

        #     cv2.circle(spatial_map, (center_x, center_y), 5, (0, 0, 255), -1)
        #     cv2.putText(spatial_map, label, (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # # Display spatial map dynamically
        # cv2.imshow("Spatial Map", spatial_map)
        # cv2.waitKey(1)  # Adjust delay as needed for smoother updates

        # # # Optionally save the map periodically (e.g., every 10 frames)
        # # if len(self.frames) % 10 == 0:
        # #     cv2.imwrite(f"spatial_map_frame_{len(self.frames)}.png", spatial_map)

        # # Update class attribute
        # self.map_representation = spatial_map

    # def _update_tracked_objects(self, detected_objects):
    #     """
    #     Update the list of tracked objects with the latest detections.

    #     Args:
    #         detected_objects (List[Dict]): List of detected objects in the current frame.
    #     """
    #     if not hasattr(self, "tracked_objects"):
    #         self.tracked_objects = []
        
    #     updated_tracked_objects = []

    #     for detected in detected_objects:
    #         matched = False
    #         for tracked in self.tracked_objects:
    #             # Compare by label and spatial proximity
    #             if (tracked["label"] == detected["label"] and distance.euclidian(tracked["center"], detected["center"]) < 50):
    #                 tracked["center"] = detected["center"]
    #                 tracked["bbox"] = detected["bbox"]
    #                 tracked["confidence"] = max(tracked["confidence"], detected["confidence"])
    #                 updated_tracked_objects.append(detected)
    #                 matched = True
    #                 break

    #             if not matched:
    #                 detected["id"] = len(self.tracked_objects) + 1
    #                 updated_tracked_objects.append(detected)
            
    #     self.tracked_objects = updated_tracked_objects
    #     self.objects = self.tracked_objects

    def _generate_map(self):
        """
        Generate a spatial map representation from video frames.
        """
        if len(self.keypoints) < 2:
            print("Not enough frames with keypoints for 3D mapping.")
            return

        # Placeholder: Camera matrix (intrinsic parameters)
        camera_matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0,  0,  1]])  # Replace fx, fy, cx, cy with calibrated values

        # Iterate through consecutive frames to match features
        point_cloud = []
        for i in range(len(self.keypoints) - 1):
            keypoints1 = self.keypoints[i]
            keypoints2 = self.keypoints[i + 1]

            # Match features between consecutive frames
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(keypoints1[1], keypoints2[1])  # Descriptors for the two frames
            matches = sorted(matches, key=lambda x: x.distance)

            # Extract matched keypoint coordinates
            points1 = np.float32([keypoints1[0][m.queryIdx].pt for m in matches])
            points2 = np.float32([keypoints2[0][m.trainIdx].pt for m in matches])

            # Compute the essential matrix
            E, mask = cv2.findEssentialMat(points1, points2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)

            # Recover pose (rotation and translation)
            _, R, t, _ = cv2.recoverPose(E, points1, points2, camera_matrix)

            # Triangulate points
            proj_matrix1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Camera 1 at origin
            proj_matrix2 = np.hstack((R, t))  # Camera 2 relative to Camera 1

            points_4d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, points1.T, points2.T)
            points_3d = points_4d[:3] / points_4d[3]  # Convert homogeneous to 3D

            point_cloud.append(points_3d.T)  # Add to the point cloud

        # Combine all 3D points
        point_cloud = np.vstack(point_cloud)

        # Visualize or save the 3D point cloud
        self._visualize_point_cloud(point_cloud)

        # # Placeholder: Implement mapping logic (e.g., SLAM, feature extraction)
        # if not self.frames:
        #     print("No frames processed to generate map from.")
        
        # # Define map dimensions
        # map_width = 800
        # map_height = 600
        # spatial_map = np.zeros((map_height, map_width, 3), dtype=np.uint8)

        # # Consolidate all tracked objects
        # for obj in self.tracked_objects:
        #     x, y, w, h = obj["bbox"]
        #     label = obj["label"]

        #     # Map object center to spatial map coordinates
        #     center_x = int((x + w / 2) / self.frames[0].shape[1] * map_width)
        #     center_y = int((y + h / 2) / self.frames[0].shape[0] * map_height)

        #     # Add object representation to the map
        #     cv2.circle(spatial_map, (center_x, center_y), 5, (0, 0, 255), -1)
        #     cv2.putText(spatial_map, label, (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # # Save the final map
        # cv2.imwrite("final_spatial_map.png", spatial_map)
        # print("Final spatial map generated. Saved as 'final_spatial_map.png'.")

        # self.map_representation = spatial_map

    def _visualize_point_cloud(self, points):
        """
        Visualize the 3D point cloud using Open3D.

        Args:
            points (np.ndarray): Array of 3D points.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        o3d.visualization.draw_geometries([pcd])

    def get_map(self) -> str:
        """
        Get the spatial map representation.

        Returns:
            str: Placeholder map representation.
        """
        return self.map_representation

    def get_objects(self) -> List[str]:
        """
        Get a list of objects detected in the environment.

        Returns:
            List[str]: Names of detected objects.
        """
        return self.objects

    def get_dimensions(self, obj: str) -> Optional[Tuple[float, float]]:
        """
        Get the dimensions of a specific object.

        Args:
            obj (str): Object name.

        Returns:
            Optional[Tuple[float, float]]: Width and height of the object, if available.
        """
        return self.dimensions.get(obj)

    def get_location(self, obj: str) -> Optional[Tuple[float, float]]:
        """
        Get the location of a specific object in the environment.

        Args:
            obj (str): Object name.

        Returns:
            Optional[Tuple[float, float]]: Coordinates of the object, if available.
        """
        # Placeholder: Add location retrieval logic
        return None

    def query_path(self, start: str, end: str) -> List[Tuple[float, float]]:
        """
        Find the path between two locations in the environment.

        Args:
            start (str): Starting location.
            end (str): Ending location.

        Returns:
            List[Tuple[float, float]]: Path as a list of coordinates.
        """
        # Placeholder: Implement pathfinding logic
        return []

# Example usage:
if __name__ == "__main__":
    env = Environment("shortened.mov")
    print(env.get_map())  # Placeholder output
    # print(env.get_objects())  # Placeholder output
    objects = env.get_objects()
    for obj in objects:
        print(obj['label'])
        print(obj['confidence'])
        print('---')
    print(env.get_dimensions(next(iter( env.get_objects().items() ))))  # Placeholder output
    print(env.query_path("Bruce Lee's office", "bathroom"))  # Placeholder output
