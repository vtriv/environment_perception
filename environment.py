import cv2
import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial import distance
# import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import depth_pro

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
        self.point_cloud = []

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

        ret, prev_frame = cap.read()
        # Convert first frame to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # point_cloud = []

        while cap.isOpened():
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            if frame_count < 1000:
                # Convert current frame to grayscale
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

                # Match features and estimate pose
                points1, points2, _, _ = self._match_features(prev_gray, curr_gray)
                # Update for next iteration
                prev_gray = curr_gray
                
                R, t = self._estimate_pose(points1, points2)
                if R is None or t is None:
                    print("Skipping frame due to invalid pose estimation.")
                    continue  # Skip the current frame
                # Triangulate points
                points_3d = self._triangulate_points(points1, points2, R, t)
                self.point_cloud.append(points_3d)

                frame_count += 1

                if frame_count % n == 0 and frame_count < self.total_frames/3:
                    # Add frame processing logic here (e.g., object detection, SLAM)
                    self._process_frame(curr_frame)
                    self.frames.append(curr_frame)
                    out.write(curr_frame)
                    print("len frames: ", len(self.frames))  # Placeholder: Display frame count 
                    print("frame count: ", frame_count)
                    print("num objects: ", len(self.objects))
                    print('---')

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        # self._generate_map()
        self._visualize_point_cloud()

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

                if confidence > 0.25:  # Confidence threshold
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

        
        self.objects.extend(detected_objects)

    def _match_features(self, frame1, frame2):
        """
        Match ORB features between two frames.

        Args:
            frame1 (np.ndarray): First frame.
            frame2 (np.ndarray): Second frame.

        Returns:
            Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], np.ndarray, np.ndarray]: 
            Keypoints and descriptors for both frames.
        """
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(frame1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(frame2, None)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        return points1, points2, keypoints1, keypoints2

    def _estimate_pose(self, points1, points2):
        """
        Estimate pose between two frames using the essential matrix.

        Args:
            points1 (np.ndarray): Matched keypoints from the first frame.
            points2 (np.ndarray): Matched keypoints from the second frame.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Rotation and translation matrices.
        """
        camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

        # Compute essential matrix
        E, mask = cv2.findEssentialMat(points1, points2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        # Validate essential matrix
        if E is None or E.shape != (3, 3):
            print("Invalid essential matrix:", E)
            return None, None

        # Filter inliers
        inliers1 = points1[mask.ravel() == 1]
        inliers2 = points2[mask.ravel() == 1]

        if len(inliers1) < 8:
            print("Not enough inliers for pose estimation.")
            return None, None

        # Recover pose
        _, R, t, _ = cv2.recoverPose(E, inliers1, inliers2, camera_matrix)
        return R, t

    
    def _triangulate_points(self, points1, points2, R, t):
        """
        Triangulate 3D points from matched 2D points and pose.

        Args:
            points1 (np.ndarray): Matched keypoints from the first frame.
            points2 (np.ndarray): Matched keypoints from the second frame.
            R (np.ndarray): Rotation matrix.
            t (np.ndarray): Translation vector.

        Returns:
            np.ndarray: 3D points.
        """
        # Placeholder camera matrix
        camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

        # Projection matrices
        proj_matrix1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        proj_matrix2 = np.hstack((R, t))

        # Convert to projection matrices in pixel space
        proj_matrix1 = camera_matrix @ proj_matrix1
        proj_matrix2 = camera_matrix @ proj_matrix2

        points_4d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, points1.T, points2.T)
        points_3d = points_4d[:3] / points_4d[3]  # Convert homogeneous to 3D
        return points_3d.T



    def _generate_map(self):
        """
        Generate a spatial map representation from video frames.
        """
        if len(self.keypoints) < 2:
            print("Not enough frames with keypoints for 3D mapping.")
            return

        fx = 26
        fy = fx
        cx = 0
        cy = 0

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

    def _visualize_point_cloud(self):
        """
        Visualize the 3D point cloud using Open3D.

        Args:
            points (np.ndarray): Array of 3D points.
        """
        points = np.vstack(self.point_cloud)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        ax.scatter(x, y, z, c='b', marker='o', s=1)
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    def get_map(self) -> str:
        """
        Get the spatial map representation.

        Returns:
            str: Placeholder map representation.
        """
        return self.map_representation

    def get_keypoints(self) -> List[Tuple]:
        """
        Get the keypoints extracted from the video frames.

        Returns:
            List[Tuple]: List of keypoints and descriptors.
        """
        return self.keypoints

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
