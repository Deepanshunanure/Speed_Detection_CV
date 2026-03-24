"""Object tracking module using centroid tracking algorithm"""
import numpy as np
from typing import Dict, List, Tuple
from src.config.models import TrackerConfig
from src.models import DetectionResult, TrackingResult, TrackedObject, BoundingBox


class Tracker:
    """Maintains object identity across frames using centroid tracking"""
    
    def __init__(self, config: TrackerConfig):
        """
        Initialize tracker with matching parameters
        
        Args:
            config: TrackerConfig with tracking parameters
        """
        self.config = config
        self._validate_config()
        
        # Tracking state
        self._next_object_id = 0
        self._objects: Dict[int, Tuple[float, float]] = {}  # id -> current centroid
        self._disappeared: Dict[int, int] = {}  # id -> disappeared frame count
        self._trajectories: Dict[int, List[Tuple[float, float]]] = {}  # id -> position history
        self._ages: Dict[int, int] = {}  # id -> frames since first detection
        self._bounding_boxes: Dict[int, BoundingBox] = {}  # id -> current bounding box
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.config.max_tracking_distance <= 0:
            raise ValueError(
                f"Invalid max_tracking_distance: {self.config.max_tracking_distance}. "
                "Must be positive"
            )
        
        if self.config.max_disappeared_frames <= 0:
            raise ValueError(
                f"Invalid max_disappeared_frames: {self.config.max_disappeared_frames}. "
                "Must be positive"
            )
        
        if self.config.trajectory_history_length <= 0:
            raise ValueError(
                f"Invalid trajectory_history_length: {self.config.trajectory_history_length}. "
                "Must be positive"
            )
    
    def update(self, detections: DetectionResult) -> TrackingResult:
        """
        Update tracking with new detections
        
        Args:
            detections: Current frame detection results
            
        Returns:
            Tracking results with IDs and trajectories
        """
        # Extract centroids from detections
        input_centroids = [bbox.centroid for bbox in detections.bounding_boxes]
        
        # If no current objects being tracked
        if len(self._objects) == 0:
            # Register all detections as new objects
            for i, centroid in enumerate(input_centroids):
                self._register(centroid, detections.bounding_boxes[i])
        else:
            # Get current tracked object IDs and centroids
            object_ids = list(self._objects.keys())
            object_centroids = list(self._objects.values())
            
            # Match input centroids to existing object centroids
            if len(input_centroids) > 0:
                # Compute distance matrix between all pairs
                distances = self._compute_distances(object_centroids, input_centroids)
                
                # Perform matching
                matched_object_ids, matched_input_indices = self._match_centroids(
                    distances, object_ids
                )
                
                # Update matched objects
                for obj_id, input_idx in zip(matched_object_ids, matched_input_indices):
                    centroid = input_centroids[input_idx]
                    bbox = detections.bounding_boxes[input_idx]
                    
                    # Update object position
                    self._objects[obj_id] = centroid
                    self._bounding_boxes[obj_id] = bbox
                    
                    # Add to trajectory history
                    self._trajectories[obj_id].append(centroid)
                    
                    # Trim trajectory to max length
                    if len(self._trajectories[obj_id]) > self.config.trajectory_history_length:
                        self._trajectories[obj_id] = self._trajectories[obj_id][
                            -self.config.trajectory_history_length:
                        ]
                    
                    # Reset disappeared counter
                    self._disappeared[obj_id] = 0
                    
                    # Increment age
                    self._ages[obj_id] += 1
                
                # Find unmatched detections and register as new objects
                unmatched_input_indices = set(range(len(input_centroids))) - set(matched_input_indices)
                for input_idx in unmatched_input_indices:
                    self._register(
                        input_centroids[input_idx],
                        detections.bounding_boxes[input_idx]
                    )
                
                # Find unmatched existing objects and increment disappeared counter
                unmatched_object_ids = set(object_ids) - set(matched_object_ids)
                for obj_id in unmatched_object_ids:
                    self._disappeared[obj_id] += 1
            else:
                # No detections, increment disappeared counter for all objects
                for obj_id in object_ids:
                    self._disappeared[obj_id] += 1
        
        # Remove objects that have disappeared for too long
        disappeared_ids = [
            obj_id for obj_id, count in self._disappeared.items()
            if count > self.config.max_disappeared_frames
        ]
        for obj_id in disappeared_ids:
            self._deregister(obj_id)
        
        # Build tracking result
        tracked_objects = []
        for obj_id in self._objects.keys():
            tracked_obj = TrackedObject(
                object_id=obj_id,
                position=self._objects[obj_id],
                bounding_box=self._bounding_boxes[obj_id],
                trajectory=self._trajectories[obj_id].copy(),
                age=self._ages[obj_id],
                disappeared_count=self._disappeared[obj_id]
            )
            tracked_objects.append(tracked_obj)
        
        result = TrackingResult(
            frame_number=detections.frame_number,
            tracked_objects=tracked_objects
        )
        
        return result
    
    def _register(self, centroid: Tuple[float, float], bbox: BoundingBox):
        """
        Register a new object with a unique ID
        
        Args:
            centroid: Object centroid position
            bbox: Object bounding box
        """
        obj_id = self._next_object_id
        self._objects[obj_id] = centroid
        self._bounding_boxes[obj_id] = bbox
        self._disappeared[obj_id] = 0
        self._trajectories[obj_id] = [centroid]
        self._ages[obj_id] = 1
        self._next_object_id += 1
    
    def _deregister(self, obj_id: int):
        """
        Remove an object from tracking
        
        Args:
            obj_id: Object ID to remove
        """
        del self._objects[obj_id]
        del self._bounding_boxes[obj_id]
        del self._disappeared[obj_id]
        del self._trajectories[obj_id]
        del self._ages[obj_id]
    
    def _compute_distances(
        self,
        centroids_a: List[Tuple[float, float]],
        centroids_b: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Compute Euclidean distance matrix between two sets of centroids
        
        Args:
            centroids_a: First set of centroids
            centroids_b: Second set of centroids
            
        Returns:
            Distance matrix of shape (len(centroids_a), len(centroids_b))
        """
        # Convert to numpy arrays for efficient computation
        a = np.array(centroids_a)
        b = np.array(centroids_b)
        
        # Compute pairwise Euclidean distances
        # Using broadcasting: (n, 1, 2) - (1, m, 2) = (n, m, 2)
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        return distances
    
    def _match_centroids(
        self,
        distances: np.ndarray,
        object_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Match centroids using nearest-neighbor with distance threshold
        
        Args:
            distances: Distance matrix (num_objects x num_detections)
            object_ids: List of current object IDs
            
        Returns:
            Tuple of (matched_object_ids, matched_input_indices)
        """
        matched_object_ids = []
        matched_input_indices = []
        
        if distances.size == 0:
            return matched_object_ids, matched_input_indices
        
        # Greedy nearest-neighbor matching
        # Create copies to track which objects/detections are already matched
        available_objects = set(range(len(object_ids)))
        available_detections = set(range(distances.shape[1]))
        
        while len(available_objects) > 0 and len(available_detections) > 0:
            # Find minimum distance among available pairs
            min_dist = float('inf')
            min_obj_idx = None
            min_det_idx = None
            
            for obj_idx in available_objects:
                for det_idx in available_detections:
                    if distances[obj_idx, det_idx] < min_dist:
                        min_dist = distances[obj_idx, det_idx]
                        min_obj_idx = obj_idx
                        min_det_idx = det_idx
            
            # Check if minimum distance is within threshold
            if min_dist <= self.config.max_tracking_distance:
                # Match found
                matched_object_ids.append(object_ids[min_obj_idx])
                matched_input_indices.append(min_det_idx)
                available_objects.remove(min_obj_idx)
                available_detections.remove(min_det_idx)
            else:
                # No more matches within threshold
                break
        
        return matched_object_ids, matched_input_indices
