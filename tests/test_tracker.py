"""Unit tests for Tracker module"""
import pytest
import numpy as np
from src.components.tracker import Tracker
from src.config.models import TrackerConfig
from src.models import DetectionResult, BoundingBox


def create_detection_result(centroids, frame_number=0):
    """Helper to create DetectionResult from centroids"""
    bboxes = []
    contours = []
    
    for cx, cy in centroids:
        bbox = BoundingBox(
            x=int(cx - 10),
            y=int(cy - 10),
            width=20,
            height=20,
            area=400.0,
            centroid=(cx, cy)
        )
        bboxes.append(bbox)
        # Create dummy contour
        contour = np.array([[[int(cx), int(cy)]]], dtype=np.int32)
        contours.append(contour)
    
    return DetectionResult(
        frame_number=frame_number,
        bounding_boxes=bboxes,
        contours=contours,
        foreground_mask=np.zeros((100, 100), dtype=np.uint8)
    )


class TestTrackerInitialization:
    """Test tracker initialization and configuration"""
    
    def test_init_with_valid_config(self):
        """Test tracker initializes with valid configuration"""
        config = TrackerConfig(
            max_tracking_distance=50.0,
            max_disappeared_frames=30,
            trajectory_history_length=100
        )
        tracker = Tracker(config)
        
        assert tracker.config == config
        assert tracker._next_object_id == 0
        assert len(tracker._objects) == 0
    
    def test_init_with_invalid_max_tracking_distance(self):
        """Test tracker raises error for invalid max_tracking_distance"""
        config = TrackerConfig(
            max_tracking_distance=-10.0,
            max_disappeared_frames=30,
            trajectory_history_length=100
        )
        
        with pytest.raises(ValueError, match="max_tracking_distance"):
            Tracker(config)
    
    def test_init_with_invalid_max_disappeared_frames(self):
        """Test tracker raises error for invalid max_disappeared_frames"""
        config = TrackerConfig(
            max_tracking_distance=50.0,
            max_disappeared_frames=0,
            trajectory_history_length=100
        )
        
        with pytest.raises(ValueError, match="max_disappeared_frames"):
            Tracker(config)
    
    def test_init_with_invalid_trajectory_history_length(self):
        """Test tracker raises error for invalid trajectory_history_length"""
        config = TrackerConfig(
            max_tracking_distance=50.0,
            max_disappeared_frames=30,
            trajectory_history_length=-5
        )
        
        with pytest.raises(ValueError, match="trajectory_history_length"):
            Tracker(config)


class TestTrackerBasicFunctionality:
    """Test basic tracking functionality"""
    
    def test_update_with_no_detections(self):
        """Test update with empty detection result"""
        config = TrackerConfig()
        tracker = Tracker(config)
        
        detections = create_detection_result([])
        result = tracker.update(detections)
        
        assert result.frame_number == 0
        assert len(result.tracked_objects) == 0
    
    def test_update_registers_new_object(self):
        """Test that first detection registers a new object"""
        config = TrackerConfig()
        tracker = Tracker(config)
        
        detections = create_detection_result([(100.0, 100.0)])
        result = tracker.update(detections)
        
        assert len(result.tracked_objects) == 1
        assert result.tracked_objects[0].object_id == 0
        assert result.tracked_objects[0].position == (100.0, 100.0)
        assert result.tracked_objects[0].age == 1
        assert result.tracked_objects[0].disappeared_count == 0
    
    def test_update_registers_multiple_objects(self):
        """Test that multiple detections register multiple objects"""
        config = TrackerConfig()
        tracker = Tracker(config)
        
        detections = create_detection_result([
            (100.0, 100.0),
            (200.0, 200.0),
            (300.0, 300.0)
        ])
        result = tracker.update(detections)
        
        assert len(result.tracked_objects) == 3
        object_ids = [obj.object_id for obj in result.tracked_objects]
        assert object_ids == [0, 1, 2]
    
    def test_update_maintains_id_for_nearby_detection(self):
        """Test that object ID persists when detection is nearby"""
        config = TrackerConfig(max_tracking_distance=50.0)
        tracker = Tracker(config)
        
        # First frame
        detections1 = create_detection_result([(100.0, 100.0)])
        result1 = tracker.update(detections1)
        obj_id = result1.tracked_objects[0].object_id
        
        # Second frame - object moved slightly
        detections2 = create_detection_result([(110.0, 110.0)])
        result2 = tracker.update(detections2)
        
        assert len(result2.tracked_objects) == 1
        assert result2.tracked_objects[0].object_id == obj_id
        assert result2.tracked_objects[0].position == (110.0, 110.0)
        assert result2.tracked_objects[0].age == 2
    
    def test_update_assigns_new_id_for_distant_detection(self):
        """Test that new ID is assigned when detection is too far"""
        config = TrackerConfig(max_tracking_distance=50.0)
        tracker = Tracker(config)
        
        # First frame
        detections1 = create_detection_result([(100.0, 100.0)])
        result1 = tracker.update(detections1)
        
        # Second frame - object moved far away (beyond threshold)
        detections2 = create_detection_result([(200.0, 200.0)])
        result2 = tracker.update(detections2)
        
        # Should have 2 objects: one disappeared, one new
        # But disappeared one should still be tracked until max_disappeared_frames
        assert len(result2.tracked_objects) == 2
        object_ids = [obj.object_id for obj in result2.tracked_objects]
        assert 0 in object_ids  # Original object still tracked
        assert 1 in object_ids  # New object registered


class TestTrackerTrajectoryManagement:
    """Test trajectory history management"""
    
    def test_trajectory_builds_over_frames(self):
        """Test that trajectory history accumulates over frames"""
        config = TrackerConfig()
        tracker = Tracker(config)
        
        positions = [(100.0, 100.0), (110.0, 110.0), (120.0, 120.0)]
        
        for i, pos in enumerate(positions):
            detections = create_detection_result([pos], frame_number=i)
            result = tracker.update(detections)
        
        assert len(result.tracked_objects) == 1
        trajectory = result.tracked_objects[0].trajectory
        assert len(trajectory) == 3
        assert trajectory == positions
    
    def test_trajectory_respects_max_length(self):
        """Test that trajectory is trimmed to max length"""
        config = TrackerConfig(trajectory_history_length=5)
        tracker = Tracker(config)
        
        # Track object for 10 frames
        for i in range(10):
            pos = (100.0 + i * 10, 100.0 + i * 10)
            detections = create_detection_result([pos], frame_number=i)
            result = tracker.update(detections)
        
        # Trajectory should only contain last 5 positions
        trajectory = result.tracked_objects[0].trajectory
        assert len(trajectory) == 5
        assert trajectory[0] == (150.0, 150.0)  # Position from frame 5
        assert trajectory[-1] == (190.0, 190.0)  # Position from frame 9


class TestTrackerDisappearanceHandling:
    """Test object disappearance and removal"""
    
    def test_disappeared_counter_increments(self):
        """Test that disappeared counter increments when object not detected"""
        config = TrackerConfig(max_disappeared_frames=5)
        tracker = Tracker(config)
        
        # First frame - detect object
        detections1 = create_detection_result([(100.0, 100.0)])
        result1 = tracker.update(detections1)
        
        # Next frames - no detections
        for i in range(3):
            detections_empty = create_detection_result([])
            result = tracker.update(detections_empty)
        
        assert len(result.tracked_objects) == 1
        assert result.tracked_objects[0].disappeared_count == 3
    
    def test_object_removed_after_max_disappeared_frames(self):
        """Test that object is removed after exceeding max_disappeared_frames"""
        config = TrackerConfig(max_disappeared_frames=3)
        tracker = Tracker(config)
        
        # First frame - detect object
        detections1 = create_detection_result([(100.0, 100.0)])
        tracker.update(detections1)
        
        # Next frames - no detections (exceed threshold)
        for i in range(4):
            detections_empty = create_detection_result([])
            result = tracker.update(detections_empty)
        
        # Object should be removed
        assert len(result.tracked_objects) == 0
    
    def test_disappeared_counter_resets_on_redetection(self):
        """Test that disappeared counter resets when object is detected again"""
        config = TrackerConfig(max_tracking_distance=50.0)
        tracker = Tracker(config)
        
        # First frame - detect object
        detections1 = create_detection_result([(100.0, 100.0)])
        tracker.update(detections1)
        
        # No detection for 2 frames
        for i in range(2):
            detections_empty = create_detection_result([])
            tracker.update(detections_empty)
        
        # Detect object again nearby
        detections2 = create_detection_result([(110.0, 110.0)])
        result = tracker.update(detections2)
        
        assert len(result.tracked_objects) == 1
        assert result.tracked_objects[0].disappeared_count == 0


class TestTrackerCentroidMatching:
    """Test centroid matching algorithm"""
    
    def test_matches_closest_centroids(self):
        """Test that tracker matches to closest centroid"""
        config = TrackerConfig(max_tracking_distance=50.0)
        tracker = Tracker(config)
        
        # First frame - two objects
        detections1 = create_detection_result([
            (100.0, 100.0),
            (200.0, 200.0)
        ])
        result1 = tracker.update(detections1)
        id1 = result1.tracked_objects[0].object_id
        id2 = result1.tracked_objects[1].object_id
        
        # Second frame - objects moved slightly
        detections2 = create_detection_result([
            (105.0, 105.0),
            (205.0, 205.0)
        ])
        result2 = tracker.update(detections2)
        
        # IDs should be maintained
        assert len(result2.tracked_objects) == 2
        positions_by_id = {obj.object_id: obj.position for obj in result2.tracked_objects}
        assert positions_by_id[id1] == (105.0, 105.0)
        assert positions_by_id[id2] == (205.0, 205.0)
    
    def test_maintains_separate_identities(self):
        """Test that separate objects maintain different IDs"""
        config = TrackerConfig(max_tracking_distance=50.0)
        tracker = Tracker(config)
        
        # Detect two objects far apart
        detections = create_detection_result([
            (100.0, 100.0),
            (300.0, 300.0)
        ])
        result = tracker.update(detections)
        
        assert len(result.tracked_objects) == 2
        assert result.tracked_objects[0].object_id != result.tracked_objects[1].object_id


class TestTrackerBoundingBoxes:
    """Test bounding box tracking"""
    
    def test_bounding_box_stored_with_object(self):
        """Test that bounding box is stored with tracked object"""
        config = TrackerConfig()
        tracker = Tracker(config)
        
        detections = create_detection_result([(100.0, 100.0)])
        result = tracker.update(detections)
        
        assert result.tracked_objects[0].bounding_box is not None
        bbox = result.tracked_objects[0].bounding_box
        assert bbox.centroid == (100.0, 100.0)
        assert bbox.width == 20
        assert bbox.height == 20
    
    def test_bounding_box_updates_with_object(self):
        """Test that bounding box updates when object moves"""
        config = TrackerConfig(max_tracking_distance=50.0)
        tracker = Tracker(config)
        
        # First frame
        detections1 = create_detection_result([(100.0, 100.0)])
        tracker.update(detections1)
        
        # Second frame - object moved
        detections2 = create_detection_result([(110.0, 110.0)])
        result2 = tracker.update(detections2)
        
        bbox = result2.tracked_objects[0].bounding_box
        assert bbox.centroid == (110.0, 110.0)
