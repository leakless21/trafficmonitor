#!/usr/bin/env python3
"""
Database utility module for traffic monitoring system.
Provides upsert operations and best practices for handling vehicle data.
"""

import sqlite3
import threading
from contextlib import contextmanager
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime

class TrafficMonitorDB:
    """Database manager for traffic monitoring with best practices"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._init_connection()
    
    def _init_connection(self):
        """Initialize database connection with proper settings"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = 10000")
        conn.row_factory = sqlite3.Row
        return conn
    
    @property
    def connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = self._init_connection()
        return self._local.connection
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions"""
        conn = self.connection
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    def upsert_vehicle(self, lp_text: str, vehicle_class: str = None, 
                      ocr_conf: float = 0.0) -> int:
        """
        Insert or update a vehicle record based on license plate.
        Returns the vehicle_id.
        """
        if not lp_text or lp_text.strip() == '':
            raise ValueError("License plate text cannot be empty")
        
        lp_text = lp_text.strip()
        current_ts = int(datetime.now().timestamp() * 1000)
        
        with self.transaction() as conn:
            cursor = conn.cursor()
            
            # Try to get existing vehicle
            cursor.execute("""
                SELECT vehicle_id, total_detections, best_ocr_conf 
                FROM vehicles 
                WHERE lp_text = ?
            """, (lp_text,))
            
            result = cursor.fetchone()
            
            if result:
                # Update existing vehicle
                vehicle_id, total_detections, best_ocr_conf = result
                new_total = total_detections + 1
                new_best_conf = max(best_ocr_conf, ocr_conf) if ocr_conf else best_ocr_conf
                
                cursor.execute("""
                    UPDATE vehicles 
                    SET last_seen = ?, total_detections = ?, best_ocr_conf = ?,
                        vehicle_class = COALESCE(?, vehicle_class)
                    WHERE vehicle_id = ?
                """, (current_ts, new_total, new_best_conf, vehicle_class, vehicle_id))
                
                return vehicle_id
            else:
                # Insert new vehicle
                cursor.execute("""
                    INSERT INTO vehicles (lp_text, vehicle_class, first_seen, last_seen, 
                                        total_detections, best_ocr_conf)
                    VALUES (?, ?, ?, ?, 1, ?)
                """, (lp_text, vehicle_class, current_ts, current_ts, ocr_conf or 0.0))
                
                return cursor.lastrowid
    
    def insert_detection(self, camera_id: str, lp_text: str, vehicle_class: str = None,
                        ocr_conf: float = 0.0, bbox: Dict[str, float] = None) -> int:
        """
        Insert a new detection record, automatically handling vehicle upsert.
        Returns the detection_id.
        """
        # First, upsert the vehicle
        vehicle_id = self.upsert_vehicle(lp_text, vehicle_class, ocr_conf)
        
        # Then insert the detection
        current_ts = int(datetime.now().timestamp() * 1000)
        bbox = bbox or {}
        
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO plate_results (ts, camera_id, vehicle_id, ocr_conf,
                                         bbox_x, bbox_y, bbox_width, bbox_height)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (current_ts, camera_id, vehicle_id, ocr_conf,
                  bbox.get('x'), bbox.get('y'), bbox.get('width'), bbox.get('height')))
            
            detection_id = cursor.lastrowid
            
            # Update latest results table
            cursor.execute("""
                INSERT OR REPLACE INTO plate_results_latest (camera_id, vehicle_id, last_seen, ocr_conf)
                VALUES (?, ?, ?, ?)
            """, (camera_id, vehicle_id, current_ts, ocr_conf))
            
            return detection_id
    
    def get_vehicle_by_plate(self, lp_text: str) -> Optional[Dict[str, Any]]:
        """Get vehicle information by license plate"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT vehicle_id, lp_text, vehicle_class, first_seen, last_seen,
                   total_detections, best_ocr_conf
            FROM vehicles
            WHERE lp_text = ?
        """, (lp_text,))
        
        result = cursor.fetchone()
        return dict(result) if result else None
    
    def get_recent_detections(self, limit: int = 100, camera_id: str = None) -> List[Dict[str, Any]]:
        """Get recent detections with vehicle information"""
        cursor = self.connection.cursor()
        
        if camera_id:
            cursor.execute("""
                SELECT pr.detection_id, pr.ts, pr.camera_id, pr.ocr_conf,
                       v.vehicle_id, v.lp_text, v.vehicle_class
                FROM plate_results pr
                JOIN vehicles v ON pr.vehicle_id = v.vehicle_id
                WHERE pr.camera_id = ?
                ORDER BY pr.ts DESC
                LIMIT ?
            """, (camera_id, limit))
        else:
            cursor.execute("""
                SELECT pr.detection_id, pr.ts, pr.camera_id, pr.ocr_conf,
                       v.vehicle_id, v.lp_text, v.vehicle_class
                FROM plate_results pr
                JOIN vehicles v ON pr.vehicle_id = v.vehicle_id
                ORDER BY pr.ts DESC
                LIMIT ?
            """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_vehicle_history(self, vehicle_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get detection history for a specific vehicle"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT pr.detection_id, pr.ts, pr.camera_id, pr.ocr_conf,
                   pr.bbox_x, pr.bbox_y, pr.bbox_width, pr.bbox_height
            FROM plate_results pr
            WHERE pr.vehicle_id = ?
            ORDER BY pr.ts DESC
            LIMIT ?
        """, (vehicle_id, limit))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_duplicate_analysis(self) -> List[Dict[str, Any]]:
        """Analyze any remaining duplicates (should be empty after migration)"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT lp_text, COUNT(*) as count
            FROM vehicles
            GROUP BY lp_text
            HAVING count > 1
            ORDER BY count DESC
        """)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        cursor = self.connection.cursor()
        
        # Total vehicles
        cursor.execute("SELECT COUNT(*) FROM vehicles")
        total_vehicles = cursor.fetchone()[0]
        
        # Total detections
        cursor.execute("SELECT COUNT(*) FROM plate_results")
        total_detections = cursor.fetchone()[0]
        
        # Recent activity (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) FROM plate_results 
            WHERE ts > (strftime('%s','now')-86400)*1000
        """)
        recent_detections = cursor.fetchone()[0]
        
        # Top vehicles by detection count
        cursor.execute("""
            SELECT lp_text, total_detections
            FROM vehicles
            ORDER BY total_detections DESC
            LIMIT 10
        """)
        top_vehicles = [dict(row) for row in cursor.fetchall()]
        
        return {
            'total_vehicles': total_vehicles,
            'total_detections': total_detections,
            'recent_detections_24h': recent_detections,
            'top_vehicles': top_vehicles
        }
    
    def cleanup_old_detections(self, days_to_keep: int = 30) -> int:
        """Clean up old detection records while keeping vehicle records"""
        cutoff_ts = int((datetime.now().timestamp() - (days_to_keep * 86400)) * 1000)
        
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM plate_results WHERE ts < ?", (cutoff_ts,))
            deleted_count = cursor.rowcount
            
            # Update vehicle last_seen timestamps based on remaining detections
            cursor.execute("""
                UPDATE vehicles 
                SET last_seen = (
                    SELECT MAX(ts) FROM plate_results 
                    WHERE plate_results.vehicle_id = vehicles.vehicle_id
                )
                WHERE vehicle_id IN (
                    SELECT DISTINCT vehicle_id FROM plate_results
                )
            """)
            
            return deleted_count
    
    def close(self):
        """Close database connections"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()

# Example usage functions
def example_usage():
    """Example of how to use the database utilities"""
    db = TrafficMonitorDB("data/db/traffic_monitor.db")
    
    # Insert a new detection
    detection_id = db.insert_detection(
        camera_id="cam_001",
        lp_text="ABC123",
        vehicle_class="car",
        ocr_conf=0.95,
        bbox={'x': 100, 'y': 200, 'width': 150, 'height': 75}
    )
    print(f"Inserted detection: {detection_id}")
    
    # Get vehicle info
    vehicle = db.get_vehicle_by_plate("ABC123")
    print(f"Vehicle info: {vehicle}")
    
    # Get recent detections
    recent = db.get_recent_detections(limit=10)
    print(f"Recent detections: {len(recent)}")
    
    # Get stats
    stats = db.get_stats()
    print(f"Database stats: {stats}")
    
    db.close()

if __name__ == "__main__":
    example_usage() 