"""
MongoDB Database Module for Diabetes Prediction Monitoring System
"""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from datetime import datetime, timedelta
import uuid
from typing import Optional, Dict, List, Any

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "diabetes_monitoring"
COLLECTION_NAME = "predictions"


def get_connection():
    """Get or create MongoDB connection - creates fresh connection each time"""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        # Test connection
        client.admin.command('ping')
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        # Create indexes for better query performance (idempotent)
        collection.create_index("timestamp")
        collection.create_index("is_alert")
        collection.create_index("risk_level")
        
        return db, collection
        
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"MongoDB connection failed: {e}")
        return None, None
    except Exception as e:
        print(f"MongoDB error: {e}")
        return None, None


def is_connected() -> bool:
    """Check if MongoDB is connected"""
    try:
        db, collection = get_connection()
        return collection is not None
    except:
        return False


def log_prediction(
    input_features: Dict[str, Any],
    risk_score: float,
    session_id: Optional[str] = None
) -> Optional[str]:
    """
    Log a prediction to MongoDB
    
    Args:
        input_features: Dictionary of input features used for prediction
        risk_score: The calculated risk score (0-1)
        session_id: Optional session identifier
    
    Returns:
        The inserted document ID or None if failed
    """
    _, collection = get_connection()
    
    if collection is None:
        return None
    
    # Determine risk level
    if risk_score < 0.3:
        risk_level = "low"
    elif risk_score < 0.6:
        risk_level = "medium"
    else:
        risk_level = "high"
    
    # Create document
    document = {
        "prediction_id": str(uuid.uuid4()),
        "session_id": session_id or str(uuid.uuid4()),
        "timestamp": datetime.now(),
        "input_features": input_features,
        "risk_score": float(risk_score),
        "risk_level": risk_level,
        "is_alert": risk_score >= 0.6
    }
    
    try:
        result = collection.insert_one(document)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Failed to log prediction: {e}")
        return None


def get_alerts(limit: int = 50) -> List[Dict]:
    """Get recent high-risk alerts"""
    _, collection = get_connection()
    
    if collection is None:
        return []
    
    try:
        alerts = list(collection.find(
            {"is_alert": True}
        ).sort("timestamp", -1).limit(limit))
        
        # Convert ObjectId to string for JSON serialization
        for alert in alerts:
            alert['_id'] = str(alert['_id'])
        
        return alerts
    except Exception as e:
        print(f"Failed to get alerts: {e}")
        return []


def get_alert_count(days: Optional[int] = None) -> int:
    """
    Get count of alerts
    
    Args:
        days: If specified, count alerts from last N days. None for total count.
    """
    _, collection = get_connection()
    
    if collection is None:
        return 0
    
    try:
        query = {"is_alert": True}
        
        if days is not None:
            start_date = datetime.now() - timedelta(days=days)
            query["timestamp"] = {"$gte": start_date}
        
        return collection.count_documents(query)
    except Exception as e:
        print(f"Failed to count alerts: {e}")
        return 0


def get_today_alert_count() -> int:
    """Get count of alerts from today"""
    _, collection = get_connection()
    
    if collection is None:
        return 0
    
    try:
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return collection.count_documents({
            "is_alert": True,
            "timestamp": {"$gte": today_start}
        })
    except Exception as e:
        print(f"Failed to count today's alerts: {e}")
        return 0


def get_total_predictions() -> int:
    """Get total number of predictions"""
    _, collection = get_connection()
    
    if collection is None:
        return 0
    
    try:
        return collection.count_documents({})
    except Exception as e:
        print(f"Failed to count predictions: {e}")
        return 0


def get_alert_trend(days: int = 7) -> List[Dict]:
    """
    Get daily alert counts for the last N days
    
    Returns:
        List of dicts with 'date' and 'count' keys
    """
    _, collection = get_connection()
    
    if collection is None:
        return []
    
    try:
        start_date = datetime.now() - timedelta(days=days)
        
        pipeline = [
            {
                "$match": {
                    "is_alert": True,
                    "timestamp": {"$gte": start_date}
                }
            },
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$timestamp"
                        }
                    },
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"_id": 1}
            }
        ]
        
        result = list(collection.aggregate(pipeline))
        return [{"date": r["_id"], "count": r["count"]} for r in result]
    except Exception as e:
        print(f"Failed to get alert trend: {e}")
        return []


def get_risk_distribution() -> Dict[str, int]:
    """Get distribution of risk levels"""
    _, collection = get_connection()
    
    if collection is None:
        return {"low": 0, "medium": 0, "high": 0}
    
    try:
        pipeline = [
            {
                "$group": {
                    "_id": "$risk_level",
                    "count": {"$sum": 1}
                }
            }
        ]
        
        result = list(collection.aggregate(pipeline))
        distribution = {"low": 0, "medium": 0, "high": 0}
        
        for r in result:
            if r["_id"] in distribution:
                distribution[r["_id"]] = r["count"]
        
        return distribution
    except Exception as e:
        print(f"Failed to get risk distribution: {e}")
        return {"low": 0, "medium": 0, "high": 0}


def get_recent_predictions(limit: int = 10) -> List[Dict]:
    """Get most recent predictions"""
    _, collection = get_connection()
    
    if collection is None:
        return []
    
    try:
        predictions = list(collection.find().sort("timestamp", -1).limit(limit))
        
        for pred in predictions:
            pred['_id'] = str(pred['_id'])
        
        return predictions
    except Exception as e:
        print(f"Failed to get recent predictions: {e}")
        return []


def get_average_risk_score(days: Optional[int] = None) -> float:
    """Get average risk score"""
    _, collection = get_connection()
    
    if collection is None:
        return 0.0
    
    try:
        match_stage = {}
        
        if days is not None:
            start_date = datetime.now() - timedelta(days=days)
            match_stage = {"timestamp": {"$gte": start_date}}
        
        pipeline = [
            {"$match": match_stage} if match_stage else {"$match": {}},
            {
                "$group": {
                    "_id": None,
                    "avg_score": {"$avg": "$risk_score"}
                }
            }
        ]
        
        result = list(collection.aggregate(pipeline))
        
        if result:
            return round(result[0]["avg_score"], 4)
        return 0.0
    except Exception as e:
        print(f"Failed to get average risk score: {e}")
        return 0.0

