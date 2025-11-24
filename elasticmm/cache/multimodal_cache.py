"""
Multimodal Input Cache System

This module implements a caching system for multimodal inputs, specifically images.
It uses hash-based caching to avoid sending duplicate image data over the network.
The cache is synchronized between client and device sides using LRU eviction policy.
"""

import hashlib
import base64
import time
import json
import threading
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
from dataclasses import dataclass, asdict
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry for multimodal content"""
    hash_key: str
    content_type: str  # 'image', 'video', 'audio', etc.
    content_data: str  # Base64 encoded content
    content_size: int  # Size in bytes
    created_at: float  # Timestamp when created
    last_accessed: float  # Timestamp when last accessed
    access_count: int  # Number of times accessed
    metadata: Dict[str, Any]  # Additional metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class CacheStats:
    """Cache statistics"""
    total_entries: int
    total_size_bytes: int
    hit_count: int
    miss_count: int
    eviction_count: int
    last_cleanup: float

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0

    @property
    def total_size_mb(self) -> float:
        """Get total size in MB"""
        return self.total_size_bytes / (1024 * 1024)


class MultimodalCache:
    """
    Multimodal input cache with LRU eviction policy
    
    This cache stores multimodal content (images, videos, etc.) using hash-based keys.
    It supports both client-side and device-side synchronization.
    """

    def __init__(self, max_size_mb: float = 1024.0, max_entries: int = 10000):
        """
        Initialize the multimodal cache
        
        Args:
            max_size_mb: Maximum cache size in MB
            max_entries: Maximum number of cache entries
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.max_entries = max_entries
        
        # LRU cache using OrderedDict
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Statistics
        self._stats = CacheStats(
            total_entries=0,
            total_size_bytes=0,
            hit_count=0,
            miss_count=0,
            eviction_count=0,
            last_cleanup=time.time()
        )
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Content type handlers
        self._content_handlers = {
            'image': self._process_image,
            'video': self._process_video,
            'audio': self._process_audio,
        }
        
        logger.info(f"MultimodalCache initialized: max_size={max_size_mb}MB, max_entries={max_entries}")

    def _generate_hash(self, content: str, content_type: str) -> str:
        """
        Generate hash key for content
        
        Args:
            content: Base64 encoded content
            content_type: Type of content (image, video, etc.)
            
        Returns:
            Hash key string
        """
        # Create hash from content and type
        hash_input = f"{content_type}:{content}"
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

    def _process_image(self, content: str) -> Tuple[str, int]:
        """
        Process image content
        
        Args:
            content: Base64 encoded image data
            
        Returns:
            Tuple of (processed_content, size_bytes)
        """
        try:
            # Decode base64 to get actual size
            decoded = base64.b64decode(content)
            size = len(decoded)
            
            # For images, we can add validation here
            # For now, just return the content as-is
            return content, size
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise ValueError(f"Invalid image content: {e}")

    def _process_video(self, content: str) -> Tuple[str, int]:
        """
        Process video content
        
        Args:
            content: Base64 encoded video data
            
        Returns:
            Tuple of (processed_content, size_bytes)
        """
        try:
            decoded = base64.b64decode(content)
            size = len(decoded)
            return content, size
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise ValueError(f"Invalid video content: {e}")

    def _process_audio(self, content: str) -> Tuple[str, int]:
        """
        Process audio content
        
        Args:
            content: Base64 encoded audio data
            
        Returns:
            Tuple of (processed_content, size_bytes)
        """
        try:
            decoded = base64.b64decode(content)
            size = len(decoded)
            return content, size
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise ValueError(f"Invalid audio content: {e}")

    def _evict_lru(self) -> int:
        """
        Evict least recently used entries
        
        Returns:
            Number of entries evicted
        """
        evicted_count = 0
        
        while (self._stats.total_size_bytes > self.max_size_bytes or 
               len(self._cache) > self.max_entries):
            
            if not self._cache:
                break
                
            # Remove least recently used entry
            key, entry = self._cache.popitem(last=False)
            
            self._stats.total_size_bytes -= entry.content_size
            self._stats.eviction_count += 1
            evicted_count += 1
            
            logger.debug(f"Evicted cache entry: {key[:16]}... (size: {entry.content_size} bytes)")
        
        return evicted_count

    def _update_access(self, key: str, entry: CacheEntry):
        """
        Update access information for cache entry
        
        Args:
            key: Cache key
            entry: Cache entry
        """
        entry.last_accessed = time.time()
        entry.access_count += 1
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)

    def put(self, content: str, content_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store content in cache
        
        Args:
            content: Base64 encoded content
            content_type: Type of content (image, video, audio, etc.)
            metadata: Optional metadata
            
        Returns:
            Hash key for the content
        """
        with self._lock:
            # Generate hash key
            hash_key = self._generate_hash(content, content_type)
            
            # Check if already exists
            if hash_key in self._cache:
                self._update_access(hash_key, self._cache[hash_key])
                logger.debug(f"Content already in cache: {hash_key[:16]}...")
                return hash_key
            
            # Process content based on type
            if content_type not in self._content_handlers:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            processed_content, size = self._content_handlers[content_type](content)
            
            # Create cache entry
            entry = CacheEntry(
                hash_key=hash_key,
                content_type=content_type,
                content_data=processed_content,
                content_size=size,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                metadata=metadata or {}
            )
            
            # Store in cache
            self._cache[hash_key] = entry
            self._stats.total_size_bytes += size
            self._stats.total_entries = len(self._cache)
            
            # Evict if necessary
            evicted = self._evict_lru()
            if evicted > 0:
                logger.info(f"Evicted {evicted} entries to make room for new content")
            
            logger.debug(f"Stored content in cache: {hash_key[:16]}... (size: {size} bytes)")
            return hash_key

    def get(self, hash_key: str) -> Optional[CacheEntry]:
        """
        Retrieve content from cache
        
        Args:
            hash_key: Hash key of the content
            
        Returns:
            Cache entry if found, None otherwise
        """
        with self._lock:
            if hash_key in self._cache:
                entry = self._cache[hash_key]
                self._update_access(hash_key, entry)
                self._stats.hit_count += 1
                logger.debug(f"Cache hit: {hash_key[:16]}...")
                return entry
            else:
                self._stats.miss_count += 1
                logger.debug(f"Cache miss: {hash_key[:16]}...")
                return None

    def contains(self, hash_key: str) -> bool:
        """
        Check if content exists in cache
        
        Args:
            hash_key: Hash key of the content
            
        Returns:
            True if exists, False otherwise
        """
        with self._lock:
            return hash_key in self._cache

    def remove(self, hash_key: str) -> bool:
        """
        Remove content from cache
        
        Args:
            hash_key: Hash key of the content
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if hash_key in self._cache:
                entry = self._cache.pop(hash_key)
                self._stats.total_size_bytes -= entry.content_size
                self._stats.total_entries = len(self._cache)
                logger.debug(f"Removed from cache: {hash_key[:16]}...")
                return True
            return False

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._stats.total_size_bytes = 0
            self._stats.total_entries = 0
            logger.info("Cache cleared")

    def get_stats(self) -> CacheStats:
        """
        Get cache statistics
        
        Returns:
            Current cache statistics
        """
        with self._lock:
            return CacheStats(
                total_entries=self._stats.total_entries,
                total_size_bytes=self._stats.total_size_bytes,
                hit_count=self._stats.hit_count,
                miss_count=self._stats.miss_count,
                eviction_count=self._stats.eviction_count,
                last_cleanup=self._stats.last_cleanup
            )

    def cleanup_expired(self, max_age_seconds: float = 3600.0) -> int:
        """
        Remove expired entries from cache
        
        Args:
            max_age_seconds: Maximum age in seconds
            
        Returns:
            Number of entries removed
        """
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, entry in self._cache.items():
                if current_time - entry.created_at > max_age_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self._cache.pop(key)
                self._stats.total_size_bytes -= entry.content_size
                self._stats.eviction_count += 1
            
            self._stats.total_entries = len(self._cache)
            self._stats.last_cleanup = current_time
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired entries")
            
            return len(expired_keys)

    def export_cache(self) -> Dict[str, Any]:
        """
        Export cache data for synchronization
        
        Returns:
            Dictionary containing cache data
        """
        with self._lock:
            return {
                'entries': {key: entry.to_dict() for key, entry in self._cache.items()},
                'stats': self.get_stats().to_dict() if hasattr(self.get_stats(), 'to_dict') else asdict(self.get_stats()),
                'exported_at': time.time()
            }

    def import_cache(self, cache_data: Dict[str, Any]) -> int:
        """
        Import cache data from synchronization
        
        Args:
            cache_data: Dictionary containing cache data
            
        Returns:
            Number of entries imported
        """
        with self._lock:
            imported_count = 0
            
            if 'entries' in cache_data:
                for key, entry_data in cache_data['entries'].items():
                    try:
                        entry = CacheEntry.from_dict(entry_data)
                        self._cache[key] = entry
                        imported_count += 1
                    except Exception as e:
                        logger.error(f"Error importing cache entry {key}: {e}")
            
            # Recalculate statistics
            self._stats.total_entries = len(self._cache)
            self._stats.total_size_bytes = sum(entry.content_size for entry in self._cache.values())
            
            logger.info(f"Imported {imported_count} cache entries")
            return imported_count

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get detailed cache information
        
        Returns:
            Dictionary with cache information
        """
        with self._lock:
            stats = self.get_stats()
            
            return {
                'cache_size': {
                    'entries': stats.total_entries,
                    'size_bytes': stats.total_size_bytes,
                    'size_mb': stats.total_size_mb,
                    'max_size_mb': self.max_size_bytes / (1024 * 1024),
                    'utilization': stats.total_size_bytes / self.max_size_bytes
                },
                'performance': {
                    'hit_rate': stats.hit_rate,
                    'hit_count': stats.hit_count,
                    'miss_count': stats.miss_count,
                    'eviction_count': stats.eviction_count
                },
                'content_types': {
                    content_type: sum(1 for entry in self._cache.values() if entry.content_type == content_type)
                    for content_type in set(entry.content_type for entry in self._cache.values())
                },
                'oldest_entry': min((entry.created_at for entry in self._cache.values()), default=0),
                'newest_entry': max((entry.created_at for entry in self._cache.values()), default=0)
            }


class CacheSynchronizer:
    """
    Cache synchronizer for client-device synchronization
    
    This class handles synchronization of cache data between client and device sides.
    """

    def __init__(self, cache: MultimodalCache):
        """
        Initialize cache synchronizer
        
        Args:
            cache: Multimodal cache instance
        """
        self.cache = cache
        self._sync_lock = threading.Lock()

    def sync_to_device(self, device_endpoint: str) -> bool:
        """
        Synchronize cache to device
        
        Args:
            device_endpoint: Device endpoint URL
            
        Returns:
            True if successful, False otherwise
        """
        with self._sync_lock:
            try:
                # Export cache data
                cache_data = self.cache.export_cache()
                
                # In a real implementation, this would send data to device
                # For now, we'll just log the operation
                logger.info(f"Syncing cache to device {device_endpoint}: {len(cache_data['entries'])} entries")
                
                # Simulate network operation
                time.sleep(0.1)
                
                return True
            except Exception as e:
                logger.error(f"Failed to sync cache to device: {e}")
                return False

    def sync_from_device(self, device_endpoint: str) -> bool:
        """
        Synchronize cache from device
        
        Args:
            device_endpoint: Device endpoint URL
            
        Returns:
            True if successful, False otherwise
        """
        with self._sync_lock:
            try:
                # In a real implementation, this would fetch data from device
                # For now, we'll just log the operation
                logger.info(f"Syncing cache from device {device_endpoint}")
                
                # Simulate network operation
                time.sleep(0.1)
                
                return True
            except Exception as e:
                logger.error(f"Failed to sync cache from device: {e}")
                return False


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create cache instance
    cache = MultimodalCache(max_size_mb=100.0, max_entries=1000)
    
    # Example image data (base64 encoded)
    sample_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    # Store image in cache
    hash_key = cache.put(sample_image, "image", {"format": "png", "width": 1, "height": 1})
    print(f"Stored image with hash: {hash_key}")
    
    # Retrieve from cache
    entry = cache.get(hash_key)
    if entry:
        print(f"Retrieved image: {entry.content_type}, size: {entry.content_size} bytes")
    
    # Get cache statistics
    stats = cache.get_stats()
    print(f"Cache stats: {stats.total_entries} entries, {stats.total_size_mb:.2f} MB, hit rate: {stats.hit_rate:.2%}")
    
    # Get detailed cache info
    info = cache.get_cache_info()
    print(f"Cache info: {json.dumps(info, indent=2)}")




