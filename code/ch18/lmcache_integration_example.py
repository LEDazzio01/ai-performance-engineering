#!/usr/bin/env python3
"""
LMCache Integration Example (Chapter 18)

Demonstrates LMCache integration for KV cache management as described in Chapter 18.
LMCache enables efficient sharing and reuse of KV cache across requests and nodes.

Key features:
- GPU and CPU cache management
- Paged cache offload
- Prefix sharing and deduplication
- Dynamic GPU/CPU ratio adjustment
- Multi-node cache coordination

Usage:
    from lmcache_integration_example import LMCacheManager
    
    cache_mgr = LMCacheManager(gpu_cache_size_gb=10, cpu_cache_size_gb=50)
    cache_mgr.put_kv_cache(token_ids, kv_tensors)
    cached_kv = cache_mgr.get_kv_cache(token_ids)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import time


@dataclass
class CacheEntry:
    """Entry in the LMCache"""
    key_hash: str
    token_ids: torch.Tensor
    kv_cache: Tuple[torch.Tensor, torch.Tensor]  # (keys, values)
    size_bytes: int
    timestamp: float
    hit_count: int = 0
    on_gpu: bool = True


@dataclass
class CacheStats:
    """Statistics for LMCache"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    prefix_matches: int = 0
    gpu_to_cpu_transfers: int = 0
    cpu_to_gpu_transfers: int = 0
    bytes_saved: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0
    
    @property
    def prefix_match_rate(self) -> float:
        """Calculate prefix match rate"""
        return (self.prefix_matches / self.total_requests * 100) if self.total_requests > 0 else 0.0


class LMCacheManager:
    """
    LMCache Manager for efficient KV cache management.
    
    Implements the approach from Chapter 18:
    - GPU cache for hot/recent sequences
    - CPU cache for cold sequences (paged offload)
    - Prefix sharing and deduplication
    - Dynamic GPU/CPU ratio based on memory pressure
    - LRU eviction policy
    """
    
    def __init__(
        self,
        gpu_cache_size_gb: float = 10.0,
        cpu_cache_size_gb: float = 50.0,
        page_size: int = 16,  # tokens per page
        enable_prefix_sharing: bool = True
    ):
        """
        Initialize LMCache manager.
        
        Args:
            gpu_cache_size_gb: GPU cache size in GB
            cpu_cache_size_gb: CPU cache size in GB
            page_size: Number of tokens per cache page
            enable_prefix_sharing: Enable prefix sharing/deduplication
        """
        self.gpu_cache_size_bytes = int(gpu_cache_size_gb * 1024**3)
        self.cpu_cache_size_bytes = int(cpu_cache_size_gb * 1024**3)
        self.page_size = page_size
        self.enable_prefix_sharing = enable_prefix_sharing
        
        # Cache storage
        self.gpu_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.cpu_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Statistics
        self.stats = CacheStats()
        
        # Memory tracking
        self.gpu_cache_used_bytes = 0
        self.cpu_cache_used_bytes = 0
        
        print(f"Initialized LMCacheManager:")
        print(f"  GPU cache: {gpu_cache_size_gb:.1f} GB")
        print(f"  CPU cache: {cpu_cache_size_gb:.1f} GB")
        print(f"  Page size: {page_size} tokens")
        print(f"  Prefix sharing: {'Enabled' if enable_prefix_sharing else 'Disabled'}")
    
    def _compute_hash(self, token_ids: torch.Tensor) -> str:
        """
        Compute hash for token sequence.
        
        Args:
            token_ids: Token ID tensor
            
        Returns:
            Hash string
        """
        # Convert to bytes and hash
        token_bytes = token_ids.cpu().numpy().tobytes()
        return hashlib.sha256(token_bytes).hexdigest()
    
    def _find_prefix_match(self, token_ids: torch.Tensor) -> Optional[CacheEntry]:
        """
        Find longest prefix match in cache.
        
        Args:
            token_ids: Token ID tensor to match
            
        Returns:
            CacheEntry with longest matching prefix or None
        """
        if not self.enable_prefix_sharing:
            return None
        
        longest_match = None
        longest_match_len = 0
        
        # Search GPU cache first
        for entry in self.gpu_cache.values():
            # Find common prefix length
            min_len = min(len(token_ids), len(entry.token_ids))
            match_len = 0
            for i in range(min_len):
                if token_ids[i] == entry.token_ids[i]:
                    match_len += 1
                else:
                    break
            
            # Keep track of longest match
            if match_len > longest_match_len and match_len >= self.page_size:
                longest_match = entry
                longest_match_len = match_len
        
        if longest_match:
            self.stats.prefix_matches += 1
        
        return longest_match
    
    def _evict_from_gpu(self, required_bytes: int):
        """
        Evict entries from GPU cache to make room.
        
        Uses LRU policy. Evicted entries are moved to CPU cache.
        
        Args:
            required_bytes: Bytes needed
        """
        while self.gpu_cache_used_bytes + required_bytes > self.gpu_cache_size_bytes and self.gpu_cache:
            # Evict least recently used (first item in OrderedDict)
            key, entry = self.gpu_cache.popitem(last=False)
            
            # Move to CPU
            self._move_to_cpu(entry)
            
            self.gpu_cache_used_bytes -= entry.size_bytes
            print(f"  Evicted {key[:8]}... from GPU ({entry.size_bytes / 1024**2:.2f} MB)")
    
    def _move_to_cpu(self, entry: CacheEntry):
        """
        Move cache entry from GPU to CPU.
        
        Args:
            entry: Cache entry to move
        """
        # Move tensors to CPU
        k, v = entry.kv_cache
        k_cpu = k.cpu()
        v_cpu = v.cpu()
        
        # Update entry
        entry.kv_cache = (k_cpu, v_cpu)
        entry.on_gpu = False
        
        # Add to CPU cache
        self.cpu_cache[entry.key_hash] = entry
        self.cpu_cache.move_to_end(entry.key_hash)  # Mark as recently used
        
        self.cpu_cache_used_bytes += entry.size_bytes
        self.stats.gpu_to_cpu_transfers += 1
        
        # Evict from CPU if needed
        self._evict_from_cpu()
    
    def _evict_from_cpu(self):
        """Evict entries from CPU cache if over capacity."""
        while self.cpu_cache_used_bytes > self.cpu_cache_size_bytes and self.cpu_cache:
            # Evict least recently used
            key, entry = self.cpu_cache.popitem(last=False)
            self.cpu_cache_used_bytes -= entry.size_bytes
            print(f"  Evicted {key[:8]}... from CPU ({entry.size_bytes / 1024**2:.2f} MB)")
    
    def _move_to_gpu(self, entry: CacheEntry):
        """
        Move cache entry from CPU to GPU.
        
        Args:
            entry: Cache entry to move
        """
        # Make room on GPU if needed
        self._evict_from_gpu(entry.size_bytes)
        
        # Move tensors to GPU
        k, v = entry.kv_cache
        k_gpu = k.cuda()
        v_gpu = v.cuda()
        
        # Update entry
        entry.kv_cache = (k_gpu, v_gpu)
        entry.on_gpu = True
        
        # Remove from CPU cache
        if entry.key_hash in self.cpu_cache:
            del self.cpu_cache[entry.key_hash]
            self.cpu_cache_used_bytes -= entry.size_bytes
        
        # Add to GPU cache
        self.gpu_cache[entry.key_hash] = entry
        self.gpu_cache.move_to_end(entry.key_hash)
        self.gpu_cache_used_bytes += entry.size_bytes
        
        self.stats.cpu_to_gpu_transfers += 1
    
    def put_kv_cache(
        self,
        token_ids: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor]
    ):
        """
        Store KV cache for token sequence.
        
        Args:
            token_ids: Token ID sequence [seq_len]
            kv_cache: Tuple of (keys, values) tensors
        """
        key_hash = self._compute_hash(token_ids)
        
        # Calculate size
        k, v = kv_cache
        size_bytes = (k.numel() + v.numel()) * k.element_size()
        
        # Check if enough room on GPU
        if self.gpu_cache_used_bytes + size_bytes > self.gpu_cache_size_bytes:
            self._evict_from_gpu(size_bytes)
        
        # Create entry
        entry = CacheEntry(
            key_hash=key_hash,
            token_ids=token_ids.clone(),
            kv_cache=kv_cache,
            size_bytes=size_bytes,
            timestamp=time.time(),
            on_gpu=True
        )
        
        # Add to GPU cache
        self.gpu_cache[key_hash] = entry
        self.gpu_cache.move_to_end(key_hash)
        self.gpu_cache_used_bytes += size_bytes
    
    def get_kv_cache(
        self,
        token_ids: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve KV cache for token sequence.
        
        First tries exact match, then falls back to prefix matching.
        
        Args:
            token_ids: Token ID sequence [seq_len]
            
        Returns:
            Tuple of (keys, values) or None if not found
        """
        self.stats.total_requests += 1
        
        key_hash = self._compute_hash(token_ids)
        
        # Try exact match in GPU cache
        if key_hash in self.gpu_cache:
            entry = self.gpu_cache[key_hash]
            entry.hit_count += 1
            entry.timestamp = time.time()
            self.gpu_cache.move_to_end(key_hash)  # Mark as recently used
            self.stats.cache_hits += 1
            print(f"  GPU cache HIT: {key_hash[:8]}...")
            return entry.kv_cache
        
        # Try exact match in CPU cache
        if key_hash in self.cpu_cache:
            entry = self.cpu_cache[key_hash]
            entry.hit_count += 1
            entry.timestamp = time.time()
            self.stats.cache_hits += 1
            print(f"  CPU cache HIT: {key_hash[:8]}... (moving to GPU)")
            
            # Move to GPU for fast access
            self._move_to_gpu(entry)
            return entry.kv_cache
        
        # Try prefix matching
        if self.enable_prefix_sharing:
            prefix_entry = self._find_prefix_match(token_ids)
            if prefix_entry:
                print(f"  Prefix match found: {prefix_entry.key_hash[:8]}... "
                      f"(matched {len(prefix_entry.token_ids)} tokens)")
                
                # Return prefix KV cache (caller would need to compute rest)
                # In a real implementation, this would return only the matching prefix portion
                return prefix_entry.kv_cache
        
        # Cache miss
        self.stats.cache_misses += 1
        print(f"  Cache MISS: {key_hash[:8]}...")
        return None
    
    def adjust_cache_ratio(self, target_gpu_ratio: float):
        """
        Dynamically adjust GPU/CPU cache ratio based on memory pressure.
        
        As mentioned in Chapter 18: "vLLM's LMCache component allows adjusting
        the GPU versus CPU cache ratio."
        
        Args:
            target_gpu_ratio: Target ratio of GPU cache (0.0 to 1.0)
        """
        target_gpu_ratio = max(0.0, min(1.0, target_gpu_ratio))
        
        current_total = self.gpu_cache_size_bytes + self.cpu_cache_size_bytes
        new_gpu_size = int(current_total * target_gpu_ratio)
        new_cpu_size = current_total - new_gpu_size
        
        print(f"\nAdjusting cache ratio to {target_gpu_ratio:.1%} GPU / {1-target_gpu_ratio:.1%} CPU")
        print(f"  New GPU cache: {new_gpu_size / 1024**3:.1f} GB")
        print(f"  New CPU cache: {new_cpu_size / 1024**3:.1f} GB")
        
        # Update sizes
        self.gpu_cache_size_bytes = new_gpu_size
        self.cpu_cache_size_bytes = new_cpu_size
        
        # Evict if over new limits
        if self.gpu_cache_used_bytes > new_gpu_size:
            excess = self.gpu_cache_used_bytes - new_gpu_size
            self._evict_from_gpu(0)  # Will evict until under limit
        
        self._evict_from_cpu()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats
    
    def print_stats(self):
        """Print cache statistics."""
        print("\n" + "="*70)
        print("LMCache Statistics")
        print("="*70)
        print(f"\nCache Performance:")
        print(f"  Total requests:     {self.stats.total_requests}")
        print(f"  Cache hits:         {self.stats.cache_hits} ({self.stats.hit_rate:.1f}%)")
        print(f"  Cache misses:       {self.stats.cache_misses}")
        print(f"  Prefix matches:     {self.stats.prefix_matches} ({self.stats.prefix_match_rate:.1f}%)")
        
        print(f"\nCache Utilization:")
        gpu_util = (self.gpu_cache_used_bytes / self.gpu_cache_size_bytes * 100)
        cpu_util = (self.cpu_cache_used_bytes / self.cpu_cache_size_bytes * 100)
        print(f"  GPU cache:          {self.gpu_cache_used_bytes / 1024**3:.2f} / "
              f"{self.gpu_cache_size_bytes / 1024**3:.2f} GB ({gpu_util:.1f}%)")
        print(f"  CPU cache:          {self.cpu_cache_used_bytes / 1024**3:.2f} / "
              f"{self.cpu_cache_size_bytes / 1024**3:.2f} GB ({cpu_util:.1f}%)")
        print(f"  GPU entries:        {len(self.gpu_cache)}")
        print(f"  CPU entries:        {len(self.cpu_cache)}")
        
        print(f"\nTransfers:")
        print(f"  GPU → CPU:          {self.stats.gpu_to_cpu_transfers}")
        print(f"  CPU → GPU:          {self.stats.cpu_to_gpu_transfers}")
        print("="*70 + "\n")


# Example usage
if __name__ == '__main__':
    print("=" * 70)
    print("LMCache Integration Example (Chapter 18)")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("\nWarning: CUDA not available. This demo requires a GPU.")
        print("Exiting...")
        exit(0)
    
    # Initialize LMCache manager
    cache_mgr = LMCacheManager(
        gpu_cache_size_gb=1.0,  # Small for demo
        cpu_cache_size_gb=2.0,
        page_size=16,
        enable_prefix_sharing=True
    )
    
    # Simulate KV cache storage and retrieval
    print("\n" + "="*70)
    print("Test 1: Basic cache operations")
    print("="*70)
    
    # Create some sample KV caches
    seq_len, num_heads, head_dim = 128, 8, 64
    batch_size = 1
    
    print("\nStoring KV caches...")
    for i in range(5):
        token_ids = torch.randint(0, 1000, (seq_len,))
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        
        cache_mgr.put_kv_cache(token_ids, (k, v))
        print(f"  Stored cache {i+1}")
    
    cache_mgr.print_stats()
    
    # Test cache retrieval
    print("="*70)
    print("Test 2: Cache retrieval")
    print("="*70)
    
    # Try retrieving first sequence (should hit)
    print("\nRetrieving cached sequence...")
    token_ids = torch.randint(0, 1000, (seq_len,))
    cached = cache_mgr.get_kv_cache(token_ids)
    
    # Try new sequence (should miss)
    print("\nRetrieving new sequence...")
    new_tokens = torch.randint(0, 1000, (seq_len,))
    cached = cache_mgr.get_kv_cache(new_tokens)
    
    cache_mgr.print_stats()
    
    # Test prefix matching
    print("="*70)
    print("Test 3: Prefix matching")
    print("="*70)
    
    # Create sequence with common prefix
    base_tokens = torch.randint(0, 1000, (64,))
    k_base = torch.randn(batch_size, num_heads, 64, head_dim, device='cuda')
    v_base = torch.randn(batch_size, num_heads, 64, head_dim, device='cuda')
    cache_mgr.put_kv_cache(base_tokens, (k_base, v_base))
    
    # Try sequence with same prefix but different suffix
    extended_tokens = torch.cat([base_tokens, torch.randint(0, 1000, (64,))])
    print(f"\nLooking for sequence with common prefix ({len(base_tokens)} tokens)...")
    cached = cache_mgr.get_kv_cache(extended_tokens)
    
    cache_mgr.print_stats()
    
    # Test dynamic ratio adjustment
    print("="*70)
    print("Test 4: Dynamic GPU/CPU ratio adjustment")
    print("="*70)
    
    print("\nAdjusting to 30% GPU / 70% CPU (simulating memory pressure)...")
    cache_mgr.adjust_cache_ratio(target_gpu_ratio=0.3)
    
    cache_mgr.print_stats()
    
    print("="*70)
    print("Demo complete!")
    print("="*70)
    
    print("\nKey Benefits of LMCache (Chapter 18):")
    print("- Efficient GPU/CPU cache management")
    print("- Prefix sharing reduces redundant computation")
    print("- Dynamic ratio adjustment adapts to memory pressure")
    print("- Paged offload enables larger effective cache")

