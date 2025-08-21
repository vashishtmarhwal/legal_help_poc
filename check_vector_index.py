#!/usr/bin/env python3
"""
Script to check if Vector Search index has any embeddings/datapoints
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.services.vertex_vector_search_service import vertex_vector_search_service
from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import FindNeighborsRequest
import numpy as np

async def check_index_embeddings():
    """Check if the Vector Search index has any embeddings"""
    
    print("CHECK: Checking Vector Search Index for Embeddings...")
    print("=" * 60)
    
    # Check service status
    status = await vertex_vector_search_service.check_vector_search_status()
    
    print("STATUS: Index Status:")
    for key, value in status.items():
        icon = "SUCCESS:" if value else "ERROR:"
        print(f"   {icon} {key}: {value}")
    print()
    
    if not status["index_exists"]:
        print("ERROR: Index doesn't exist - cannot check for embeddings")
        return False
    
    # Try to check if we can perform a simple query
    print("🧪 Testing Index with Sample Query...")
    try:
        # Generate a test embedding
        test_texts = ["test query for checking if index has embeddings"]
        test_embeddings = await vertex_vector_search_service._generate_embeddings(test_texts)
        test_embedding = test_embeddings[0]
        
        # Try to query the index with this embedding
        # This will help us determine if there are any datapoints
        from google.cloud.aiplatform_v1 import MatchServiceClient
        from google.cloud.aiplatform_v1.types import FindNeighborsRequest, FindNeighborsResponse
        
        # Initialize the client
        client = MatchServiceClient()
        
        # Prepare the request
        index_endpoint_name = f"projects/{vertex_vector_search_service.project_id}/locations/{vertex_vector_search_service.location}/indexEndpoints/{vertex_vector_search_service.endpoint_id}"
        
        # Create a neighbor query
        neighbor_query = FindNeighborsRequest.Query(
            datapoint=FindNeighborsRequest.Query.Datapoint(
                feature_vector=test_embedding
            ),
            neighbor_count=5
        )
        
        request = FindNeighborsRequest(
            index_endpoint=index_endpoint_name,
            deployed_index_id=vertex_vector_search_service.deployed_index_id,
            queries=[neighbor_query]
        )
        
        # Execute the query
        response = client.find_neighbors(request=request)
        
        # Check results
        if response.nearest_neighbors and len(response.nearest_neighbors) > 0:
            neighbors = response.nearest_neighbors[0].neighbors
            print(f"SUCCESS: Index has embeddings! Found {len(neighbors)} similar vectors")
            
            if neighbors:
                print("INFO: Sample Results:")
                for i, neighbor in enumerate(neighbors[:3]):  # Show first 3
                    print(f"   {i+1}. ID: {neighbor.datapoint.datapoint_id}")
                    print(f"      Distance: {neighbor.distance:.4f}")
            
            return True
        else:
            print("ERROR: Index exists but appears to be empty (no neighbors found)")
            return False
            
    except Exception as e:
        print(f"ERROR: Error querying index: {e}")
        
        # If querying fails, the index might be empty or there might be permission issues
        if "NOT_FOUND" in str(e) or "empty" in str(e).lower():
            print("TIP: This suggests the index has no embeddings uploaded yet")
            return False
        else:
            print("WARNING:  Could not determine if index has embeddings due to error")
            return None

async def main():
    """Main function"""
    try:
        has_embeddings = await check_index_embeddings()
        
        print()
        print("TARGET: SUMMARY:")
        print("=" * 60)
        
        if has_embeddings is True:
            print("SUCCESS: Your Vector Search index HAS embeddings!")
            print("   You can perform similarity searches and queries.")
        elif has_embeddings is False:
            print("ERROR: Your Vector Search index is EMPTY!")
            print("   You need to upload some documents first.")
            print()
            print("NOTE: To upload documents:")
            print("   curl -X POST http://localhost:8000/upload-to-vertex-vector-search/ \\")
            print("        -F 'files=@your-document.pdf'")
        else:
            print("WARNING:  Could not determine index status")
            print("   Check your permissions and try again.")
        
        return has_embeddings is True
        
    except Exception as e:
        print(f"💥 Error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)