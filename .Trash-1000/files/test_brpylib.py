#!/usr/bin/env python3
"""
Test brpylib NEV Reading
========================

Test script to verify brpylib functionality with NEV files.
"""

import numpy as np
import os
import sys

def test_brpylib():
    """Test brpylib functionality"""
    try:
        from brpylib import NevFile
        print("✓ brpylib imported successfully")
        
        nev_path = "/media/ubuntu/sda/Monkey/TVSD/monkeyF/20240112/Block_1/NSP-instance1_B001.nev"
        
        if not os.path.exists(nev_path):
            print(f"✗ NEV file not found: {nev_path}")
            return False
        
        print(f"✓ NEV file found: {nev_path}")
        
        # Try to open the NEV file
        print("Opening NEV file...")
        file = NevFile(nev_path)
        print("✓ NEV file opened successfully")
        
        # Try to get some basic information
        print("Getting file information...")
        
        # Try to get digital events
        try:
            digital_events = file.getdata('digitalserial')
            if digital_events is not None:
                print(f"✓ Found {len(digital_events)} digital events")
                
                # Show first few events
                if len(digital_events) > 0:
                    print("\nFirst 5 digital events:")
                    for i in range(min(5, len(digital_events))):
                        event = digital_events[i]
                        print(f"  Event {i}: Timestamp={event['TimeStamp']}, Data={event['UnparsedData']}")
                
                # Find stimulus events (bit 0 = stimulus marker)
                stim_mask = (digital_events['UnparsedData'] & 1) > 0
                stim_timestamps = digital_events['TimeStamp'][stim_mask]
                
                print(f"\n✓ Found {len(stim_timestamps)} stimulus events")
                
                if len(stim_timestamps) > 0:
                    print("First 5 stimulus timestamps:")
                    for i in range(min(5, len(stim_timestamps))):
                        print(f"  Stimulus {i}: {stim_timestamps[i]}")
                
                return True
            else:
                print("✗ No digital events found")
                return False
                
        except Exception as e:
            print(f"✗ Error getting digital events: {e}")
            return False
            
    except ImportError as e:
        print(f"✗ Failed to import brpylib: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Main function"""
    print("Testing brpylib NEV reading functionality")
    print("=" * 50)
    
    success = test_brpylib()
    
    if success:
        print("\n✓ brpylib test completed successfully!")
        print("You can now use the nev_brpy_extractor.py script")
    else:
        print("\n✗ brpylib test failed")
        print("Please check your brpylib installation")

if __name__ == "__main__":
    main()
