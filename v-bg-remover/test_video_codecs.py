#!/usr/bin/env python3
import cv2
import numpy as np
import tempfile
import os

def test_video_codecs():
    
    print("Testing video codec support...")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Test parameters
    width, height = 640, 480
    fps = 30
    frames_to_write = 5
    
    codecs_to_test = [
        ('mp4v', 'MP4V', '.mp4'),
        ('XVID', 'XVID', '.avi'),
        ('MJPG', 'MJPG', '.avi'),
        ('H264', 'H264', '.mp4'),
        ('avc1', 'avc1', '.mp4'),
        ('X264', 'X264', '.mp4'),
        ('FMP4', 'FMP4', '.mp4'),
        ('DIV3', 'DIV3', '.avi'),
        ('DIVX', 'DIVX', '.avi'),
    ]
    
    working_codecs = []
    
    for codec_name, fourcc_str, extension in codecs_to_test:
        try:
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            print(f"\nTesting codec: {codec_name} ({fourcc_str})")
            
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height), True)
            
            if out.isOpened():
                print(f"✓ Codec {codec_name} initialized successfully")
                
                for i in range(frames_to_write):
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    frame[:, :, 0] = (i * 50) % 256  # Blue channel
                    frame[:, :, 1] = (i * 100) % 256  # Green channel
                    frame[:, :, 2] = (i * 150) % 256  # Red channel
                    
                    success = out.write(frame)
                    if not success:
                        print(f"✗ Failed to write frame {i} with codec {codec_name}")
                        break
                else:
                    print(f"✓ Successfully wrote {frames_to_write} frames with codec {codec_name}")
                    working_codecs.append((codec_name, fourcc_str, extension))
                
                out.release()
            else:
                print(f"✗ Failed to initialize codec {codec_name}")
            
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
                
        except Exception as e:
            print(f"✗ Exception with codec {codec_name}: {e}")
    
    print(f"\n{'='*50}")
    print("SUMMARY:")
    print(f"{'='*50}")
    
    if working_codecs:
        print("Working codecs:")
        for codec_name, fourcc_str, extension in working_codecs:
            print(f"  - {codec_name} ({fourcc_str}) -> {extension}")
    else:
        print("No working codecs found!")
        
    # Test auto codec selection
    print(f"\nTesting auto codec selection...")
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        out = cv2.VideoWriter(temp_path, -1, fps, (width, height), True)
        if out.isOpened():
            print("✓ Auto codec selection works")
            out.release()
        else:
            print("✗ Auto codec selection failed")
        
        os.unlink(temp_path)
    except Exception as e:
        print(f"✗ Auto codec selection exception: {e}")

if __name__ == "__main__":
    test_video_codecs()