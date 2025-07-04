import os
import uuid
import json
import logging
import sys
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS # type: ignore
import cv2
import tempfile
import time
from werkzeug.utils import secure_filename
import threading
import shutil
from rembg import remove, new_session
import torch
import numpy as np

app = Flask(__name__)
CORS(app)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/app.log')
    ]
)
logger = logging.getLogger(__name__)


UPLOAD_FOLDER = '/tmp/uploads'
PROCESSED_FOLDER = '/tmp/processed'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'webm'}
API_KEY = "GPukTcc2FXcAo32U6j6y5rOK8LJW5QAf"


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

progress_data = {}

def setup_gpu():
    """Setup GPU and log system information"""
    logger.info("=== System Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        logger.warning("CUDA not available - running on CPU")
    

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        logger.info(f"ONNX Runtime providers: {providers}")
        if 'CUDAExecutionProvider' in providers:
            logger.info("CUDA provider available for ONNX Runtime")
        else:
            logger.warning("CUDA provider NOT available for ONNX Runtime")
    except ImportError:
        logger.error("ONNX Runtime not installed")
    
    logger.info("=== End System Information ===")


setup_gpu()


try:
    logger.info("Initializing rembg session with GPU support...")

    rembg_session = new_session('u2net', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    logger.info("rembg session initialized successfully with GPU support")
except Exception as e:
    logger.error(f"Failed to initialize rembg with GPU, falling back to CPU: {e}")
    try:
        rembg_session = new_session('u2net')
        logger.info("rembg session initialized with CPU fallback")
    except Exception as e2:
        logger.error(f"Failed to initialize rembg session: {e2}")
        rembg_session = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remove_background_from_video(input_video_path, output_video_path, progress_callback):
    """Remove background from video using rembg with GPU acceleration"""
    logger.info(f"Starting video processing: {input_video_path} -> {output_video_path}")
    
    if rembg_session is None:
        raise Exception("rembg session not initialized")
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {input_video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video properties: {width}x{height}, {fps} fps, {frame_count} frames")
    

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), True)
    
    if not out.isOpened():
        logger.error("Failed to open output video writer")
        raise Exception("Could not create output video file")
    
    processed_frames = 0
    start_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video reached")
                break
            
            try:
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                output_rgba = remove(frame_rgb, session=rembg_session)
                
                if output_rgba.shape[2] == 4:  
                    background = np.ones((height, width, 3), dtype=np.uint8) * 255
                    
                    alpha = output_rgba[:, :, 3:4] / 255.0
                    
                    output_rgb = output_rgba[:, :, :3] * alpha + background * (1 - alpha)
                    output_bgr = cv2.cvtColor(output_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                else:
                    output_bgr = cv2.cvtColor(output_rgba, cv2.COLOR_RGB2BGR)
                
                success = out.write(output_bgr)
                if not success:
                    logger.error(f"Failed to write frame {processed_frames}")
                
                processed_frames += 1
                progress_percentage = (processed_frames / frame_count) * 100
                
                if processed_frames % 10 == 0:
                    elapsed_time = time.time() - start_time
                    fps_current = processed_frames / elapsed_time if elapsed_time > 0 else 0
                    logger.info(f"Processed {processed_frames}/{frame_count} frames ({progress_percentage:.1f}%), Current FPS: {fps_current:.1f}")
                
                progress_callback(progress_percentage)
                
            except Exception as frame_error:
                logger.error(f"Error processing frame {processed_frames}: {frame_error}")
                out.write(frame)
                processed_frames += 1
                progress_percentage = (processed_frames / frame_count) * 100
                progress_callback(progress_percentage)
                
    except Exception as e:
        logger.error(f"Error during video processing: {e}")
        raise e
    finally:
        cap.release()
        out.release()
        total_time = time.time() - start_time
        logger.info(f"Video processing completed: {processed_frames} frames in {total_time:.2f}s (avg {processed_frames/total_time:.1f} fps)")

@app.before_request
def verify_api_key():
    if request.path == '/progress':
        return
        
    api_key = request.headers.get('X-Api-Key')
    if api_key != API_KEY:
        logger.warning(f"Unauthorized access attempt from {request.remote_addr}")
        return jsonify({'error': 'Unauthorized'}), 401

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and validation"""
    logger.info(f"Upload request from {request.remote_addr}")
    
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'success': False, 'error': 'No file part'}), 400
        
    file = request.files['file']
    user_id = request.headers.get('X-User-ID', 'unknown')
    
    logger.info(f"Upload request from user: {user_id}, filename: {file.filename}")
    
    if file.filename == '':
        logger.error("No file selected")
        return jsonify({'success': False, 'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        logger.error(f"File type not allowed: {file.filename}")
        return jsonify({'success': False, 'error': 'File type not allowed'}), 400
    
    try:
        unique_id = uuid.uuid4().hex
        filename = f"{user_id}_{unique_id}_{secure_filename(file.filename)}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        logger.info(f"Saving file to: {file_path}")
        
        file.save(file_path)
        
        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
        
        if file_size > 100 * 1024 * 1024:  # 100MB
            os.remove(file_path)
            logger.error("File too large")
            return jsonify({'success': False, 'error': 'File too large'}), 400
            
        # Check duration
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            os.remove(file_path)
            logger.error("Could not open video file")
            return jsonify({'success': False, 'error': 'Could not open video file'}), 400
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        logger.info(f"Video duration: {duration:.2f} seconds")
        
        if duration > 300:  # 5 minutes max
            os.remove(file_path)
            logger.error("Video too long")
            return jsonify({'success': False, 'error': 'Video too long (max 5 minutes)'}), 400
        
        progress_data[filename] = {
            'status': 'uploaded',
            'progress': 0,
            'message': 'Upload complete'
        }
        
        logger.info(f"Upload successful: {filename}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'duration': int(duration),
            'original_name': file.filename
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

def process_video_background(filename, user_id):
    """Background removal processing function"""
    logger.info(f"Starting background processing for {filename} (user: {user_id})")
    
    try:
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        output_filename = f"processed_{filename}"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        
        logger.info(f"Processing: {input_path} -> {output_path}")
        
        progress_data[filename] = {
            'status': 'processing',
            'progress': 5,
            'message': 'Starting background removal'
        }
        
        def update_progress(progress_percentage):
            progress_data[filename] = {
                'status': 'processing',
                'progress': min(99, int(progress_percentage)),
                'message': f'Processing: {progress_percentage:.1f}% complete'
            }
        
        # Process video using the remove_background_from_video function
        remove_background_from_video(input_path, output_path, update_progress)
        
        # Mark as complete
        progress_data[filename] = {
            'status': 'complete',
            'progress': 100,
            'message': 'Processing complete',
            'output_path': output_path
        }
        
        logger.info(f"Processing completed successfully for {filename}")
        
        # Clean up original upload
        try:
            os.remove(input_path)
            logger.info(f"Cleaned up original file: {input_path}")
        except Exception as cleanup_error:
            logger.warning(f"Could not clean up original file: {cleanup_error}")
            
    except Exception as e:
        logger.error(f"Processing error for {filename}: {str(e)}", exc_info=True)
        progress_data[filename] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}'
        }

@app.route('/process', methods=['POST'])
def process_video():
    """Start video processing"""
    logger.info(f"Process request from {request.remote_addr}")
    
    try:
        data = request.get_json()
        if not data:
            logger.error("No data provided in process request")
            return jsonify({'error': 'No data provided'}), 400
            
        filename = data.get('filename')
        user_id = data.get('user_id', 'unknown')
        
        logger.info(f"Process request: filename={filename}, user_id={user_id}")
        
        if not filename:
            logger.error("No filename provided")
            return jsonify({'error': 'Filename required'}), 400
            
        # Check if file exists
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
            
        processing_thread = threading.Thread(
            target=process_video_background,
            args=(filename, user_id)
        )
        processing_thread.start()
        logger.info(f"Started processing thread for {filename}")
        
        max_wait = 600  # 10 minutes max wait
        wait_time = 0
        sleep_interval = 1
        
        logger.info(f"Polling for completion (max {max_wait}s)")
        
        while wait_time < max_wait:
            progress = progress_data.get(filename, {})
            status = progress.get('status')
            
            if status == 'complete':
                logger.info(f"Processing completed for {filename}")
                output_path = progress.get('output_path')
                if os.path.exists(output_path):
                    return send_file(output_path, as_attachment=True, 
                                    download_name=f"processed_{filename}")
                else:
                    logger.error(f"Output file not found: {output_path}")
                    return jsonify({'error': 'Output file not found'}), 500
                    
            elif status == 'error':
                error_msg = progress.get('message', 'Unknown error')
                logger.error(f"Processing failed for {filename}: {error_msg}")
                return jsonify({'error': error_msg}), 500
                
            time.sleep(sleep_interval)
            wait_time += sleep_interval
            
        logger.error(f"Processing timeout for {filename} after {max_wait}s")
        return jsonify({'error': 'Processing timeout'}), 408
        
    except Exception as e:
        logger.error(f"Process request error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/progress', methods=['GET'])
def progress_stream():
    """Server-sent events endpoint for progress updates"""
    def generate():
        last_data = {}
        
        while True:
            current_data = progress_data.copy()
            
            if current_data != last_data:
                data_str = json.dumps(current_data)
                yield f"data: {data_str}\n\n"
                last_data = current_data.copy()
                
            time.sleep(1)
    
    return Response(generate(), content_type='text/event-stream')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

@app.route('/gpu-status', methods=['GET'])
def gpu_status():
    """GPU status endpoint"""
    status = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'rembg_session_initialized': rembg_session is not None
    }
    
    if torch.cuda.is_available():
        status['gpus'] = []
        for i in range(torch.cuda.device_count()):
            gpu_info = {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_cached': torch.cuda.memory_reserved(i)
            }
            status['gpus'].append(gpu_info)
    
    return jsonify(status)

if __name__ == '__main__':
    port = 5550
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)